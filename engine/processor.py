import logging
import os
import time
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in ('yes', 'true', '1', 'on')
    return bool(value)


def _make_evaluator(cfg, num_query):
    reranking = _as_bool(cfg.TEST.RE_RANKING)
    if cfg.DATASETS.NAMES == "MSVR310":
        return R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=reranking)
    return R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=reranking)


def _log_eval_results(logger, cmc, mAP, epoch=None):
    if epoch is None:
        logger.info("Validation Results")
    else:
        logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))


def _stage_scale(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def _pair_loss_weight(cfg, pair_idx, num_pairs):
    if pair_idx == 0:
        return float(cfg.MODEL.FUSE_LOSS_WEIGHT)
    if cfg.MODEL.PART_BRANCH and pair_idx == num_pairs - 1:
        return float(cfg.MODEL.PART_LOSS_WEIGHT)
    return float(cfg.MODEL.BRANCH_LOSS_WEIGHT)


def _compute_train_loss(cfg, output, loss_fn, target, target_cam, epoch):
    pair_end = len(output) - 1 if len(output) % 2 == 1 else len(output)
    num_pairs = pair_end // 2
    loss = 0
    for pair_idx, i in enumerate(range(0, pair_end, 2)):
        loss_tmp = loss_fn(score=output[i], feat=output[i + 1],
                           target=target, target_cam=target_cam)
        loss = loss + _pair_loss_weight(cfg, pair_idx, num_pairs) * loss_tmp
    if len(output) % 2 == 1:
        aux_weight = float(cfg.MODEL.AUX_LOSS_WEIGHT) * _stage_scale(
            epoch, cfg.MODEL.AUX_WARMUP_EPOCHS)
        loss = loss + aux_weight * output[-1]
    return loss


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    save_checkpoints = cfg.SOLVER.SAVE_CHECKPOINTS
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("HTL-ReID.train")
    logger.info('start training')
    # Create SummaryWriter
    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs'))

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)
    evaluator_m = _make_evaluator(cfg, num_query)
    evaluator_m.reset()


    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler()
    scheduler_in_epochs = getattr(scheduler, 't_in_epochs', True)
    updates_per_epoch = len(train_loader)

    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        evaluator_m.reset()
        acc_meter.reset()
        if scheduler_in_epochs:
            scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, imgpath) in enumerate(train_loader):
            if not scheduler_in_epochs:
                num_updates = (epoch - 1) * updates_per_epoch + n_iter
                scheduler.step_update(num_updates)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view, img_path=imgpath,
                               writer=writer, epoch=epoch)
                loss = _compute_train_loss(cfg, output, loss_fn, target, target_cam, epoch)
            writer.add_scalar('Loss', loss.item(), epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if output[0].shape[0] != target.shape[0]:
                target = torch.cat((target, target), dim=0)
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler._get_lr(epoch))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, optimizer.param_groups[0]['lr']))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if save_checkpoints and checkpoint_period > 0 and epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('!!!Mutil-Modal Testing!!!')
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = {'RGB': img['RGB'].to(device),
                                   'NI': img['NI'].to(device),
                                   'TI': img['TI'].to(device)}
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator_m.update((feat, vid, camid, target_view, _))
                            else:
                                evaluator_m.update((feat, vid, camid))


                    # 计算多模态性能
                    cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
                    _log_eval_results(logger, cmc, mAP, epoch=epoch)
                    writer.add_scalar('MM/mAP', mAP.item(), epoch)
                    writer.add_scalar('MM/Rank-1', cmc[0].item(), epoch)

                    if mAP >= best_index['mAP']:
                        best_index['mAP'] = mAP
                        best_index['Rank-1'] = cmc[0]
                        best_index['Rank-5'] = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        if save_checkpoints:
                            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("Best Multi-Modal mAP: {:.2%}".format(best_index['mAP']))
                    logger.info("Best Multi-Modal Rank-1: {:.2%}".format(best_index['Rank-1']))
                    logger.info("Best Multi-Modal Rank-5: {:.2%}".format(best_index['Rank-5']))
                    logger.info("Best Multi-Modal Rank-10: {:.2%}".format(best_index['Rank-10']))
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    torch.cuda.empty_cache()

            else:
                model.eval()
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('!!!Mutil-Modal Testing!!!')
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {'RGB': img['RGB'].to(device),
                               'NI': img['NI'].to(device),
                               'TI': img['TI'].to(device)}
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator_m.update((feat, vid, camid, target_view, _))
                        else:
                            evaluator_m.update((feat, vid, camid))

                # 计算多模态性能
                cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
                _log_eval_results(logger, cmc, mAP, epoch=epoch)
                writer.add_scalar('MM/mAP', mAP.item(), epoch)
                writer.add_scalar('MM/Rank-1', cmc[0].item(), epoch)


                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    if save_checkpoints:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                logger.info("Best Multi-Modal mAP: {:.2%}".format(best_index['mAP']))
                logger.info("Best Multi-Modal Rank-1: {:.2%}".format(best_index['Rank-1']))
                logger.info("Best Multi-Modal Rank-5: {:.2%}".format(best_index['Rank-5']))
                logger.info("Best Multi-Modal Rank-10: {:.2%}".format(best_index['Rank-10']))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                torch.cuda.empty_cache()


    writer.close()
    return None


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("HTL-ReID.test")
    logger.info("Enter inferencing")

    evaluator_m = _make_evaluator(cfg, num_query)
    evaluator_m.reset()


    if cfg.MODEL.DIST_TRAIN:
        if dist.get_rank() == 0:
            model.eval()
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('!!!Mutil-Modal Testing!!!')
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = {'RGB': img['RGB'].to(device),
                           'NI': img['NI'].to(device),
                           'TI': img['TI'].to(device)}
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                    if cfg.DATASETS.NAMES == "MSVR310":
                        evaluator_m.update((feat, vid, camid, target_view, _))
                    else:
                        evaluator_m.update((feat, vid, camid))


            torch.cuda.empty_cache()
            cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
            _log_eval_results(logger, cmc, mAP)

    else:
        model.eval()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('!!!Mutil-Modal Testing!!!')
        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
            with torch.no_grad():
                img = {'RGB': img['RGB'].to(device),
                       'NI': img['NI'].to(device),
                       'TI': img['TI'].to(device)}
                camids = camids.to(device)
                target_view = target_view.to(device)
                feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                if cfg.DATASETS.NAMES == "MSVR310":
                    evaluator_m.update((feat, vid, camid, target_view, _))
                else:
                    evaluator_m.update((feat, vid, camid))

        torch.cuda.empty_cache()
        cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
        _log_eval_results(logger, cmc, mAP)
