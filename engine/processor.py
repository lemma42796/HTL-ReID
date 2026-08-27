import logging
import hashlib
import json
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


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _reconstruction_history(model):
    recon = getattr(_unwrap_model(model), 'CROSS_MODAL_RECON', None)
    if recon is None or not hasattr(recon, 'target_history'):
        return ()
    return recon.target_history()


def _sequence_digest(values):
    payload = ','.join(str(value) for value in values).encode('ascii')
    return hashlib.sha256(payload).hexdigest()


def _write_determinism_trace(path, trace):
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as handle:
        json.dump(trace, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    os.replace(temp_path, path)


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
    save_best_checkpoint = cfg.SOLVER.SAVE_BEST_CHECKPOINT
    save_periodic_checkpoints = cfg.SOLVER.SAVE_PERIODIC_CHECKPOINTS
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    train_epochs = int(getattr(cfg.SOLVER, 'TRAIN_EPOCHS', 0))
    run_epochs = train_epochs if train_epochs > 0 else epochs
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("HTL-ReID.train")
    logger.info('start training')
    if run_epochs != epochs:
        logger.info('diagnostic early stop enabled: training %d/%d scheduler epochs',
                    run_epochs, epochs)
    # Create SummaryWriter
    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs'))

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)
    trace_path = os.path.join(cfg.OUTPUT_DIR, 'determinism_trace.json')
    determinism_trace = {
        'seed': int(cfg.SOLVER.SEED),
        'runtime': {
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cublas_workspace_config': os.environ.get(
                'CUBLAS_WORKSPACE_CONFIG', ''),
            'pythonhashseed': os.environ.get('PYTHONHASHSEED', ''),
        },
        'epochs': [],
    }
    _write_determinism_trace(trace_path, determinism_trace)
    evaluator_m = _make_evaluator(cfg, num_query)
    evaluator_m.reset()


    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler()
    scheduler_in_epochs = getattr(scheduler, 't_in_epochs', True)
    updates_per_epoch = len(train_loader)

    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, run_epochs + 1):
        target_history_start = len(_reconstruction_history(model))
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
            optimizer.zero_grad(set_to_none=True)
            optimizer_center.zero_grad(set_to_none=True)
            img = {'RGB': img['RGB'].to(device, non_blocking=True),
                   'NI': img['NI'].to(device, non_blocking=True),
                   'TI': img['TI'].to(device, non_blocking=True)}
            target = vid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)
            target_view = target_view.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view, img_path=imgpath,
                               writer=writer, epoch=epoch)
                loss = _compute_train_loss(cfg, output, loss_fn, target, target_cam, epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if output[0].shape[0] != target.shape[0]:
                target = torch.cat((target, target), dim=0)
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.detach(), img['RGB'].shape[0])
            acc_meter.update(acc.detach(), 1)

            if (n_iter + 1) % log_period == 0:
                # One host synchronization per logging interval instead of
                # multiple loss.item()/cuda.synchronize calls per iteration.
                loss_avg, acc_avg = torch.stack([
                    loss_meter.avg.float(), acc_meter.avg.float()
                ]).cpu().tolist()
                global_step = (epoch - 1) * updates_per_epoch + n_iter + 1
                writer.add_scalar('Loss/train', loss_avg, global_step)
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_avg, acc_avg, optimizer.param_groups[0]['lr']))

        # Synchronize once so epoch throughput includes all queued GPU work.
        torch.cuda.synchronize()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if save_periodic_checkpoints and checkpoint_period > 0 and epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        should_eval = (eval_period > 0 and epoch % eval_period == 0) or epoch == run_epochs
        eval_metrics = None
        if should_eval:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('!!!Mutil-Modal Testing!!!')
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = {'RGB': img['RGB'].to(device, non_blocking=True),
                                   'NI': img['NI'].to(device, non_blocking=True),
                                   'TI': img['TI'].to(device, non_blocking=True)}
                            camids = camids.to(device, non_blocking=True)
                            target_view = target_view.to(device, non_blocking=True)
                            feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator_m.update((feat, vid, camid, target_view, _))
                            else:
                                evaluator_m.update((feat, vid, camid))


                    # 计算多模态性能
                    cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
                    _log_eval_results(logger, cmc, mAP, epoch=epoch)
                    eval_metrics = {
                        'mAP': float(mAP.item()),
                        'Rank1': float(cmc[0].item()),
                        'Rank5': float(cmc[4].item()),
                        'Rank10': float(cmc[9].item()),
                    }
                    writer.add_scalar('MM/mAP', mAP.item(), epoch)
                    writer.add_scalar('MM/Rank-1', cmc[0].item(), epoch)

                    if mAP >= best_index['mAP']:
                        best_index['mAP'] = mAP
                        best_index['Rank-1'] = cmc[0]
                        best_index['Rank-5'] = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        if save_best_checkpoint:
                            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("Best Multi-Modal mAP: {:.2%}".format(best_index['mAP']))
                    logger.info("Best Multi-Modal Rank-1: {:.2%}".format(best_index['Rank-1']))
                    logger.info("Best Multi-Modal Rank-5: {:.2%}".format(best_index['Rank-5']))
                    logger.info("Best Multi-Modal Rank-10: {:.2%}".format(best_index['Rank-10']))
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            else:
                model.eval()
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('!!!Mutil-Modal Testing!!!')
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {'RGB': img['RGB'].to(device, non_blocking=True),
                               'NI': img['NI'].to(device, non_blocking=True),
                               'TI': img['TI'].to(device, non_blocking=True)}
                        camids = camids.to(device, non_blocking=True)
                        target_view = target_view.to(device, non_blocking=True)
                        feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator_m.update((feat, vid, camid, target_view, _))
                        else:
                            evaluator_m.update((feat, vid, camid))

                # 计算多模态性能
                cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
                _log_eval_results(logger, cmc, mAP, epoch=epoch)
                eval_metrics = {
                    'mAP': float(mAP.item()),
                    'Rank1': float(cmc[0].item()),
                    'Rank5': float(cmc[4].item()),
                    'Rank10': float(cmc[9].item()),
                }
                writer.add_scalar('MM/mAP', mAP.item(), epoch)
                writer.add_scalar('MM/Rank-1', cmc[0].item(), epoch)


                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    if save_best_checkpoint:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                logger.info("Best Multi-Modal mAP: {:.2%}".format(best_index['mAP']))
                logger.info("Best Multi-Modal Rank-1: {:.2%}".format(best_index['Rank-1']))
                logger.info("Best Multi-Modal Rank-5: {:.2%}".format(best_index['Rank-5']))
                logger.info("Best Multi-Modal Rank-10: {:.2%}".format(best_index['Rank-10']))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        epoch_loss, epoch_acc = torch.stack([
            loss_meter.avg.float(), acc_meter.avg.float()
        ]).cpu().tolist()
        target_sequence = list(
            _reconstruction_history(model)[target_history_start:])
        sampler = getattr(train_loader, 'sampler', None)
        trace_entry = {
            'epoch': epoch,
            'sampler_epoch': getattr(sampler, 'last_epoch', None),
            'sampler_order_sha256': getattr(
                sampler, 'last_order_digest', None),
            'sampler_order_length': getattr(
                sampler, 'last_order_length', None),
            'reconstruction_targets': target_sequence,
            'reconstruction_targets_sha256': _sequence_digest(
                target_sequence),
            'train_loss': float(epoch_loss),
            'train_accuracy': float(epoch_acc),
            'evaluation': eval_metrics,
        }
        determinism_trace['epochs'].append(trace_entry)
        _write_determinism_trace(trace_path, determinism_trace)
        logger.info(
            'Determinism Trace - Epoch %d: sampler=%s recon=%s',
            epoch, trace_entry['sampler_order_sha256'],
            trace_entry['reconstruction_targets_sha256'])

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
                    img = {'RGB': img['RGB'].to(device, non_blocking=True),
                           'NI': img['NI'].to(device, non_blocking=True),
                           'TI': img['TI'].to(device, non_blocking=True)}
                    camids = camids.to(device, non_blocking=True)
                    target_view = target_view.to(device, non_blocking=True)
                    feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                    if cfg.DATASETS.NAMES == "MSVR310":
                        evaluator_m.update((feat, vid, camid, target_view, _))
                    else:
                        evaluator_m.update((feat, vid, camid))


            cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
            _log_eval_results(logger, cmc, mAP)

    else:
        model.eval()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('!!!Mutil-Modal Testing!!!')
        for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
            with torch.no_grad():
                img = {'RGB': img['RGB'].to(device, non_blocking=True),
                       'NI': img['NI'].to(device, non_blocking=True),
                       'TI': img['TI'].to(device, non_blocking=True)}
                camids = camids.to(device, non_blocking=True)
                target_view = target_view.to(device, non_blocking=True)
                feat = model(img, cam_label=camids, view_label=target_view, mode=1, img_path=_)
                if cfg.DATASETS.NAMES == "MSVR310":
                    evaluator_m.update((feat, vid, camid, target_view, _))
                else:
                    evaluator_m.update((feat, vid, camid))

        cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
        _log_eval_results(logger, cmc, mAP)
