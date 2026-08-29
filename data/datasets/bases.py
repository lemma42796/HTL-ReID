from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

_IMAGE_READ_MAX_ATTEMPTS = 3
_IMAGE_READ_RETRY_DELAY_SECONDS = 0.05


def _open_image(img_path, convert_rgb=False):
    """Load one image with bounded retries and close its file handle."""
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))

    last_error = None
    for attempt in range(1, _IMAGE_READ_MAX_ATTEMPTS + 1):
        try:
            with Image.open(img_path) as source:
                if convert_rgb:
                    return source.convert('RGB')
                # PIL decodes lazily. copy() forces the pixels to be loaded
                # before the context manager closes the underlying file.
                return source.copy()
        except OSError as error:
            last_error = error
            if attempt < _IMAGE_READ_MAX_ATTEMPTS:
                time.sleep(_IMAGE_READ_RETRY_DELAY_SECONDS)

    raise IOError(
        "failed to read '{}' after {} attempts".format(
            img_path, _IMAGE_READ_MAX_ATTEMPTS)
    ) from last_error


def read_image(img_list):
    """Read one packed image or a list of modality image paths."""
    if isinstance(img_list, str):
        img_path = img_list
        img = _open_image(img_path, convert_rgb=True)
        #判断图像的宽度是256的几倍，如果是三倍，则进行下面的代码，否则只裁剪前两个
        img3 = [img.crop((256 * i, 0, 256 * (i + 1), 128)) for i in range(img.size[0] // 256)]
    else:
        img3 = [_open_image(img_path) for img_path in img_list]
    return img3


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img3 = read_image(img_path)

        if self.transform is not None:
            img = [self.transform(img) for img in img3]
        if isinstance(img_path, str):
            return img, pid, camid, trackid, img_path.split('/')[-1]
        else:
            return img, pid, camid, trackid, img_path[0].split('/')[-1]
