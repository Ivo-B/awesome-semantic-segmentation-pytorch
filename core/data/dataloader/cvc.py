"""Prepare PTX dataset"""
import os
import torch
import numpy as np
import random

from PIL import Image
from .segbase import SegmentationDataset
import nibabel

class CVCSegmentation(SegmentationDataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'input'
    NUM_CLASS = 8

    def __init__(self, root, split='test', mode=None, transform=None, **kwargs):
        super(CVCSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please setup the dataset using ../datasets/ade20k.py"
        if mode == 'test':
            self.images = _get_ptx_pairs(root, split)
        else:
            self.images, self.masks = _get_ptx_pairs(root, split)
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # img = Image.open(self.images[index])

        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask_array = nibabel.load(self.masks[index][0]).get_fdata()

        mask = Image.fromarray(mask_array, mode='RGB')

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask_array)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask_array)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and to Tensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("Background", "Pneumothorax")



IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',\
                  '.PPM', '.bmp', '.BMP', '.nii.gz', '.tif', '.tiff', '.svs',\
                  '.mrxs', '.nii']

####################
# Files & IO
####################


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(data_path):
    images = []
    if type(data_path) == list:
        for dpath in data_path:
            assert os.path.isdir(dpath), '{:s} is not a valid directory'.format(dpath)
            for dir_path, _, fnames in sorted(os.walk(dpath)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        if 'mask' not in fname:
                            img_path = os.path.join(dir_path, fname)
                            images.append(img_path)
    else:
        assert os.path.isdir(data_path), '{:s} is not a valid directory'.format(data_path)
        for dir_path, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    if 'mask' not in fname:
                        img_path = os.path.join(dir_path, fname)
                        images.append(img_path)
    assert images, '{:s} has no valid image file'.format(data_path)
    return sorted(images)


def _get_ptx_pairs(folder, mode='train'):
    import csv
    if mode != 'test':
        # filter img without PTX
        data_list_mask = get_image_paths(r'E:\repos\CVCPlaceCheck\mask')

        #split data_list
        train_len = int(np.ceil(len(data_list_mask) * 0.8))
        img_paths = []
        mask_paths = []
        if mode == 'train':
            data_list = data_list_mask[:train_len]
        else:
            data_list = data_list_mask[train_len:]

        for d in data_list:
            # finding masks
            mask_names = [d]
            if os.path.isfile(d):
                img_paths.append(os.path.join(r'E:\datasets\UKEcentralVenousCatheter\Kaggel-NIH-subselection\CVCimg', os.path.basename(d).split('.ni')[0]))
                mask_paths.append(mask_names)
            else:
                print(d, os.path.join(r'E:\datasets\UKEcentralVenousCatheter\Kaggel-NIH-subselection\CVCimg', os.path.basename(d).split('.ni')[0] + '.png'))

        return img_paths, mask_paths
    else:
        root_folder = os.path.join(folder, r'test\images_dcm')
        data_list = get_image_paths(root_folder)
        img_paths = []
        for d in data_list:
            img_paths.append(d)
        return img_paths
        raise NotImplementedError('Need to implent test net load!')

if __name__ == '__main__':
    train_dataset = PTXSegmentation()
