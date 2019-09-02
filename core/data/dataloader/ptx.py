"""Prepare PTX dataset"""
import os
import torch
import numpy as np
import random

from PIL import Image
from .segbase import SegmentationDataset


class PTXSegmentation(SegmentationDataset):
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
    NUM_CLASS = 2

    def __init__(self, root, split='test', mode=None, transform=None, **kwargs):
        super(PTXSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(root, self.BASE_DIR)
        #assert os.path.exists(root), "Please setup the dataset using ../datasets/ade20k.py"
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

        #mask = Image.open(random.choice(self.masks[index]))
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
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
                        #if 'mask' not in fname:
                            img_path = os.path.join(dir_path, fname)
                            images.append(img_path)
    else:
        assert os.path.isdir(data_path), '{:s} is not a valid directory'.format(data_path)
        for dir_path, _, fnames in sorted(os.walk(data_path)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    #f 'mask' not in fname:
                        img_path = os.path.join(dir_path, fname)
                        images.append(img_path)
    assert images, '{:s} has no valid image file'.format(data_path)
    return sorted(images)


def _get_ptx_pairs(folder, mode='train'):
    import csv
    import cv2
    if mode != 'test':
        # Ordnerstruktur und/oder File in -> List[Image]
        data_list = get_image_paths(r'E:\repos\PTXSegmentationForReal\stage2_dataset\segData')
        imgID_dict = {}
        imgID_withMask = []
        # filter img without PTX
        with open(r'E:\repos\PTXSegmentationForReal\stage2_dataset\stage_2_train.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    if int(row[1].strip().split(' ')[0]) != -1:
                        imgID_withMask.append(row[0])
                line_count += 1

        # extract imgID from path
        neg_counter = 0
        debugging = False
        for d in data_list:
            if 'mask' in d:
                imgID = os.path.basename(d).split('_mask.png')[0]
            else:
                imgID = os.path.basename(d).split('.png')[0]
                # imgID = d.split('images_dcm')[1]
                # imgID = imgID.split('/')[1]
            if imgID.split('_')[-1] not in imgID_withMask:
                neg_counter += 1
                if neg_counter >= len(imgID_withMask) + len(imgID_withMask) * (1/3) and not imgID in imgID_dict.keys():
                #if neg_counter >= 0 and not imgID in imgID_dict.keys():
                    continue

            # imgID = imgID.split('case_')[1]
            if imgID.split('_')[-1] in imgID_withMask:
                if not imgID in imgID_dict.keys():
                    imgID_dict[imgID] = {}
                    imgID_dict[imgID]['patID'] = imgID
                if 'mask' not in [d][0]:

                    if os.path.isfile(d):
                        imgID_dict[imgID]['imageFilepath'] = d
                    else:
                        print(d)
                elif 'mask' in [d][0]:
                    # correct maks to 0,1
                    #img = cv2.imread(d, 0)
                    #if np.max(img) == 255:
                    #    img = img / 255
                    #    cv2.imwrite(d, img)

                    if os.path.isfile(d):
                        imgID_dict[imgID]['segmentationFilepath'] = d
                    else:
                        print(d)
            elif True:
                if not imgID in imgID_dict.keys():
                    imgID_dict[imgID] = {}
                    imgID_dict[imgID]['patID'] = imgID
                if 'mask' not in [d][0]:

                    if os.path.isfile(d):
                        imgID_dict[imgID]['imageFilepath'] = d
                    else:
                        print(d)
                elif 'mask' in [d][0]:
                    if os.path.isfile(d):
                        imgID_dict[imgID]['segmentationFilepath'] = d
                    else:
                        print(d)
    # if mode != 'test':
    #     root_folder = os.path.join(folder, r'train\images_dcm')
    #     data_list = get_image_paths(root_folder)
    #     imgID_withMask = []
    #     # filter img without PTX
    #     with open(r'E:\repos\PTXSegmentation\input\train-rle.csv') as csv_file:
    #         csv_reader = csv.reader(csv_file, delimiter=',')
    #         line_count = 0
    #         for row in csv_reader:
    #             if line_count > 0:
    #                 if int(row[1].strip().split(' ')[0]) != -1:
    #                     imgID_withMask.append(row[0])
    #             line_count += 1
    #     # extract imgID from path
    #     neg_counter = 0
    #     debugging = False
    #     for d in data_list:
    #         if 'mask' in d:
    #             imgID = os.path.basename(d).split('_mask.png')[0]
    #         else:
    #             imgID = os.path.basename(d).split('.png')[0]
    #             # imgID = d.split('images_dcm')[1]
    #             # imgID = imgID.split('/')[1]
    #         if imgID.split('_')[-1] not in imgID_withMask:
    #             neg_counter += 1
    #             # if neg_counter >= len(imgID_withMask) + len(imgID_withMask) * (1/3) and not imgID in imgID_dict.keys():
    #             if neg_counter >= 0 and not imgID in imgID_dict.keys():
    #                 continue
    #
    #         if imgID.split('_')[-1] in imgID_withMask:
    #             if not imgID in imgID_dict.keys():
    #                 imgID_dict[imgID] = {}
    #                 imgID_dict[imgID]['patID'] = imgID
    #             if 'mask' not in [d][0]:
    #
    #                 if os.path.isfile(d):
    #                     imgID_dict[imgID]['imageFilepath'] = d
    #                 else:
    #                     print(d)
    #             elif 'mask' in [d][0]:
    #                 # correct maks to 0,1
    #                 img = cv2.imread(d, 0)
    #                 if np.max(img) == 255:
    #                     img = img / 255
    #                     cv2.imwrite(d, img)
    #
    #                 if os.path.isfile(d):
    #                     imgID_dict[imgID]['segmentationFilepath'] = d
    #                 else:
    #                     print(d)
    #         elif True:
    #             if not imgID in imgID_dict.keys():
    #                 imgID_dict[imgID] = {}
    #                 imgID_dict[imgID]['patID'] = imgID
    #             if 'mask' not in [d][0]:
    #
    #                 if os.path.isfile(d):
    #                     imgID_dict[imgID]['imageFilepath'] = d
    #                 else:
    #                     print(d)
    #             elif 'mask' in [d][0]:
    #                 if os.path.isfile(d):
    #                     imgID_dict[imgID]['segmentationFilepath'] = d
    #                 else:
    #                     print(d)
        # data_list_tmp = []
        # for d in data_list:
        #     imgID = d.split('images_dcm')[1]
        #     imgID = imgID.split('\\')[1]
        #     if imgID in imgID_withMask:
        #         data_list_tmp.append(d)
        # data_list = data_list_tmp
        # #split data_list
        # train_len = int(np.ceil(len(data_list) * 0.8))
        # img_paths = []
        # mask_paths = []
        # if mode == 'train':
        #     data_list = data_list[:train_len]
        # else:
        #     data_list = data_list[train_len:]
        #
        # for d in data_list:
        #     # finding masks
        #     mask_name = os.path.join(os.path.dirname(d), 'mask_{}_0.png'.format(os.path.basename(d).split('.pn')[0]))
        #     mask_names = []
        #     counter = 1
        #     while os.path.isfile(mask_name):
        #         mask_names.append(mask_name)
        #         mask_name = os.path.join(os.path.dirname(d), 'mask_{}_{}.png'.format(os.path.basename(d).split('.pn')[0], counter))
        #         counter += 1
        #     if os.path.isfile(d):
        #         img_paths.append(d)
        #         mask_paths.append(mask_names)
        #     else:
        #         print(d, mask_name)
        #convert dict to list:
        img_paths = list()
        mask_paths = list()
        for k in imgID_dict.keys():
            img_paths.append(imgID_dict[k]['imageFilepath'])
            mask_paths.append(imgID_dict[k]['segmentationFilepath'])
        train_len = int(np.ceil(len(img_paths) * 0.85))
        if mode == 'train':
            img_paths = img_paths[:train_len]
            mask_paths = mask_paths[:train_len]
        else:
            img_paths = img_paths[train_len:]
            mask_paths = mask_paths[train_len:]
        return img_paths, mask_paths
    else:
        root_folder = os.path.join(r'E:\repos\PTXSegmentationForReal\stage2_dataset\testing_preFullSize')
        data_list = get_image_paths(root_folder)
        img_paths = []
        for d in data_list:
            img_paths.append(d)
        return img_paths


if __name__ == '__main__':
    train_dataset = PTXSegmentation()
