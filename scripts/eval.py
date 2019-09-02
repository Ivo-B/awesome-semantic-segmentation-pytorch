from __future__ import print_function

import argparse
import os
import sys
import platform

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
import core.utils.options as option

import util
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# from train import parse_args


class Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.run_config = config['run_config']
        self.optim_config = config['optim_config']
        self.data_config = config['data_config']
        self.model_config = config['model_config']

        self.device = torch.device(self.run_config["device"])

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(self.data_config['dataset_name'], root=self.data_config['dataset_root'], split='test', mode='test', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, self.run_config['distributed'])
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=10, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=4,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if self.run_config['distributed'] else nn.BatchNorm2d
        self.model = get_segmentation_model(model=self.model_config['model'],
                                            dataset=self.data_config['dataset_name'],
                                            backbone=self.model_config['backbone'],
                                            aux=self.optim_config['aux'],
                                            jpu=self.model_config['jpu'],
                                            norm_layer=BatchNorm2d,
                                            root=run_config['path']['eval_model_root'],
                                            pretrained=run_config['eval_model'],
                                            pretrained_base=False,
                                            local_rank=self.run_config['local_rank']).to(self.device)

        if self.run_config['distributed']:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.run_config['local_rank']],
                                                             output_device=self.run_config['local_rank'])
        elif len(run_config['gpu_ids']) > 1:
            assert torch.cuda.is_available()
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        rles = []
        images = []
        filenames = []
        if self.run_config['distributed']:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, filename) in enumerate(self.val_loader):
            print(i)
            image = image.to(self.device)
            # target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            # self.metric.update(outputs[0], target)
            # pixAcc, mIoU = self.metric.get()
            # logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            #     i + 1, pixAcc * 100, mIoU * 100))

            if self.run_config['save_pred']:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                for predict, f_name in zip(pred, filename):
                    # predict = p.squeeze(0)
                    images.append(predict)
                    filenames.append(f_name.split('.pn')[0])

                # mask = get_color_pallete(predict, self.data_config['dataset_name'])
                # mask.save(os.path.join(run_config['path']['pred_pic'], os.path.splitext(filename[0])[0] + '.png'))
        synchronize()

        try:
            pool = Pool(8)
            for rle in tqdm(pool.map(mask2rle, images), total=len(rles)):
                rles.append(rle)
            #pool.map(process_image, mdlParams['im_paths'])  # process data_inputs iterable with pool
        finally: # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        # ids = [o.split('.pn')[0] for o in filenames]
        sub_df = pd.DataFrame({'ImageId': filenames, 'EncodedPixels': rles})
        sub_df.loc[sub_df.EncodedPixels == '', 'EncodedPixels'] = '-1'
        sub_df.to_csv('submission.csv', index=False)



def mask2rle(img, width=1024, height=1024):
    img = cv2.resize(img, (1024, 1024))
    img = (img.T * 255).astype(np.uint8)

    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    opt_p = './cfgs/fcn-resnet50_001.json'
    parser.add_argument('-opt', default=opt_p, type=str, required=False, help='Path to option JSON file.')

    config = option.parse(parser.parse_args().opt, False)
    config = option.dict_to_nonedict(config)
    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # pytorch dus not support distributed on windows but multi-gpu training
    run_config['distributed'] = num_gpus > 1 and platform.system().lower() != 'windows'
    if not run_config['gpu_ids'] == [] and torch.cuda.is_available():
        cudnn.benchmark = True
        run_config['device'] = "cuda"
        num_gpus = len(run_config['gpu_ids'])
        run_config['num_gpus'] = num_gpus
        config['optim_config']['batch_size'] = num_gpus * config['optim_config']['batch_size']
    else:
        run_config['distributed'] = False
        run_config['device'] = "cpu"
    if run_config['distributed']:
        torch.cuda.set_device(run_config['local_rank'])
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    run_config['save_pred'] = True
    run_config['eval_model'] = r'E:\repos\awesome-semantic-segmentation-pytorch\runs\experiments\fcn_resnet50_LR0.0001_001\models\fcn_resnet50_LR0.0001_001_200000.pth'
    if run_config['save_pred']:
        if not os.path.exists(run_config['path']['pred_pic']):
            os.makedirs(run_config['path']['pred_pic'])

    logger = setup_logger("semantic_segmentation", run_config['path']['log_dir'], get_rank(),
                          filename='{}_log.txt'.format(util.get_timestamp()), mode='a+')

    evaluator = Evaluator(config)
    evaluator.eval()
    torch.cuda.empty_cache()
