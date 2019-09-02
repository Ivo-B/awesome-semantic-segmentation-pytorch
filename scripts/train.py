import argparse
import time
import datetime
import os
import shutil
import sys
import platform
import random

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
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric
import core.utils.options as option

from tensorboardX import SummaryWriter
from torch.autograd import Variable

import util


def parse_args():

    # model and dataset
    # parser.add_argument('--root-dir', type=str, default='../runs'),

    # parser.add_argument('--model', type=str, default='fcn',
    #                     choices=['fcn32s', 'fcn16s', 'fcn8s',
    #                              'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
    #                              'danet', 'denseaspp', 'bisenet',
    #                              'encnet', 'dunet', 'icnet',
    #                              'enet', 'ocnet', 'ccnet', 'psanet',
    #                              'cgnet', 'espnet', 'lednet', 'dfanet'],
    #                     help='model name (default: fcn32s)')
    # parser.add_argument('--backbone', type=str, default='resnet50',
    #                     choices=['vgg16', 'resnet18', 'resnet50',
    #                              'resnet101', 'resnet152', 'densenet121',
    #                              'densenet161', 'densenet169', 'densenet201'],
    #                     help='backbone name (default: vgg16)')
    # parser.add_argument('--dataset', type=str, default='ptx',
    #                     choices=['pascal_voc', 'pascal_aug', 'ade20k',
    #                              'citys', 'sbu'],
    #                     help='dataset name (default: pascal_voc)')
    # parser.add_argument('--base-size', type=int, default=1024,
    #                     help='base image size')
    # parser.add_argument('--crop-size', type=int, default=480,
    #                     help='crop image size')
    # parser.add_argument('--workers', '-j', type=int, default=4,
    #                     metavar='N', help='dataloader threads')
    # training hyper params
    # parser.add_argument('--jpu', action='store_true', default=False,
    #                     help='JPU')
    # parser.add_argument('--use-ohem', type=bool, default=False,
    #                     help='OHEM Loss for cityscapes dataset')
    # parser.add_argument('--aux', action='store_true', default=False,
    #                     help='Auxiliary loss')
    # parser.add_argument('--aux-weight', type=float, default=0.4,
    #                     help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    # parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
    #                     help='learning rate (default: 1e-4)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
    #                     help='w-decay (default: 5e-4)')

    # parser.add_argument('--warmup-iters', type=int, default=0,
    #                     help='warmup iters')
    # parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
    #                     help='lr = warmup_factor * lr')
    # parser.add_argument('--warmup-method', type=str, default='linear',
    #                     help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=0)
    # checkpoint and log
    # parser.add_argument('--resume', type=str, default=None,
    #                     help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='models',
                        help='Directory for saving checkpoint models')
    # parser.add_argument('--save-epoch', type=int, default=10,
    #                     help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory for saving checkpoint models')
    # parser.add_argument('--log-iter', type=int, default=10,
    #                     help='print log every log-iter')
    # evaluation only
    # parser.add_argument('--val-epoch', type=int, default=1,
    #                     help='run validation every val-epoch')
    # parser.add_argument('--skip-val', action='store_true', default=False,
    #                     help='skip validation during training')
    args = parser.parse_args()

    #dir structure for experiment
    experiment_id = f'{args.model}-{args.backbone}-Epochs{args.epochs}-LR{args.lr}'
    args.root_dir = os.path.join(args.root_dir, experiment_id)
    util.mkdir_and_rename(args.root_dir)  # rename old folder if exists
    args.save_dir = os.path.join(args.root_dir, args.save_dir)
    args.log_dir = os.path.join(args.root_dir, args.log_dir)
    util.mkdirs([args.save_dir, args.log_dir])

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.run_config = config['run_config']
        self.optim_config = config['optim_config']
        self.data_config = config['data_config']
        self.model_config = config['model_config']

        self.device = torch.device(self.run_config["device"])
        if 'debug' not in self.run_config['id']:
            util.mkdir_and_rename(os.path.join(run_config['path']['root_dir'], 'tb_logs', self.run_config['id']))
        self.writer = SummaryWriter(os.path.join(run_config['path']['root_dir'], 'tb_logs', self.run_config['id']))

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            transforms.Normalize([.485], [1.]),
        ])
        # dataset and dataloader
        logger.info("===> Loading datasets")
        data_kwargs = {'transform': input_transform, 'base_size': self.data_config['base_size'], 'crop_size': self.data_config['crop_size']}
        train_dataset = get_segmentation_dataset(self.data_config['dataset_name'], root=self.data_config['dataset_root'], split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(self.data_config['dataset_name'], root=self.data_config['dataset_root'], split='val', mode='val', **data_kwargs)

        if self.run_config['distributed']:
            self.optim_config['iters_per_epoch'] = len(train_dataset) // (len(self.run_config['num_gpus']) * self.optim_config['batch_size'])
        else:
            self.optim_config['iters_per_epoch'] = len(train_dataset) // self.optim_config['batch_size']
        self.optim_config['max_expochs'] = self.optim_config['max_iters'] // len(train_dataset)

        train_sampler = make_data_sampler(train_dataset, shuffle=self.data_config['train']['use_shuffle'], distributed=self.run_config['distributed'])
        train_batch_sampler = make_batch_data_sampler(train_sampler, self.optim_config['batch_size'], self.optim_config['max_iters'])
        val_sampler = make_data_sampler(val_dataset, self.data_config['val']['use_shuffle'], self.run_config['distributed'])
        val_batch_sampler = make_batch_data_sampler(val_sampler, self.optim_config['batch_size'])

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=self.data_config['train']['n_workers'],
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=self.data_config['val']['n_workers'],
                                          pin_memory=True)

        # create network
        logger.info("===> Building model")
        BatchNorm2d = nn.SyncBatchNorm if self.run_config['distributed'] else nn.BatchNorm2d

        self.model = get_segmentation_model(model=self.model_config['model'],
                                            dataset=self.data_config['dataset_name'],
                                            backbone=self.model_config['backbone'],
                                            aux=self.optim_config['aux'],
                                            jpu=self.model_config['jpu'],
                                            norm_layer=BatchNorm2d,
                                            root=run_config['path']['root_dir']).to(self.device)
        self.model = self.model.to(self.device)
        # dummy_input = Variable(torch.rand(4, 3, args.crop_size, args.crop_size)).to(self.device)
        # self.writer.add_graph(self.model, dummy_input, True)

        # resume checkpoint if needed
        if run_config['path']['resume']:
            if os.path.isfile(run_config['path']['resume']):
                name, ext = os.path.splitext(run_config['path']['resume'])
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(run_config['path']['resume']))
                self.model.load_state_dict(torch.load(run_config['path']['resume'], map_location=lambda storage, loc: storage))

        # create criterion
        logger.info("===> Setting up loss")
        self.criterion = get_segmentation_loss(self.model_config['model'],
                                               use_ohem=self.optim_config['use_ohem'],
                                               aux=self.optim_config['aux'],
                                               aux_weight=self.optim_config['aux_weight'],
                                               ignore_index=-1).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': self.optim_config['lr']})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': self.optim_config['lr'] * 10})
        # TODO: implement other optimizer methods!
        if optim_config['optim'].lower() == "sgd":
            self.optimizer = torch.optim.SGD(params_list,
                                             lr=self.optim_config['lr'],
                                             momentum=self.optim_config['momentum'],
                                             weight_decay=self.optim_config['weight_decay'])
        elif optim_config['optim'].lower() == "adam":
            self.optimizer = torch.optim.Adam(params_list,
                                             lr=self.optim_config['lr'],
                                             # momentum=self.optim_config['momentum'],
                                             weight_decay=self.optim_config['weight_decay'])
        elif optim_config['optim'].lower() == "radam":
            from scripts.radam import RAdam
            self.optimizer = RAdam(params_list, lr=self.optim_config['lr'])
        else:
            raise NotImplementedError('Optimizer [{:s}] is not implemented.'.format(optim_config['optim']))

        # lr scheduling
        # TODO: implement other scheduling methods!
        if optim_config['lr_scheduler'].lower() == "warmuppolylr":
            self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                             max_iters=optim_config['max_iters'],
                                             power=optim_config['power'],
                                             warmup_factor=optim_config['warmup_factor'],
                                             warmup_iters=optim_config['warmup_iters'],
                                             warmup_method=optim_config['warmup_method'])
        else:
            raise NotImplementedError('LR scheduler [{:s}] is not implemented.'.format(optim_config['lr_scheduler']))

        if self.run_config['distributed']:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.run_config['local_rank']],
                                                             output_device=self.run_config['local_rank'])
        elif len(run_config['gpu_ids']) > 1:
            assert torch.cuda.is_available()
            self.model = nn.DataParallel(self.model)


        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = int(self.optim_config['max_expochs']), int(self.optim_config['max_iters'])
        log_per_iters, val_per_iters = self.run_config['logger']['print_freq'], self.optim_config['val_freq']
        save_per_iters = self.run_config['logger']['save_checkpoint_freq']
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            self.lr_scheduler.step()

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))
                self.writer.add_scalar('train/loss', losses_reduced.item(), iteration)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.config, iteration, is_best=False)

            if not self.optim_config['skip_val'] and iteration % val_per_iters == 0:
                self.validation(iteration)
                self.model.train()

        save_checkpoint(self.model, self.config, iteration, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, iteration):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.run_config['distributed']:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        self.writer.add_scalar(f'val/pixAcc', pixAcc, iteration)
        self.writer.add_scalar(f'val/mIoU', mIoU, iteration)
        self.writer.add_scalar(f'val/pred', (pixAcc + mIoU) / 2, iteration)
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.config, iteration, is_best)
        synchronize()


def save_checkpoint(model, config, iteration, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(config['run_config']['path']['models'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(config['run_config']['id'], iteration)
    filename = os.path.join(directory, filename)

    if config['run_config']['distributed']:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_best_model.pth'.format(config['run_config']['id'])
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    opt_p = './cfgs/fcn-resnet50_001.json'
    parser.add_argument('-opt', default=opt_p, type=str, required=False, help='Path to option JSON file.')

    config = option.parse(parser.parse_args().opt, True)
    config = option.dict_to_nonedict(config)
    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    #pytorch dus not support distributed on windows but multi-gpu training
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
    config['optim_config']['lr'] = config['optim_config']['lr'] * num_gpus

    # config loggers. Before it, the log will not work

    logger = setup_logger("semantic_segmentation", run_config['path']['log_dir'], get_rank(), filename='{}_log.txt'.format(util.get_timestamp()))
    logger.info(option.dict2str(config))
    if run_config['device'] == "cuda":
        logger.info("Using [{}] GPUs with id [{}]".format(num_gpus, run_config['gpu_ids']))
        if len(run_config['gpu_ids']) > 1 and torch.cuda.is_available():
            logger.info("Batch size is scaled by num_gps=[{}] to [{}]".format(run_config['num_gpus'], config['optim_config']['batch_size']))
    else:
        logger.info("Using CPU")

    # set random seed
    logger.info("===> Set seed")
    seed = run_config['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        logger.info("=> Random seed: {}".format(seed))
    else:
        seed = int(seed, 16)
        logger.info("=> Manual seed: {}".format(seed))
    seed = int(run_config['manual_seed'], 16)
    util.set_random_seed(seed)

    trainer = Trainer(config)
    trainer.train()
    torch.cuda.empty_cache()
