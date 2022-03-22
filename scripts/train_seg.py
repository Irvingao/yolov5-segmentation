import os
import sys
sys.path.append(os.path.abspath('.')) # add the workspace path to system
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from tqdm import tqdm


def train(opt):
    torch.cuda.empty_cache()
    
    save_dir, epochs, batch_size, weights = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights

    # Define Saver
    from utils.saver import Saver
    saver = Saver(opt)
    saver.save_experiment_config()
    
    # Tensorboard
    # summary
    from utils.summaries import TensorboardSummary
    summary = TensorboardSummary(opt.save_dir)
    writer = summary.create_summary()

    # cuda mode
    cuda = torch.cuda.is_available()
    print("use cuda:", cuda)

    # Trainloader
    from data import make_data_loader
    kwargs = {'num_workers': opt.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(opt, **kwargs)

    print("number classes:", nclass)

    # Model
    from models.seg_model import Model
    model = Model(opt.cfg, nclass)  # create

    # Define Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9,
                                weight_decay=0.0005, nesterov=False)

    if cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()    

    # Pretrained
    if weights != None:
        pretrained = weights if os.path.exists(weights) else None

        if pretrained != None:
            checkpoint = torch.load(pretrained)
            model.load_state_dict(checkpoint['state_dict'],False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_pred = checkpoint['best_pred']
            print('-------------------------------------------------------------------')  # report
            print('load pretrained model sucessfully')  # report
            print('-------------------------------------------------------------------')  # report

   
    from models.loss import SegmentationLosses
    weight = None
    criterion = SegmentationLosses(weight=weight, cuda=cuda).build_loss()

    # Define Evaluator
    from utils.metrics import Evaluator
    evaluator = Evaluator(nclass)
    
    # Define lr scheduler
    from utils.lr_scheduler import LR_Scheduler
    scheduler = LR_Scheduler("poly", 0.1, opt.epochs, len(train_loader))

    best_pred = 0.0
    if opt.resume:
        if not os.path.isfile(opt.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(opt.resume))
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch']
        if cuda:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        if not opt.ft:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(opt.resume, checkpoint['epoch']))
        
    print('Starting Epoch:', 0)
    print('Total Epoches:', opt.epochs)
    
    for epoch in range(0, opt.epochs):
        train_loss = 0.0
        model.train()

        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if cuda:
                image, target = image.cuda(), target.cuda()
            # 学习率更新策略
            scheduler(optimizer, i, epoch, best_pred) 
            optimizer.zero_grad()
            # forward
            output = model(image)

            # loss
            loss = criterion(output, target)
            # backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # tensorboard
            writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                # print("[visualize image]")
                global_step = i + num_img_tr * epoch
                summary.visualize_image(writer, opt.dataset, image, target, output, global_step)

        writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * opt.batch_size + image.data.shape[0]))
        
        train_loss = train_loss / len(tbar)

        # save checkpoint every epoch
        if opt.no_val:
            is_best = False
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
            }, is_best)
            
        # eval
        if not opt.no_val:
            tbar = tqdm(val_loader, desc='\r')
            test_loss = 0.0
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                if cuda:
                    image, target = image.cuda(), target.cuda()
                
                # 不更新梯度
                with torch.no_grad():
                    output = model(image)
                # print("test input:", image.shape)
                # print("test output:", output.shape)

                loss = criterion(output, target)

                test_loss += loss.item()

                tbar.set_description('Test loss: %.5f' % (test_loss / (i + 1)))
                
                target = target.cpu().numpy()
                pred = output.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(target, pred)

            # Fast test during the training
            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
            writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            writer.add_scalar('val/mIoU', mIoU, epoch)
            writer.add_scalar('val/Acc', Acc, epoch)
            writer.add_scalar('val/Acc_class', Acc_class, epoch)
            writer.add_scalar('val/fwIoU', FWIoU, epoch)
            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * opt.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            
            test_loss = test_loss / len(tbar)

            # only save the best model
            new_pred = mIoU
            if new_pred >= best_pred:
                is_best = True
                best_pred = new_pred
                saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_pred,
                }, is_best)

    # end training
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg', action='store_true', default=True, help='enable segmentation')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--cfg', type=str, default='configs/model_yolo_segmentation.yaml', help='model.yaml path')
    parser.add_argument('--dataset-cfg', type=str, default='configs/data.yaml', help='data.yaml path')
    parser.add_argument('--dataset', type=str, default='coco',choices=['pascal', 'coco', 'cityscapes'],help='dataset name (default: coco)')
    parser.add_argument('--base-size', type=int, default=608, help='segmentation base image size')
    parser.add_argument('--crop-size', type=int, default=608, help='segmentation crop image size')
    parser.add_argument('--checkname', type=str, default='checkpoint', help='set the checkpoint name')
    parser.add_argument('--save-dir', type=str, default='log_dir', help='set the save dir')
    # finetuning pre-trained models
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    parser.add_argument('--resume', type=str, default='', help='resume most recent training')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()


    # train
    print('Start Tensorboard with "tensorboard --logdir {}", view at http://localhost:6006/'.format(opt.save_dir))
    
    train(opt)
