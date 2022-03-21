
import sys
sys.path.append("/home/pc/workspace/torch_ws/innox_ws/2D_seg_ws/yolov5-segmentation")

import os
import torch
import argparse
import cv2
import matplotlib.pyplot as plt

from data.dataloaders.utils import *
from models.seg_model import Model

from torchvision import transforms
from data.dataloaders import custom_transforms as tr

from data.dataloaders.utils import get_cityscapes_labels

def proprecess(img):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    img = np.array(img).astype(np.float32)
    img /= 255.0
    img -= mean
    img /= std
    img = np.array(img).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img[None, :] # 增加一维
    return img

def predict(opt, model, input):
    output = model(input)
    return output

def decode_segmap(mask):
    label_colours = get_cityscapes_labels()
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for ll in range(0, 20): # 选择颜色
        r[mask == ll] = label_colours[ll, 0]
        g[mask == ll] = label_colours[ll, 1]
        b[mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def post_process(img, output):
    mask =  torch.max(output[:3], 1)[1].detach().cpu().numpy() # 合并分类结果
    mask = np.squeeze(mask) # 去掉第一维
    mask = decode_segmap(mask) # 将mask合成为fgb图
    img = img/255 # 
    img = cv2.addWeighted(img, 1, mask, 0.8, 0, dtype = cv2.CV_32F) # 原图掩膜
    return img, mask 

def visualize2image(img, mask):
    plt.figure()
    plt.title('display')
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(mask)
    plt.show()

def visualize4image(img, mask, img_gt, mask_gt):
    plt.figure()
    plt.title('visualize')
    plt.subplot(2,2,1)
    plt.title('img with predict mask')
    plt.imshow(img)
    plt.subplot(2,2,2)
    plt.title('predict mask')
    plt.imshow(mask)
    plt.subplot(2,2,3)
    plt.title('img with gt mask')
    plt.imshow(img_gt)
    plt.subplot(2,2,4)
    plt.title('gt mask')
    plt.imshow(mask_gt)
    plt.show()

def run(opt):
    # 加载模型
    model = Model(opt.cfg, opt.num_class)  # create
    if opt.cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()  
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['state_dict'],False)

    # 读取图片
    if opt.dataset_dir != None:
        img_list, mask_gt_list = get_dataset_image(opt)
        for i, input_tensor in enumerate(img_list):
            # input_tensor = proprecess(img)
            print("input_tensor:", input_tensor.shape)
            # img = img_precess(input_tensor.numpy())
            img = np.squeeze(input_tensor.numpy())
            img = img_precess(img)
            print("img:", img.shape)
            output = predict(opt, model, input_tensor) 
            img, mask = post_process(img, output)
            img_gt, mask_gt = gt_process(img, mask_gt_list[i])
            visualize4image(img, mask, img_gt, mask_gt)
    else:
        img_list = os.listdir(opt.img_dir)
        for img in img_list:
            img = cv2.imread(os.path.join(opt.img_dir, img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (opt.crop_size, opt.crop_size))
            print("img:", img.shape)
            input_tensor = proprecess(img)
            print("input_tensor:", input_tensor.shape)

            output = predict(opt, model, input_tensor) 
            img, mask = post_process(img, output)
            visualize2image(img, mask)

def img_precess(img):
    img = np.transpose(img, axes=[1, 2, 0])
    img *= (0.229, 0.224, 0.225)
    img += (0.485, 0.456, 0.406)
    img *= 255.0
    img = img.astype(np.uint8)
    return img

def gt_process(img, mask_gt):
    mask_gt = np.squeeze(mask_gt) # 去掉第一维
    mask_gt = decode_segmap(mask_gt)
    img = cv2.addWeighted(img, 1, mask_gt, 0.8, 0, dtype = cv2.CV_32F) # 原图掩膜
    return img, mask_gt

def get_dataset_image(opt):
    from torch.utils.data import DataLoader
    from data.dataloaders.datasets import cityscapes, coco, pascal
    if opt.dataset == 'cityscapes':
        dataset = cityscapes.CityscapesSegmentation(opt, split='val', root=opt.dataset_dir, group=False)
    elif opt.dataset == 'coco':
        dataset = coco.COCOSegmentation(opt, split='val', root=opt.dataset_dir, group=False)
    elif opt.dataset == 'pascal':
        dataset = pascal.VOCSegmentation(opt, split='val', root=opt.dataset_dir, group=False)
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    img_list = []
    mask_gt_list = []
    # for i in range(10):
        # img_list.append(dataloader[i]['image'].numpy())
        # mask_gt_list.append(dataloader[i]['label'].numpy())
    for ii, sample in enumerate(dataloader):
        img_list.append(sample['image'])
        mask_gt_list.append(sample['label'].numpy())
        if ii >= 10:
            break

    return img_list, mask_gt_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='enable cuda')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--cfg', type=str, default='configs/model_yolo_segmentation.yaml', help='model.yaml path')
    parser.add_argument('--num_class', type=int, default=9, help='number classes of model output')
    parser.add_argument('--crop-size', type=int, default=608, help='segmentation crop image size')
    parser.add_argument('--img-dir', type=str, default='images', help='set the image dir')
    parser.add_argument('--dataset', type=str, default='cityscapes',choices=['pascal', 'coco', 'cityscapes'],help='dataset name (default: coco)')
    parser.add_argument('--dataset-dir', type=str, default=None, help='set the dataset dir')
    parser.add_argument('--save-dir', type=str, default='output', help='set the save dir')
    parser.add_argument('--weights', type=str, default="pretrained_models/model.pth.tar", help='pretrained weights path')
    
    opt = parser.parse_args()

    run(opt)