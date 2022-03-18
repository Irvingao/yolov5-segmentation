
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
    n_classes = 20 # 加一个背景类
    label_colours = get_cityscapes_labels()
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for ll in range(0, n_classes):
        r[mask == ll] = label_colours[ll, 0]
        g[mask == ll] = label_colours[ll, 1]
        b[mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def visualize_image(img, output):

    mask =  torch.max(output[:3], 1)[1].detach().cpu().numpy() # 合并分类结果
    mask = np.squeeze(mask) # 去掉第一维
    mask = decode_segmap(mask) # 将mask合成为fgb图
 
    img = img/255 # 
    img = cv2.addWeighted(img, 1, mask, 0.7, 0, dtype = cv2.CV_32F)
    plt.figure()
    plt.title('display')
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(mask)
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
    img_list = os.listdir(opt.img_dir)
    for img in img_list:
        img = cv2.imread(os.path.join(opt.img_dir, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (opt.img_size, opt.img_size))
        input_tensor = proprecess(img)

        output = predict(opt, model, input_tensor) 

        visualize_image(img, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='enable cuda')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--cfg', type=str, default='configs/model_yolo_segmentation.yaml', help='model.yaml path')
    parser.add_argument('--dataset', type=str, default='cityscapes',choices=['pascal', 'coco', 'cityscapes'],help='dataset name (default: coco)')
    parser.add_argument('--num_class', type=int, default=9, help='number classes of model output')
    parser.add_argument('--img-size', type=int, default=608, help='segmentation base image size')
    parser.add_argument('--img-dir', type=str, default='images', help='set the image dir')
    parser.add_argument('--save-dir', type=str, default='output', help='set the save dir')
    parser.add_argument('--weights', type=str, default="pretrained_models/model.pth.tar", help='pretrained weights path')
    
    opt = parser.parse_args()

    run(opt)