# -*- coding: utf-8 -*-
from addict import Dict
from torch import nn
import math
import yaml
import torch
from models.modules.common import Conv
from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head


class Model(nn.Module):
    def __init__(self, model_config, num_class):
        """
        :param model_config:
        """

        super(Model, self).__init__()
        if type(model_config) is str:
            model_config = yaml.load(open(model_config, 'r'),Loader=yaml.FullLoader)
        model_config = Dict(model_config)
        # backbone type
        backbone_type = model_config.backbone.pop('type')
        print("model_config:", model_config)
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        backbone_out = self.backbone.out_shape
        # backbone version and neck 
        backbone_out['version'] = model_config.backbone.version
        self.fpn = build_neck('FPN', **backbone_out)
        fpn_out = self.fpn.out_shape

        fpn_out['version'] = model_config.backbone.version
        self.pan = build_neck('PAN', **fpn_out)
        pan_out = self.pan.out_shape

        seg = model_config.head.pop('seg')
        model_config.head["num_class"] = num_class

        self.segmentation = build_head("SegmentationHead", **model_config.head)
        


    def forward(self, x):
        out = self.backbone(x)
        # print("len(out)", len(out))
        head_ = out[3:]
        out = out[:3]
        # print("len(head_)", len(head_))
        out = self.fpn(out)
        out = self.pan(out)
        # y = self.detection(list(out))
        # print(type(out))
        # print(len(out))
        # print(out.shape)
        # head_input = list(out) +  head_
        # print(type(head_[0]))
        # print(type(head_[1]))
        # print("head_input: ", len(head_input))
        y = self.segmentation(out[0], head_[0], head_[1])
        return y


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 608, 608).to(device)

    model = Model(model_config='configs/model_yolo_segmentation.yaml').to(device)
    # model.fuse()
    print("foward")
    y = model.forward(x)
    for item in y:
        print("item:", item.shape)
    print("done")