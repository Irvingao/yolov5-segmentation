# -*- coding: utf-8 -*-
from .yolov5_seg import YOLOv5_seg

__all__ = ['build_backbone']

support_backbone = ['YOLOv5_seg']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
