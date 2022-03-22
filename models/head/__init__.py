# -*- coding: utf-8 -*-
from .segmentation import SegmentationHead

__all__ = ['build_head', 'SegmentationHead']
support_head = ['SegmentationHead']


def build_head(head_name, **kwargs):
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head