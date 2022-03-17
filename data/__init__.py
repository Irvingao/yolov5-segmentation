# -*- coding: utf-8 -*-
import yaml
from addict import Dict

from data.dataloaders.datasets import cityscapes, coco, pascal
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    dataset_config = yaml.load(open(args.dataset_cfg, 'r'),Loader=yaml.FullLoader)
    dataset_config = Dict(dataset_config)['dataset']
    if args.dataset == 'pascal':
        base_dir = dataset_config['pascal']['dataset_dir']
        train_set = pascal.VOCSegmentation(args, base_dir=base_dir, split='train')
        val_set = pascal.VOCSegmentation(args, base_dir=base_dir, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        root = dataset_config['cityscapes']['dataset_dir']
        group = dataset_config['cityscapes']['group']
        with_background = dataset_config['cityscapes']['with_background']
        train_set = cityscapes.CityscapesSegmentation(args, root=root, split='train', group=group, with_background=with_background)
        val_set = cityscapes.CityscapesSegmentation(args, root=root, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, root=root, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        base_dir = dataset_config['coco']['dataset_dir']
        train_set = coco.COCOSegmentation(args, base_dir, split='train')
        val_set = coco.COCOSegmentation(args, base_dir, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

