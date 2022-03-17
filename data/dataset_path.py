class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            # return '/root/ghz_ws/cityscapes'     # foler that contains leftImg8bit/
            return '/home/pc/workspace/dataset/cityscapes'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/root/ghz_ws/coco'
            # return '/home/pc/workspace/dataset/coco2017'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
