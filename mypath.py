class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/media/raid5-array/datasets/VOC2012/VOCdevkit/VOC2012'
        elif dataset == 'sbd':
            return ''
        elif dataset == 'cityscapes':
            return ''
        elif dataset == 'coco':
            return ''
        elif dataset == 'ade':
            return './Dataset'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
