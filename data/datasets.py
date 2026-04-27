import torch.utils.data as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'gopro':
            if is_for_train:
                from data.gopro import TrainDataset
                dataset = TrainDataset(opt, is_for_train=is_for_train)  # 传递 is_for_train 参数
            else:
                from data.gopro import TestDataset
                dataset = TestDataset(opt, is_for_train=is_for_train)  # 传递 is_for_train 参数
        elif dataset_name == 'revd':
            if is_for_train:
                from data.revd import TrainDataset
                dataset = TrainDataset(opt, is_for_train=is_for_train)  # 传递 is_for_train 参数
            else:
                from data.revd import TestDataset
                dataset = TestDataset(opt, is_for_train=is_for_train)  # 传递 is_for_train 参数
        elif dataset_name == 'realdata':
            from data.realdata import Dataset
            dataset = Dataset(opt, is_for_train=is_for_train)  # 传递 is_for_train 参数
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)
        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._opt = opt
        self._is_for_train = is_for_train

    @property
    def name(self):
        return self._name