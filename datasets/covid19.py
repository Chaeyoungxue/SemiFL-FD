import anytree
import numpy as np
import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
from sklearn.model_selection import train_test_split

class COVID19(Dataset):
    data_name = 'COVID19'
    file = [('https://wjdcloud.blob.core.windows.net/dataset/cycfed/covid19.tar.gz', 'fcebcd5633975dd4a3950f8264413088')]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        #print(self.data[index])
        #print(self.target[index])
        #print(self.data[index].shape)
        data, target = Image.fromarray(self.data[index]), torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, 'data': data, 'target': target}
        #print("self.transfrom:",self.transform)
        if self.transform is not None:
            #print("input_data_origin:", input['data'].size)
            input = self.transform(input)
            #print("input_data:",input['data'].shape)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        #data = np.load(os.path.join(self.raw_folder, 'covid19/xdata.npy'))
        #计算数据均值和标准差
        #data = data.astype(np.float32) / 255.0
        #mean = np.mean(data, axis=(0, 2, 3), keepdims=True)
        #mean=np.mean(mean)
        #std = np.std(data, axis=(0, 2, 3), keepdims=True)
        #std=np.mean(std)
        #print('mean:', mean)
        #print('std:', std)
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            #download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        data = np.load(os.path.join(self.raw_folder, 'covid19/xdata.npy'))

        #data = np.transpose(data, (0, 3, 1, 2))
        #print("data:", data.shape)
        targets = np.load(os.path.join(self.raw_folder, 'covid19/ydata.npy'))

        train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=42)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        #print("train_id:", train_id)
        #print("test_id:", test_id)
        #print("train_data:", train_data.shape)
        #print("train_target:", train_target.shape)
        #print("test_data:", test_data.shape)
        #print("test_target:", test_target.shape)
        train_data = (train_data * 255).astype(np.uint8)
        test_data = (test_data * 255).astype(np.uint8)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(4))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (
            classes_to_labels, target_size)

