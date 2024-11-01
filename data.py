import copy
import torch
import numpy as np
import models
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687)),
              'COVID19': ((0.4906, 0.4906, 0.4906), (0.2322, 0.2322, 0.2322))}


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['STL10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['COVID19']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = iid(dataset['train'], num_users)
        data_split['test'] = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'] = non_iid(dataset['train'], num_users)
        data_split['test'] = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    return data_split


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset, num_users):
    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
        for target_i in range(cfg['target_size']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['target_size']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split


def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    return separated_dataset


def separate_dataset_su(server_dataset, client_dataset=None, supervised_idx=None):
    # 如果supervised_idx为空，则根据数据集类型和配置参数生成supervised_idx
    if supervised_idx is None:
        # 如果数据集类型为STL10
        if cfg['data_name'] in ['STL10']:
            # 如果num_supervised为-1，则将前5000个样本作为supervised_idx
            if cfg['num_supervised'] == -1:
                supervised_idx = torch.arange(5000).tolist()
            else:
                # 获取前5000个样本的标签
                target = torch.tensor(server_dataset.target)[:5000]
                # 计算每个类别的样本数量
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                # 遍历每个类别
                for i in range(cfg['target_size']):
                    # 获取当前类别的样本索引
                    idx = torch.where(target == i)[0]
                    # 随机打乱索引
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    # 将索引添加到supervised_idx中
                    supervised_idx.extend(idx)
        else:
            # 如果num_supervised为-1，则将所有样本作为supervised_idx
            if cfg['num_supervised'] == -1:
                supervised_idx = list(range(len(server_dataset)))
            else:
                # 获取所有样本的标签
                target = torch.tensor(server_dataset.target)
                # 计算每个类别的样本数量
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                # 遍历每个类别
                for i in range(cfg['target_size']):
                    # 获取当前类别的样本索引
                    idx = torch.where(target == i)[0]
                    # 随机打乱索引
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    # 将索引添加到supervised_idx中
                    supervised_idx.extend(idx)
    # 获取所有样本的索引
    idx = list(range(len(server_dataset)))
    # 获取unsupervised_idx，即不在supervised_idx中的样本索引
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    # 根据supervised_idx将server_dataset划分为_server_dataset
    _server_dataset = separate_dataset(server_dataset, supervised_idx)
    # 如果client_dataset为空，则将server_dataset划分为_client_dataset
    if client_dataset is None:
        _client_dataset = separate_dataset(server_dataset, unsupervised_idx)
    else:
        # 否则，将client_dataset划分为_client_dataset
        _client_dataset = separate_dataset(client_dataset, unsupervised_idx)
        # 设置_client_dataset的transform为FixTransform
        transform = FixTransform(cfg['data_name'])
        _client_dataset.transform = transform
    # 返回_server_dataset，_client_dataset和supervised_idx
    return _server_dataset, _client_dataset, supervised_idx



def make_batchnorm_dataset_su(server_dataset, client_dataset):
    """
    将服务器数据集和客户端数据集合并为一个符合批归一化要求的数据集。

    这个函数的目的是为了处理在分布式学习中，需要在服务器端进行批归一化的情况。
    它通过深复制服务器数据集，并将客户端数据集的数据、目标和其他信息（如id）合并到复制的数据集中，
    从而创建一个新的数据集，这个数据集包含了服务器和客户端的数据，用于批归一化计算。

    参数:
    - server_dataset: 服务器端的数据集，包含数据、目标和其他信息。
    - client_dataset: 客户端的数据集，包含数据、目标和其他信息。

    返回:
    - batchnorm_dataset: 一个新的数据集，包含了服务器和客户端的数据，用于批归一化计算。
    """
    # 深复制服务器数据集以创建批归一化数据集的基础
    batchnorm_dataset = copy.deepcopy(server_dataset)

    # 将服务器和客户端的数据合并到批归一化数据集中
    batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data

    # 将服务器和客户端的目标合并到批归一化数据集中
    batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target

    # 将服务器和客户端的id合并到批归一化数据集中
    batchnorm_dataset.other['id'] = batchnorm_dataset.other['id'] + client_dataset.other['id']

    # 返回新的批归一化数据集
    return batchnorm_dataset



def make_dataset_normal(dataset):
    """
    将数据集标准化为符合正常分布的形式。

    这个函数的主要作用是将传入的数据集进行标准化处理，使其均值和方差符合标准正态分布。
    这对于后续的深度学习模型训练非常有帮助，因为标准化后的数据可以加速训练过程并提高模型性能。
    """
    import datasets
    _transform = dataset.transform
    transform = datasets.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg['data_name']])])
    dataset.transform = transform
    return dataset, _transform


def make_batchnorm_stats(dataset, model, tag):
    """
    计算并更新模型的批归一化统计量。

    该函数通过遍历数据集来计算批归一化层的运行均值和运行方差，
    然后将这些统计量更新到模型中。这个过程是为了确保模型在
    推理时能够使用准确的批归一化统计量，从而提高模型的泛化性能。

    参数:
    - dataset: 数据集，用于计算批归一化统计量。
    - model: 模型，需要更新批归一化统计量的模型。
    - tag: 标签，用于标识数据加载器的配置。

    返回:
    - 更新了批归一化统计量的模型。
    """
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        dataset, _transform = make_dataset_normal(dataset)
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
        dataset.transform = _transform
    return test_model


class FixTransform(object):
    def __init__(self, data_name):
        import datasets
        if data_name in ['CIFAR10', 'CIFAR100']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['SVHN']:
            self.weak = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['STL10']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        # 如果data_name为COVID19
        elif data_name in ['COVID19']:
            # 定义弱增强变换
            self.weak = transforms.Compose([
                # 随机水平翻转
                transforms.RandomHorizontalFlip(),
                # 随机裁剪，padding为4，填充模式为reflect
                transforms.RandomCrop(128, padding=12, padding_mode='reflect'),
                # 转换为张量
                transforms.ToTensor(),
                # 标准化
                transforms.Normalize(*data_stats[data_name])
            ])
            # 定义强增强变换
            self.strong = transforms.Compose([
                # 随机水平翻转
                transforms.RandomHorizontalFlip(),
                # 随机裁剪，padding为4，填充模式为reflect
                transforms.RandomCrop(128, padding=12, padding_mode='reflect'),
                # RandAugment增强，n为2，m为10
                datasets.RandAugment(n=2, m=10),
                # 转换为张量
                transforms.ToTensor(),
                # 标准化
                transforms.Normalize(*data_stats[data_name])
            ])
        else:
            raise ValueError('Not valid dataset')

    def __call__(self, input):
        data = self.weak(input['data'])
        aug = self.strong(input['data'])
        input = {**input, 'data': data, 'aug': aug}
        return input


class MixDataset(Dataset):
    """
    一个混合数据集类，用于从给定的数据集中随机抽取数据。

    这个类继承自PyTorch的Dataset类，主要用于在训练模型时，以一种随机的方式
    从已有的数据集中获取样本数据。其主要用途是在数据增强或数据重采样的场景中，
    提供对数据集的灵活访问。
    """
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        input = {'data': input['data'], 'target': input['target']}
        return input

    def __len__(self):
        return self.size
