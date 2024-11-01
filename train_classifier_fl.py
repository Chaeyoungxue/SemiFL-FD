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
import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    batchnorm_dataset = dataset['train']
    data_split = split_dataset(dataset, cfg['num_clients'], cfg['data_split_mode'])
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            server = result['server']
            client = result['client']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train_client(dataset['train'], server, client, optimizer, metric, logger, epoch)
        server.update(client)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(), 'data_split': data_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def make_server(model):
    server = Server(model)
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    return client


def train_client(dataset, server, client, optimizer, metric, logger, epoch):
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        dataset_m = separate_dataset(dataset, client[m].data_split['train'])
        if dataset_m is not None:
            client[m].active = True
            client[m].train(dataset_m, lr, metric, logger)
        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
