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
import torch.nn.functional as F
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
    server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])
    process_dataset(server_dataset)
    server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                           client_dataset['train'])
    data_loader = make_data_loader(server_dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    if cfg['sbn'] == 1:
        batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    elif cfg['sbn'] == 0:
        batchnorm_dataset = server_dataset['train']
    else:
        raise ValueError('Not valid sbn')
    data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            supervised_idx = result['supervised_idx']
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
    ada_cm=construct_confidence_thresholds(server,batchnorm_dataset)
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        num_classes=4
        print("epoch",epoch)
        ada_cm=update_adacm(ada_cm,num_classes,epoch)
        #ada_cm= {0: 0.10110517091807565, 1: 0.10110517091807565, 2: 0.16989710835043373, 3: 0.10110517091807565, 4: 0.10110517091807565, 5: 0.10110517091807565, 6: 0.10110517091807565, 7: 0.16610914789081413, 8: 0.10110517091807565, 9: 0.10110517091807565}
        #ada_cm = {0: 0.95, 1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.95, 7: 0.95, 8: 0.95, 9: 0.95}
        #ada_cm= {0: 0.95, 1: 0.95, 2: 0.95, 3: 0.95}
        print("adaptive threshold:",ada_cm)
        train_client(batchnorm_dataset, client_dataset['train'], server, client, optimizer, metric, logger, epoch,ada_cm)
        if 'ft' in cfg and cfg['ft'] == 0:
            train_server(server_dataset['train'],server, optimizer, metric, logger, epoch)
            logger.reset()
            server.update_parallel(client)
        else:
            logger.reset()
            server.update(client)
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test(data_loader['test'], test_model, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
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



def train_client(batchnorm_dataset, client_dataset, server, client, optimizer, metric, logger, epoch,ada_cm):
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client, batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        dataset_m = separate_dataset(client_dataset, client[m].data_split['train'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            dataset_m = client[m].make_dataset(dataset_m, metric, logger,ada_cm)
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


def train_server(dataset,server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset,lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
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


def construct_confidence_thresholds(server,labeled_dataset):
    #自适应相关参数
    confidence_thresholds = {}
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    model.load_state_dict(server.model_state_dict)
    model.eval()

    # 创建数据加载器
    data_loader = make_data_loader({'train': labeled_dataset}, 'server')['train']

    # 初始化置信度分数字典
    confidence_scores = {}

    with torch.no_grad():
        for batch in data_loader:
            # 使用 to_device 函数处理批次数据
            batch = to_device(batch, cfg['device'])

            if isinstance(batch, dict):
                data = batch['data']
                target = batch['target']
            elif isinstance(batch, list):
                data = batch[0]
                target = batch[1]
            else:
                raise ValueError("Unexpected batch format")

            # 确保数据是正确的形状
            if isinstance(data, list):
                data = torch.stack(data)

            # 确保 target 是 Tensor 类型
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, device=cfg['device'])

            # 将数据和目标都包装在字典中
            input_dict = {'data': data, 'target': target}
            output = model(input_dict)

            if isinstance(output, dict):
                output = output['target']
            pred = output.argmax(dim=1)
            confidence = F.softmax(output, dim=1).max(dim=1)[0]

            for i, t in enumerate(target):
                t_item = t.item()
                if t_item not in confidence_scores:
                    confidence_scores[t_item] = []
                if pred[i] == t:
                    confidence_scores[t_item].append(confidence[i].item())

    # 设置类别数量
    num_classes = len(confidence_scores)
    #print("confidence:",confidence_scores)

    for c in range(num_classes):
        if c in confidence_scores and confidence_scores[c]:
            mu = np.mean(confidence_scores[c])
            confidence_thresholds[c] =mu
        else:
            confidence_thresholds[c] = 0.10
    return confidence_thresholds

def update_adacm(ada_cm,num_classes,epoch):
    B = 0.01  # 控制阈值增加速率的常数
    gamma = 0.01  # 控制阈值增加速率的常数
    epoch_factor = 1 + B * np.exp(gamma)
    for c in range(num_classes):
        ada_cm[c] =  ada_cm[c]* epoch_factor
        if ada_cm[c]>0.95:
            ada_cm[c] = 0.95

    return ada_cm


if __name__ == "__main__":
    main()