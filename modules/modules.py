#modules.py

import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device
from metrics import Accuracy
import metrics
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
'''
    待解决问题：
    1、提取性能敏感特征
    2、性能敏感特征结合
'''
'''
# 定义 EfficientModel 类
class EfficientModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EfficientModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 特征提取器输出特征的维度
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 假设输入图像大小为32x32
        self.fc2 = nn.Linear(128, output_dim)  # 输出维度为分类任务的类别数

    def extract_feature(self, x):
        x = self.feature_extractor(x)  # 提取高维特征
        x = x.view(x.size(0), -1)  # 展平成全连接层输入
        return self.fc1(x)  # 返回提取到的特征

    def classifier(self, x):
        return self.fc2(x)  # 返回分类结果
'''

'''
特征蒸馏：

Client类：
    train函数
        添加feature_distillation函数：
            实现功能：客户端的fix_dataset输入后,经过生成器生成性能鲁棒特征r，相减得到性能敏感特征s。s经过分类器输出预测标签，
                    与原标签y比较，构建相应损失函数，训练n次或到达损失设定线，输出提取的加密性能敏感特征。            
            输入：client类中make_datasets函数生成的fix_dataset
            输出：加密性能敏感特征

Server类：
    update函数:
        实现功能：除去原有的上传各client模型参数，加上上传client的加密性能敏感特征。

    train函数：
        实现功能：修改dataset，使每一次的dataset都加上一次的各客户端上传的加密性能敏感特征。
'''


class Autoencoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(Autoencoder, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 32, 32]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 16, 16, 16]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 32, 8, 8]
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # [B, 16, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, output_dim, kernel_size=2, stride=2),  # [B, output_dim, 32, 32]
            nn.Sigmoid()  # 假设输入图像已经归一化到 [0, 1]
        )



    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        #print("decoded shape:", decoded.shape)
        return decoded


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)

class Server:
    def __init__(self, model):
        self.model = model
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())

        # 初始化自适应置信度阈值相关参数
        self.confidence_thresholds = {}
        self.beta = 1.0  # 控制置信区间宽度的超参数
        self.B = 0.97  # 控制阈值增加速率的常数
        self.gamma = np.e  # 控制阈值增加速率的常数
        self.t = 0  # 时间步
        self.num_classes = None# 我们将在 update_confidence_thresholds 中设置这个值

        #性能敏感特征
        '''
        性能敏感特征集的参数
        '''
        self.sensitive_features = None
        self.sensitive_features_target=None

    def distribute(self, client, batchnorm_dataset=None):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        if batchnorm_dataset is not None:
            model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return



    def update(self, client):
        if 'fmatch' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())

                    # 整合性能敏感特征
                    self.sensitive_features = [client[i].z for i in range(len(client)) if client[i].active]
                    self.sensitive_features= torch.cat(self.sensitive_features, dim=0)
                    self.sensitive_features_target = torch.cat([client[i].z_target for i in range(len(client)) if client[i].active], dim=0)
                    #print("sensitive_features:",self.sensitive_features.shape)  # 输出合并后的张量形状
                    #print("sensitive_features_target:",self.sensitive_features_target.shape)  # 输出合并后的张量形状
                    #清空已激活的客户端的性能敏感特征
                    for i in range(len(client)):
                        if client[i].active:
                            client[i].z = []
                            client[i].z_target = []


        elif 'fmatch' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())

                    # 整合性能敏感特征
                    self.sensitive_features = [client[i].z for i in range(len(client)) if client[i].active]
                    self.sensitive_features= torch.cat(self.sensitive_features, dim=0)
                    self.sensitive_features_target = torch.cat([client[i].z_target for i in range(len(client)) if client[i].active], dim=0)
                    #print("sensitive_features:",self.sensitive_features.shape)  # 输出合并后的张量形状
                    #print("sensitive_features_target:",self.sensitive_features_target.shape)  # 输出合并后的张量形状
                    #清空已激活的客户端的性能敏感特征
                    for i in range(len(client)):
                        if client[i].active:
                            client[i].z = []
                            client[i].z_target = []

        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return



    def update_parallel(self, client):
        if 'frgd' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server))
                    weight = weight / (2 * (weight.sum() - 1))
                    weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client_server)):
                                tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'frgd' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v_1 = v.data.new_zeros(v.size())
                            tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(1, num_valid_client // 2 + 1):
                                tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v_2 = v.data.new_zeros(v.size())
                            tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v = (tmp_v_1 + tmp_v_2) / 2
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            data_sen_loader=self.genger_sensitive_dataset()
            self.sensitive_features = []
            self.sensitive_features_target = []
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)

            model_s=eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None

            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    #print("output['loss']:", output['loss'])
                    #print("output['loss'].shape:", output['loss'].shape)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
            dic_initial=save_model_state_dict(model.state_dict())
            model_s.load_state_dict(dic_initial)
            for i, inputs in enumerate(data_sen_loader):
                converted_input = {
                    'data': inputs[0],
                    'target': inputs[1],
                }
                #print("input:", converted_input)
                converted_input = to_device(converted_input, cfg['device'])
                optimizer.zero_grad()
                outputs =model_s(converted_input)
                outputs['loss']= outputs['loss']*0.5
                outputs['loss'].backward()
                optimizer.step()
            self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
            self.model_state_dict = save_model_state_dict(model_s.state_dict())

        else:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            v.grad[(v.grad.size(0) // 2):] = 0
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
            self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
            self.model_state_dict = save_model_state_dict(model.state_dict())
        return

    def genger_sensitive_dataset(self):
        sen_dataset = TensorDataset(self.sensitive_features, self.sensitive_features_target)
        batch_size = 4
        dataloader = DataLoader(sen_dataset, batch_size=batch_size, shuffle=True)
        return dataloader

class Client:

    def __init__(self, client_id, model, data_split):
        self.model = model  # 确保模型被存储为 self.model
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_phi_parameters(), 'local')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']

        self.z = []
        self.z_target=[]

    def make_hard_pseudo_label(self, soft_pseudo_label, adacm):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = torch.zeros_like(max_p, dtype=torch.bool)
        # print("max_p",max_p)
        # print("hard_pseudo_label",hard_pseudo_label)
        for i in range(len(max_p)):
            label = hard_pseudo_label[i].item()
            conf_th = adacm[label]
            # 检查最高置信度是否在相应类别的阈值区间内
            if conf_th <= max_p[i]:
                mask[i] = True
        # print("mask",mask)
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger, ada_cm):
        if 'sup' in cfg['loss_mode']:
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    output_ = model(input)
                    output_i = output_['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'], ada_cm)
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                logger.append(evaluation, 'train', n=len(input_['target']))
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
                else:
                    return None
        else:
            raise ValueError('Not valid client loss mode')

    def train(self, dataset, lr, metric, logger):
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(fix_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                             'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'batch' in cfg['loss_mode'] or 'frgd' in cfg['loss_mode'] or 'fmatch' in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if 'fmatch' in cfg['loss_mode']:
                optimizer = make_optimizer(model.make_phi_parameters(), 'local')
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    with torch.no_grad():
                        model.train(False)
                        input_ = collate(input)
                        input_ = to_device(input_, cfg['device'])
                        output_ = model(input_)
                        output_i = output_['target']
                        output_['target'] = F.softmax(output_i, dim=-1)
                        new_target, mask = self.make_hard_pseudo_label(output_['target'])
                        output_['mask'] = mask
                        evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                        logger.append(evaluation, 'train', n=len(input_['target']))
                    if torch.all(~mask):
                        continue
                    model.train(True)
                    input = {'data': input['data'][mask], 'aug': input['aug'][mask], 'target': new_target[mask]}
                    input = to_device(input, cfg['device'])
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'fix'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            raise ValueError('Not valid client loss mode')

        # 特征蒸馏部分
        self.feature_distillation(fix_dataset)

        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return

    def feature_distillation(self, fix_dataset):
        # 初始化模型
        model_g = Autoencoder(input_dim=3, output_dim=3).to(cfg["device"])

        # 加载预训练的模型权重
        model_g.load_state_dict(self.model_state_dict, strict=False)
        data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']

        # 初始化分类器
        classifier = Classifier(input_dim=3 * 128 * 128, num_classes=10).to(cfg["device"])  # 假设有10个类别

        # 优化器
        optimizer = optim.Adam(list(model_g.parameters()) + list(classifier.parameters()), lr=0.001)

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        for epoch in range(cfg['distill_epochs']):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()

                # 提取出性能敏感特征
                #print("input:", input['data'].size())
                s_feature = model_g(input['data'])  # 只使用编码器部分
                #print("s_feature:", s_feature.size())

                # 将性能敏感特征传入分类器训练
                s_feature_flat = s_feature.view(s_feature.size(0), -1)  # 展平特征

                output = classifier(s_feature_flat)

                #计算s_feature的L2范数
                l2_norms = torch.norm(s_feature_flat, p=2, dim=1).mean()
                #print("l2_norms:", l2_norms.size())
                # 分类器的损失函数计算
                loss = criterion(output, input['target'])+l2_norms
                loss.backward()
                optimizer.step()
                if epoch== cfg['distill_epochs'] - 1 and i == len(data_loader) - 1:
                    self.z_target= input['target'].detach().cpu().numpy()

        # 输出提取的加密性能敏感特征
        #好像没有传标签？
        self.z.append(s_feature.detach().cpu().numpy())
        self.z=torch.tensor(self.z)
        self.z=self.z.squeeze(0)
        self.z_target=torch.tensor(self.z_target)
        #print("z_size:",self.z.size())
        #print("z_target:",self.z_target)





def save_model_state_dict(model_state_dict):
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_