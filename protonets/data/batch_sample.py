import os
import sys
import glob

from functools import partial

import numpy as np


import torch

from torch.utils.data import Dataset, DataLoader
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler, \
    extract_episode

KWS_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/asr/data')

KWS_DATA_DIR_TEST = os.path.join(os.path.dirname(__file__), '../../data/asr/data_test')
KWS_CACHE = { }


def load_class_features(path, d):
    if d['class'] not in KWS_CACHE:

        files = sorted(os.listdir(path))

        class_names = []
        feat_ds = []
        for file in files:
            #class_name = file.split('_')[0]
            if file.__contains__(d['class']):
                class_names.append(file)
                #feat = np.load(os.path.join(KWS_DATA_DIR, file))
                feat = np.load(os.path.join(path, file))
                if len(feat) > 160:
                    feat = feat[:160]
                elif len(feat) < 160:
                    feat = np.concatenate((feat, np.zeros((160 - len(feat), feat.shape[1]))), axis=0)
                feat_ds.append(feat)

        sample = {}
        sample['file_name']=class_names
        sample['data']=torch.Tensor(feat_ds).unsqueeze(dim=1)
        KWS_CACHE[d['class']] = sample['data']

    return { 'class': d['class'], 'data': KWS_CACHE[d['class']] }

def load_kws(opt, splits):
    #split_dir = os.path.join(KWS_DATA_DIR, 'splits', opt['data.split'])
    dataset_self = {}
    if splits[0]=='test':
        files = sorted(os.listdir(KWS_DATA_DIR_TEST))
        class_names = []
        for file in files:
            class_name = file.split('_')[0]
            if not class_names.__contains__(class_name):
                class_names.append(class_name)
        dataset_self['test'] = class_names
        data_dir = KWS_DATA_DIR_TEST
    else:
        data_dir = KWS_DATA_DIR
        files=sorted(os.listdir(KWS_DATA_DIR))
        val_class_names=['label01', 'label13','label03', 'label13', 'label03', 'label13', 'label03', 'label03']
        class_names=[]
        for file in files:
            class_name = file.split('_')[0]
            if not class_names.__contains__(class_name) and not val_class_names.__contains__(class_name):
                class_names.append(class_name)
        train_data={}
        for name in class_names:
            name_files = []
            for file in files:
                if file.__contains__(name):
                    name_files.append(file)
            train_data[name] = name_files

        val_data = {}
        for name in val_class_names:
            name_files = []
            for file in files:
                if file.__contains__(name):
                    name_files.append(file)
            val_data[name] = name_files

        dataset_self['train'] = class_names
        dataset_self['val'] = val_class_names
    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_features,data_dir),
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)
        ds = TransformDataset(ListDataset(dataset_self[split]), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)


    return ret