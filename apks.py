from settings import config
import numpy as np
import os
from common.util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import logging
from scipy import sparse
import math

class ApkData:
    """The Dataset for training the malware detection methods"""

    def __init__(self, detection, granularity, classifier, attacker, base_clf_dir=config['base_clf_dir'], data_source=config['data_source'], dim_to_be_changed = True, time_test_size = 10, args=None):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_adv = None
        self.y_adv = None
        self.detection = detection
        self.granularity = granularity
        self.classifier = classifier
        self.attacker = attacker
        self.data_source = data_source

        self.base_clf_name = "_".join([detection, 'apg', classifier]) + ".model"
        self.base_clf_dir = base_clf_dir
        self.base_clf_path = None
        self.base_clf = None
        self.base_clf_predict_y_train = None

        self.load_data(detection, granularity, classifier, attacker, data_source, args)

        # 设置随机种子
        random_seed = 256
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # change data dim
        if dim_to_be_changed:
            if detection == "drebin" or detection == "apigraph":
                self.x_train = self.x_train.A
                self.x_test = self.x_test.A
                self.x_adv = self.x_adv.A

            dim = math.ceil(math.sqrt(self.x_train.shape[1]))
            dim_pad = int(math.pow(dim, 2)) - self.x_train.shape[1]
            self.x_train = np.pad(self.x_train, ((0, 0), (0, dim_pad)), 'constant', constant_values=(0, 0))
            self.x_train.resize((self.x_train.shape[0], dim, dim, 1))

            self.x_test = np.pad(self.x_test, ((0, 0), (0, dim_pad)), 'constant', constant_values=(0, 0))
            self.x_test.resize((self.x_test.shape[0], dim, dim, 1))

            self.x_adv = np.pad(self.x_adv, ((0, 0), (0, dim_pad)), 'constant', constant_values=(0, 0))
            self.x_adv.resize((self.x_adv.shape[0], dim, dim, 1))

            # train_loader、test_loader
            self.train_loader = DataLoader(TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.y_train).float()), shuffle=True, batch_size=64)
            self.test_loader = DataLoader(TensorDataset(torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float()), shuffle=True, batch_size=10)
            self.time_test_loader = DataLoader(TensorDataset(torch.tensor(self.x_train[:time_test_size]).float(), torch.tensor(self.y_train[:time_test_size]).float()), shuffle=True, batch_size=128)
        else:
            # train_loader、test_loader
            if detection == "drebin" or detection == "apigraph":
                self.train_loader = DataLoader(TensorDataset(torch.tensor(self.x_train.toarray()).float(), torch.tensor(self.base_clf_predict_y_train).float()), shuffle=True, batch_size=64)
                self.test_loader = DataLoader(TensorDataset(torch.tensor(self.x_test.toarray()).float(), torch.tensor(self.y_test).float()), shuffle=False, batch_size=128)
                self.time_test_loader = DataLoader(
                    TensorDataset(torch.tensor(self.x_train[:time_test_size].toarray()).float(), torch.tensor(self.y_train[:time_test_size]).float()),
                    shuffle=True, batch_size=10)
            else:
                self.train_loader = DataLoader(
                    TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.base_clf_predict_y_train).float()), shuffle=True,
                    batch_size=64)
                self.test_loader = DataLoader(
                    TensorDataset(torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float()), shuffle=False,
                    batch_size=10)
                self.time_test_loader = DataLoader(
                    TensorDataset(torch.tensor(self.x_train[:time_test_size]).float(), torch.tensor(self.y_train[:time_test_size]).float()),
                    shuffle=True, batch_size=128)

    def load_data(self, detection, granularity, classifier, attacker, data_source, args, isOneHotEncoding = True):
        detection_granularity_dir = detection + ('_' + granularity if granularity != '' else '')

        attacker_for_train_test = 'BagAmmo' if attacker in {'HIV_CW', 'HIV_JSMA'} else attacker        

        if detection == "drebin" or detection == "apigraph":
            x_train = sparse.load_npz(os.path.join(data_source, attacker_for_train_test, 'train_sample', detection, detection + '_' + classifier + '_train.npz'))
            x_test = sparse.load_npz(os.path.join(data_source, attacker_for_train_test, 'test_sample', detection,
                                          detection + '_' + classifier + '_test.npz'))
            x_adv = sparse.load_npz(os.path.join(data_source, attacker, 'attack_sample', detection, detection + '_' + classifier + '_attack.npz'))

        elif detection == "malscan" or detection == "mamadroid" or detection == "vae_fd":
            x_train = np.load(os.path.join(data_source, attacker_for_train_test, 'train_sample',
                                           detection_granularity_dir, detection + '_' + classifier + '_train.npy'))
            x_test = np.load(os.path.join(data_source, attacker_for_train_test, 'test_sample',
                                          detection_granularity_dir, detection + '_' + classifier + '_test.npy'))
            x_adv = np.load(os.path.join(data_source, attacker, 'attack_sample',
                                         detection_granularity_dir, detection + '_' + classifier + '_attack.npy'))
            if args and args.adaptive_attack:
                x_adv = np.load(os.path.join(data_source, attacker, 'attack_sample',
                                             detection_granularity_dir, detection + '_' + args.adaptive_attack + '_' + classifier + '_attack.npy'))
            if len(x_adv.shape) > 2:
                x_adv.resize(x_adv.shape[0], x_adv.shape[1] * x_adv.shape[2])
        y_test = np.load(os.path.join(data_source, attacker_for_train_test, 'test_sample',
                                      detection_granularity_dir, detection + '_' + classifier + '_test_label.npy'))
        y_train = np.load(os.path.join(data_source, attacker_for_train_test, 'train_sample',
                                       detection_granularity_dir, detection + '_' + classifier + '_train_label.npy'))
        y_adv = np.load(os.path.join(data_source, attacker, 'attack_sample',
                                     detection_granularity_dir, detection + '_' + classifier + '_attack_label.npy'))
        if args and args.adaptive_attack:
            y_adv = np.load(os.path.join(data_source, attacker, 'attack_sample',
                                         detection_granularity_dir, detection + '_' + args.adaptive_attack + '_' + classifier + '_attack_label.npy'))

        #convert labels to one_hot
        if isOneHotEncoding:
            self.num_classes = 2
            y_adv = y_adv.astype(int)
            self.y_train = to_onehot_encode(y_train, self.num_classes)
            self.y_test = to_onehot_encode(y_test, self.num_classes)
            self.y_adv = to_onehot_encode(y_adv, self.num_classes)

        self.x_train = x_train
        self.x_test = x_test
        self.x_adv = x_adv

        if attacker == 'AdvDroidZero':
            self.base_clf_path = os.path.join(self.base_clf_dir, attacker, self.base_clf_name)
            if os.path.exists(self.base_clf_path + ".clf"):
                logging.debug(blue('Loading model from {}...'.format(self.base_clf_path + ".clf")))
                with open(self.base_clf_path + ".clf", "rb") as f:
                    self.base_clf = pickle.load(f)

            base_clf_predict_y_train = self.base_clf.predict(self.x_train)
            self.base_clf_predict_y_train = to_onehot_encode(base_clf_predict_y_train, self.num_classes)
        else:
            base_clf_predict_y_train = np.load(os.path.join(data_source, attacker_for_train_test, 'train_sample',
                                           detection_granularity_dir, detection + '_' + classifier + '_train_pred_label.npy'))
            self.base_clf_predict_y_train = to_onehot_encode(base_clf_predict_y_train, self.num_classes)

    def mixData(self):
        return


    def __iter__(self):
        return iter(self.data)


