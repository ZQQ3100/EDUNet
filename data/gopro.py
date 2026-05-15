import os
from data.datasets import DatasetBase
from utils import cv_utils
import numpy as np
from collections import OrderedDict
import sys
from utils import event2frame
import operator
import torch.nn.functional as F
import torch
import random

class TrainDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(TrainDataset, self).__init__(opt, is_for_train=is_for_train)
        self._name = 'gopro'
        print('Loading gopro dataset...')
        self.opt = opt
        self.is_for_train = is_for_train
        self.dataset_acc_num = [0]
        self.dataset_acc_num_e = [0]
        self.dataset_acc_num_gt = [0]
        self._read_dataset_paths()

    def _read_dataset_paths(self):
        self.root_blur = os.path.expanduser(self.opt.input_blur_path)
        self.root = os.path.expanduser(self.opt.input_event_path)
        self.root_gt = os.path.expanduser(self.opt.input_gt_path)
        self.load_blur()
        self.load_event()
        self.load_gt()
        if not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e) or not operator.eq(self.dataset_acc_num, self.dataset_acc_num_gt):
            print('The number of blurry images is not equal to the number of eventstream or ground truth images')
            sys.exit(1)

    def load_blur(self):
        self.blur_paths = OrderedDict()
        self.dataset_name = []
        for subroot in sorted(os.listdir(self.root_blur)):
            imgroot = os.path.join(self.root_blur, subroot) 
            imglist = os.listdir(imgroot)
            imglist.sort(key=lambda x: float(x[:-4]))
            self.blur_paths[subroot] = imglist
            self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1])
            self.dataset_name.append(subroot)
    def load_event(self):
        self.event_paths = OrderedDict()
        for subroot in sorted(os.listdir(self.root)):
            eventroot = os.path.join(self.root, subroot)
            eventlist = os.listdir(eventroot)
            eventlist.sort(key=lambda x: float(x[5:-4]))
            self.event_paths[subroot] = eventlist
            self.dataset_acc_num_e.append(len(eventlist) + self.dataset_acc_num_e[-1])

    def load_gt(self):
        self.gt_paths = OrderedDict()
        for subroot in sorted(os.listdir(self.root_gt)):
            gtroot = os.path.join(self.root_gt, subroot)
            gtlist = os.listdir(gtroot)
            gtlist.sort(key=lambda x: float(x[:-4]))
            self.gt_paths[subroot] = gtlist
            self.dataset_acc_num_gt.append(len(gtlist) + self.dataset_acc_num_gt[-1])

    def __len__(self):
        return self.dataset_acc_num[-1]

    def __getitem__(self, index):
        dataset_idx = np.searchsorted(self.dataset_acc_num, index + 1)
        img_idx = index - self.dataset_acc_num[dataset_idx - 1]

        dataname = self.dataset_name[dataset_idx - 1]
        blur_paths = self.blur_paths.get(dataname)
        blur_path = os.path.join(self.root_blur, dataname, blur_paths[img_idx])
        blur = cv_utils.read_cv2_img(blur_path, input_nc=1)

        gt_paths = self.gt_paths.get(dataname)
        gt_path = os.path.join(self.root_gt, dataname, gt_paths[img_idx])
        gt = cv_utils.read_cv2_img(gt_path, input_nc=1)

        event_paths = self.event_paths.get(dataname)
        event_path = os.path.join(self.root, dataname, event_paths[img_idx])
        section_event_timestamp = cv_utils.read_mat_gopro(event_path, 'section_event_timestamp')
        section_event_polarity = cv_utils.read_mat_gopro(event_path, 'section_event_polarity')
        section_event_x = cv_utils.read_mat_gopro(event_path, 'section_event_x')  # x,y exchange
        section_event_y = cv_utils.read_mat_gopro(event_path, 'section_event_y')  # x-->[1,m]  change  x-->[0,m-1]
        start_timestamp = cv_utils.read_mat_gopro(event_path, 'start_timestamp')
        end_timestamp = cv_utils.read_mat_gopro(event_path, 'end_timestamp')
        # 如果字段是 (N,) 或 (1, N)，则将其转换为 (N, 1)
        section_event_timestamp = section_event_timestamp.reshape(-1, 1)
        section_event_polarity = section_event_polarity.reshape(-1, 1)
        section_event_x = section_event_x.reshape(-1, 1)
        section_event_y = section_event_y.reshape(-1, 1)
        section_event = np.concatenate(
            (section_event_timestamp, section_event_polarity, section_event_x, section_event_y), axis=1)
        
        event_img_lst = []
        
        events_lst = event2frame.split_events_by_time(section_event, split_num=16, start_ts=start_timestamp, end_ts=end_timestamp)

        for e_idx, events_split in enumerate(events_lst):
            event_img = event2frame.event_to_cnt_img(events_split, height=blur.shape[1], width=blur.shape[2])
            event_img_lst.append(event_img)
        event_img_bins = np.concatenate(event_img_lst)


        C, H, W = event_img_bins.shape
        h_chunk, w_chunk = 256, 256


        start_h = random.randint(0, H - h_chunk)
        start_w = random.randint(0, W - w_chunk)
        end_h = start_h + h_chunk
        end_w = start_w + w_chunk

        sample = {
            'event_bins': event_img_bins[:, start_h:end_h, start_w:end_w],
            'blurred': blur[:, start_h:end_h, start_w:end_w],
            'gt': gt[:, start_h:end_h, start_w:end_w]
        }

        return sample

class TestDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(TestDataset, self).__init__(opt, is_for_train=is_for_train)
        self._name = 'gopro'
        print('Loading dataset...')
        self.opt = opt
        self.is_for_train = is_for_train
        self.dataset_acc_num = [0]
        self.dataset_acc_num_e = [0]
        self._read_dataset_paths()

    def _read_dataset_paths(self):
        self.root_blur = os.path.expanduser(self.opt.input_blur_path)
        self.root = os.path.expanduser(self.opt.input_event_path)
        self.load_blur()
        self.load_event()
        if not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e):
            print('The number of blurry images is not equal to the number of eventstream')
            sys.exit(1)

    def load_blur(self):
        self.blur_paths = OrderedDict()
        self.dataset_name = []
        for subroot in sorted(os.listdir(self.root_blur)):
            imgroot = os.path.join(self.root_blur, subroot)
            imglist = os.listdir(imgroot)
            imglist.sort(key=lambda x: float(x[:-4]))
            self.blur_paths[subroot] = imglist
            self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1])
            self.dataset_name.append(subroot)

    def load_event(self):
        self.event_paths = OrderedDict()
        for subroot in sorted(os.listdir(self.root)):
            eventroot = os.path.join(self.root, subroot)
            eventlist = os.listdir(eventroot)
            eventlist.sort(key=lambda x: float(x[5:-4]))
            self.event_paths[subroot] = eventlist
            self.dataset_acc_num_e.append(len(eventlist) + self.dataset_acc_num_e[-1])

    def __len__(self):
        return self.dataset_acc_num[-1]

    def __getitem__(self, index):
        dataset_idx = np.searchsorted(self.dataset_acc_num, index + 1)
        img_idx = index - self.dataset_acc_num[dataset_idx - 1]

        dataname = self.dataset_name[dataset_idx - 1]
        blur_paths = self.blur_paths.get(dataname)
        blur_path = os.path.join(self.root_blur, dataname, blur_paths[img_idx])
        blur = cv_utils.read_cv2_img(blur_path, input_nc=1)
        # 修改为（读RGB图）：
        # blur = cv_utils.read_cv2_img(blur_path, input_nc=3)

        event_paths = self.event_paths.get(dataname)
        event_path = os.path.join(self.root, dataname, event_paths[img_idx])
        section_event_timestamp = cv_utils.read_mat_gopro(event_path, 'section_event_timestamp')
        section_event_polarity = cv_utils.read_mat_gopro(event_path, 'section_event_polarity')
        section_event_x = cv_utils.read_mat_gopro(event_path, 'section_event_x')  # x,y exchange
        section_event_y = cv_utils.read_mat_gopro(event_path, 'section_event_y')  # x-->[1,m]  change  x-->[0,m-1]
        start_timestamp = cv_utils.read_mat_gopro(event_path, 'start_timestamp')
        end_timestamp = cv_utils.read_mat_gopro(event_path, 'end_timestamp')
        # 如果字段是 (N,) 或 (1, N)，则将其转换为 (N, 1)
        section_event_timestamp = section_event_timestamp.reshape(-1, 1)
        section_event_polarity = section_event_polarity.reshape(-1, 1)
        section_event_x = section_event_x.reshape(-1, 1)
        section_event_y = section_event_y.reshape(-1, 1)
        section_event = np.concatenate(
            (section_event_timestamp, section_event_polarity, section_event_x, section_event_y), axis=1)

        event_img_lst = []
        events_lst = event2frame.split_events_by_time(section_event, split_num=16, start_ts=start_timestamp, end_ts=end_timestamp)
        for e_idx, events_split in enumerate(events_lst):
            event_img = event2frame.event_to_cnt_img(events_split, height=blur.shape[1], width=blur.shape[2])
            event_img_lst.append(event_img)
        event_img_bins = np.concatenate(event_img_lst)

        sample = {
            'event_bins': event_img_bins,
            'blurred': blur,
            'dataname': dataname,
            'img_idx': img_idx
        }
        return sample