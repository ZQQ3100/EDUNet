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
        self.dataset_acc_num_gt = [0] # ground truth 图像数量的累积列表
        self._read_dataset_paths()

    def _read_dataset_paths(self):
        self.root_blur = os.path.expanduser(self.opt.input_blur_path) # D:\ZQQ\111\test_data\gopro_test\blur_images
        self.root = os.path.expanduser(self.opt.input_event_path) # D:\ZQQ\111\test_data\gopro_test\eventstream_mat
        self.root_gt = os.path.expanduser(self.opt.input_gt_path)  # ground truth 图像路径
        self.load_blur()
        self.load_event()
        self.load_gt()  # 加载 ground truth 图像
        if not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e) or not operator.eq(self.dataset_acc_num, self.dataset_acc_num_gt):
            print('The number of blurry images is not equal to the number of eventstream or ground truth images')
            sys.exit(1)

    def load_blur(self):
        self.blur_paths = OrderedDict() # 字典
        self.dataset_name = []
        for subroot in sorted(os.listdir(self.root_blur)): # 返回一个列表，其中每个元素是一个字符串，代表目录中的文件名或子目录名。
            # subroot是子文件名000、001、002...
            imgroot = os.path.join(self.root_blur, subroot) # 子文件夹的路径：D:\ZQQ\111\test_data\gopro_test\blur_images\000
            imglist = os.listdir(imgroot) # 返回的list中每个元素是一张图像的名称['00000016.png', '00000019.png', '00000022.png']
            imglist.sort(key=lambda x: float(x[:-4])) # 去掉扩展名后的字符串转换为浮点数进行排序。
            self.blur_paths[subroot] = imglist # 子文件夹000做key，该子文件夹下的图像列表做值
            self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1]) #最后一项就是最新的dataset number
            self.dataset_name.append(subroot) # ['000','001','002']

    def load_event(self):
        self.event_paths = OrderedDict()
        for subroot in sorted(os.listdir(self.root)):
            eventroot = os.path.join(self.root, subroot)
            eventlist = os.listdir(eventroot)
            eventlist.sort(key=lambda x: float(x[5:-4]))
            self.event_paths[subroot] = eventlist
            self.dataset_acc_num_e.append(len(eventlist) + self.dataset_acc_num_e[-1])

    def load_gt(self):  # 加载 ground truth 图像
        self.gt_paths = OrderedDict()
        for subroot in sorted(os.listdir(self.root_gt)):
            gtroot = os.path.join(self.root_gt, subroot)
            gtlist = os.listdir(gtroot)
            gtlist.sort(key=lambda x: float(x[:-4]))
            self.gt_paths[subroot] = gtlist
            self.dataset_acc_num_gt.append(len(gtlist) + self.dataset_acc_num_gt[-1])

    # def extract_patches(self, img_tensor, patch_size):
    #     """从张量中提取固定大小的分块"""
    #     _, img_height, img_width = img_tensor.size()
    #     patch_height, patch_width = 256, 256
    #
    #     # 计算分块的数量，并考虑最后一行/列可能不足的情况
    #     num_patches_h = -(-img_height // patch_height)  # 向上取整
    #     num_patches_w = -(-img_width // patch_width)  # 向上取整
    #
    #     patches = []
    #     for i in range(num_patches_h):
    #         for j in range(num_patches_w):
    #             start_h = i * patch_height
    #             start_w = j * patch_width
    #             end_h = min(start_h + patch_height, img_height)
    #             end_w = min(start_w + patch_width, img_width)
    #
    #             patch = img_tensor[:, start_h:end_h, start_w:end_w]
    #
    #             # 如果分块小于目标尺寸，则进行填充
    #             if patch.size(1) < patch_height or patch.size(2) < patch_width:
    #                 patch = F.pad(patch, [0, max(0, patch_width - patch.size(2)),
    #                                       0, max(0, patch_height - patch.size(1))], "constant", 0)
    #             patches.append(patch)
    #     return torch.stack(patches)

    def __len__(self):
        return self.dataset_acc_num[-1]

    def __getitem__(self, index):
        # np.searchsorted 用于查找 index + 1 在 self.dataset_acc_num 中应该插入的位置，以保持数组的有序性。
        # index + 1 的目的是为了确保当 index 恰好等于某个累积计数时，searchsorted 返回正确的子数据集索引。如果不加 1，可能会导致 index 落入错误的子数据集。
        # - 1 是因为 searchsorted 返回的是插入位置，而我们想要的是最后一个累积计数小于等于 index + 1 的位置，即当前子数据集的索引。
        dataset_idx = np.searchsorted(self.dataset_acc_num, index + 1)  # 这是子数据集的索引，表示 index 对应的子数据集。
        img_idx = index - self.dataset_acc_num[dataset_idx - 1] # 这是在子数据集中图像的索引，表示 index 对应的具体图像。

        dataname = self.dataset_name[dataset_idx - 1] # 子数据集名000、001
        # blurred images
        blur_paths = self.blur_paths.get(dataname) # imagelist['00000016.png', '00000019.png', '00000022.png']
        # index对应blur图像的完整路径D:\ZQQ\111\test_data\gopro_test\blur_images\000\00000016.png
        blur_path = os.path.join(self.root_blur, dataname, blur_paths[img_idx])
        blur = cv_utils.read_cv2_img(blur_path, input_nc=1) # 读取这张模糊图像,这里就归一化了

        # ground truth images
        gt_paths = self.gt_paths.get(dataname)  # gtlist['00000016.png', '00000019.png', '00000022.png']
        # index对应gt图像的完整路径D:\ZQQ\111\test_data\gopro_test\gt\000\00000016.png
        gt_path = os.path.join(self.root_gt, dataname, gt_paths[img_idx])
        gt = cv_utils.read_cv2_img(gt_path, input_nc=1)

        # event images
        event_paths = self.event_paths.get(dataname) # eventlist['event00000016.mat', 'event00000019.mat', 'event00000022.mat']
        # index对应event的完整路径D:\ZQQ\111\test_data\gopro_test\eventstream_mat\000\event00000016.mat
        event_path = os.path.join(self.root, dataname, event_paths[img_idx])
        section_event_timestamp = cv_utils.read_mat_gopro(event_path, 'section_event_timestamp')
        section_event_polarity = cv_utils.read_mat_gopro(event_path, 'section_event_polarity')
        section_event_x = cv_utils.read_mat_gopro(event_path, 'section_event_x')  # x,y exchange
        section_event_y = cv_utils.read_mat_gopro(event_path, 'section_event_y')  # x-->[1,m]  change  x-->[0,m-1]
        start_timestamp = cv_utils.read_mat_gopro(event_path, 'start_timestamp')
        end_timestamp = cv_utils.read_mat_gopro(event_path, 'end_timestamp')
        # print(section_event_x.shape)
        # 如果字段是 (N,) 或 (1, N)，则将其转换为 (N, 1)
        section_event_timestamp = section_event_timestamp.reshape(-1, 1)
        section_event_polarity = section_event_polarity.reshape(-1, 1)
        section_event_x = section_event_x.reshape(-1, 1)
        section_event_y = section_event_y.reshape(-1, 1)
        # print("x:", max(section_event_x))
        # print("y:", max(section_event_y))
        section_event = np.concatenate(
            (section_event_timestamp, section_event_polarity, section_event_x, section_event_y), axis=1)
        # Timestamps:[[1.][2.][3.]]
        # Polarities:[[1][0][1]]
        # X coordinates:[[10][20][30]]
        # Y coordinates:[[100][200][300]]
        # Combined events:[[ 1.  1. 10. 100.][ 2.  0. 20. 200.][ 3.  1. 30. 300.]]

        # if self.opt.num_frames_for_blur < 2:
        #     num_frames = 2
        # else:
        #     num_frames = self.opt.num_frames_for_blur

        event_img_lst = []
        # split_events_by_time 函数将事件数据分为“中间时刻之前”（reversal）和“中间时刻之后”（shift）两部分
        # 对于发生在中间时刻之前的事件 event_reversal 进行极性反转
        # 根据指定的时间间隔split_num对两部分进行进一步的时间分割。
        # 最终返回一个包含所有子事件列表的列表 events_lst --> split_num*2
        events_lst = event2frame.split_events_by_time(section_event, split_num=16, start_ts=start_timestamp, end_ts=end_timestamp)
        # print(events_lst[0].shape)
        for e_idx, events_split in enumerate(events_lst): # events_split(N, 4)
            # event_to_cnt_img 函数用于将事件数据转换为事件计数图像（Event Count Image）。
            # 根据事件的极性（正或负）分别统计每个像素位置上的事件数量，并生成两个通道的图像：一个通道表示正事件的数量，另一个通道表示负事件的数量。
            # 最终返回一个形状为 (2, height, width) 的三维数组，其中第一个维度表示极性通道。
            event_img = event2frame.event_to_cnt_img(events_split, height=blur.shape[1], width=blur.shape[2])
            event_img_lst.append(event_img) # split_num*2个元素的列表，每个元素都是形状为 (2, height, width) 的三维数组
        # event_img_acc_lst = event2frame.accumulate_event_images(event_img_lst, split_num=16)
        event_img_bins = np.concatenate(event_img_lst) # 合并之后变成形状为(split_num*2*2(64), height(720), width(1280)) 的三维数组
        # print("min:", event_img_bins.min().item())
        # print("max:", event_img_bins.max().item())
        # # 分块处理
        # event_img_bins = torch.tensor(event_img_bins, dtype=torch.float32)
        # blur = torch.tensor(blur, dtype=torch.float32)
        # gt = torch.tensor(gt, dtype=torch.float32)
        # event_patches = self.extract_patches(event_img_bins, patch_size=(256, 256))
        # blur_patches = self.extract_patches(blur, patch_size=(256, 256))
        # gt_patches = self.extract_patches(gt, patch_size=(256, 256))
        # print(gt_patches.shape)

        # 随机分块
        C, H, W = event_img_bins.shape
        h_chunk, w_chunk = 256, 256

        # 如果数据少于或等于所需的块大小，则直接返回全部数据
        # if H <= h_chunk or W <= w_chunk:

        start_h = random.randint(0, H - h_chunk)
        start_w = random.randint(0, W - w_chunk)
        end_h = start_h + h_chunk
        end_w = start_w + w_chunk

        sample = {
            'event_bins': event_img_bins[:, start_h:end_h, start_w:end_w],
            'blurred': blur[:, start_h:end_h, start_w:end_w],
            'gt': gt[:, start_h:end_h, start_w:end_w]
        }

        # sample = {
        #     'event_bins': event_img_bins, # (64,720,1280)
        #     'blurred': blur, # 一张模糊图像(1,720,1280)
        #     'gt': gt, # 一张清晰图像(1,720,1280)
        #     # 'dataname': dataname, # 子数据集名000、001(字符串)
        #     # 'img_idx': img_idx # 在子数据集中图像的索引(整数)
        # }
        # sample = {
        #     'event_bins': event_patches,  # (num_patches(15), 64,720,1280)
        #     'blurred': blur_patches,  # 一组模糊图像块(num_patches(15), 1,720,1280)
        #     'gt': gt_patches,  # 一组清晰图像块(num_patches(15), 1,720,1280)
        # }
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
        self.root_blur = os.path.expanduser(self.opt.input_blur_path) # D:\ZQQ\111\test_data\gopro_test\blur_images
        self.root = os.path.expanduser(self.opt.input_event_path) # D:\ZQQ\111\test_data\gopro_test\eventstream_mat
        self.load_blur()
        self.load_event()
        if not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e):
            print('The number of blurry images is not equal to the number of eventstream')
            sys.exit(1)

    def load_blur(self):
        self.blur_paths = OrderedDict() # 字典
        self.dataset_name = []
        for subroot in sorted(os.listdir(self.root_blur)): # 返回一个列表，其中每个元素是一个字符串，代表目录中的文件名或子目录名。
            # subroot是子文件名000、001、002...
            imgroot = os.path.join(self.root_blur, subroot) # 子文件夹的路径：D:\ZQQ\111\test_data\gopro_test\blur_images\000
            imglist = os.listdir(imgroot) # 返回的list中每个元素是一张图像的名称['00000016.png', '00000019.png', '00000022.png']
            imglist.sort(key=lambda x: float(x[:-4])) # 去掉扩展名后的字符串转换为浮点数进行排序。
            self.blur_paths[subroot] = imglist # 子文件夹000做key，该子文件夹下的图像列表做值
            self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1]) #最后一项就是最新的dataset number
            self.dataset_name.append(subroot) # ['000','001','002']

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
        # np.searchsorted 用于查找 index + 1 在 self.dataset_acc_num 中应该插入的位置，以保持数组的有序性。
        # index + 1 的目的是为了确保当 index 恰好等于某个累积计数时，searchsorted 返回正确的子数据集索引。如果不加 1，可能会导致 index 落入错误的子数据集。
        # - 1 是因为 searchsorted 返回的是插入位置，而我们想要的是最后一个累积计数小于等于 index + 1 的位置，即当前子数据集的索引。
        dataset_idx = np.searchsorted(self.dataset_acc_num, index + 1)  # 这是子数据集的索引，表示 index 对应的子数据集。
        img_idx = index - self.dataset_acc_num[dataset_idx - 1] # 这是在子数据集中图像的索引，表示 index 对应的具体图像。

        dataname = self.dataset_name[dataset_idx - 1] # 子数据集名000、001
        # blurred images
        blur_paths = self.blur_paths.get(dataname) # imagelist['00000016.png', '00000019.png', '00000022.png']
        # index对应blur图像的完整路径D:\ZQQ\111\test_data\gopro_test\blur_images\000\00000016.png
        blur_path = os.path.join(self.root_blur, dataname, blur_paths[img_idx])
        blur = cv_utils.read_cv2_img(blur_path, input_nc=1) # 读取这张模糊图像
        # 修改为（读RGB图）：
        # blur = cv_utils.read_cv2_img(blur_path, input_nc=3)

        # event images
        event_paths = self.event_paths.get(dataname) # eventlist['event00000016.mat', 'event00000019.mat', 'event00000022.mat']
        # index对应event的完整路径D:\ZQQ\111\test_data\gopro_test\eventstream_mat\000\event00000016.mat
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
        # Timestamps:[[1.][2.][3.]]
        # Polarities:[[1][0][1]]
        # X coordinates:[[10][20][30]]
        # Y coordinates:[[100][200][300]]
        # Combined events:[[ 1.  1. 10. 100.][ 2.  0. 20. 200.][ 3.  1. 30. 300.]]

        # if self.opt.num_frames_for_blur < 2:
        #     num_frames = 2
        # else:
        #     num_frames = self.opt.num_frames_for_blur

        event_img_lst = []
        # split_events_by_time 函数将事件数据分为“中间时刻之前”（reversal）和“中间时刻之后”（shift）两部分
        # 对于发生在中间时刻之前的事件 event_reversal 进行极性反转
        # 根据指定的时间间隔split_num对两部分进行进一步的时间分割。
        # 最终返回一个包含所有子事件列表的列表 events_lst --> split_num*2
        events_lst = event2frame.split_events_by_time(section_event, split_num=16, start_ts=start_timestamp, end_ts=end_timestamp)
        for e_idx, events_split in enumerate(events_lst):
            # event_to_cnt_img 函数用于将事件数据转换为事件计数图像（Event Count Image）。
            # 根据事件的极性（正或负）分别统计每个像素位置上的事件数量，并生成两个通道的图像：一个通道表示正事件的数量，另一个通道表示负事件的数量。
            # 最终返回一个形状为 (2, height, width) 的三维数组，其中第一个维度表示极性通道。
            event_img = event2frame.event_to_cnt_img(events_split, height=blur.shape[1], width=blur.shape[2])
            event_img_lst.append(event_img) # split_num*2个元素的列表，每个元素都是形状为 (2, height, width) 的三维数组
        # event_img_acc_lst = event2frame.accumulate_event_images(event_img_lst, split_num=16)
        event_img_bins = np.concatenate(event_img_lst) # 合并之后变成形状为(split_num*2*2(64), height(720), width(1280)) 的三维数组

        # # 随机分块
        # C, H, W = event_img_bins.shape
        # h_chunk, w_chunk = 256, 256
        #
        # # 如果数据少于或等于所需的块大小，则直接返回全部数据
        # # if H <= h_chunk or W <= w_chunk:
        #
        # start_h = random.randint(0, H - h_chunk)
        # start_w = random.randint(0, W - w_chunk)
        # end_h = start_h + h_chunk
        # end_w = start_w + w_chunk
        #
        # sample = {
        #     'event_bins': event_img_bins[:, start_h:end_h, start_w:end_w],
        #     'blurred': blur[:, start_h:end_h, start_w:end_w],
        #     'dataname': dataname,  # 子数据集名000、001(字符串)
        #     'img_idx': img_idx # 在子数据集中图像的索引(整数)
        # }
        sample = {
            'event_bins': event_img_bins, # (64,720,1280)
            'blurred': blur, # 一张模糊图像(1,720,1280)
            'dataname': dataname, # 子数据集名000、001(字符串)
            'img_idx': img_idx # 在子数据集中图像的索引(整数)
        }
        return sample

# class Dataset(DatasetBase):
#     def __init__(self, opt, is_for_train):
#         super(Dataset, self).__init__(opt, is_for_train=is_for_train)
#         self._name = 'gopro'
#         print('Loading dataset...')
#         self.opt = opt
#         self.is_for_train = is_for_train
#         self.dataset_acc_num = [0]
#         self.dataset_acc_num_e = [0]
#         self.dataset_acc_num_gt = [0] if self.is_for_train else None  # 新增：ground truth 图像数量的累积列表
#         self._read_dataset_paths()
#
#     def _read_dataset_paths(self):
#         self.root_blur = os.path.expanduser(self.opt.input_blur_path) # D:\ZQQ\111\test_data\gopro_test\blur_images
#         self.root = os.path.expanduser(self.opt.input_event_path) # D:\ZQQ\111\test_data\gopro_test\eventstream_mat
#         self.load_blur()
#         self.load_event()
#         if self.is_for_train:
#             self.root_gt = os.path.expanduser(self.opt.input_gt_path)  # 新增：ground truth 图像路径
#             self.load_gt()  # 新增：加载 ground truth 图像
#         if self.is_for_train and (not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e) or not operator.eq(self.dataset_acc_num, self.dataset_acc_num_gt)):
#             print('The number of blurry images is not equal to the number of eventstream or ground truth images')
#             sys.exit(1)
#         elif not self.is_for_train and not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e):
#             print('The number of blurry images is not equal to the number of eventstream')
#             sys.exit(1)
#
#     def load_blur(self):
#         self.blur_paths = OrderedDict() # 字典
#         self.dataset_name = []
#         for subroot in sorted(os.listdir(self.root_blur)): # 返回一个列表，其中每个元素是一个字符串，代表目录中的文件名或子目录名。
#             # subroot是子文件名000、001、002...
#             imgroot = os.path.join(self.root_blur, subroot) # 子文件夹的路径：D:\ZQQ\111\test_data\gopro_test\blur_images\000
#             imglist = os.listdir(imgroot) # 返回的list中每个元素是一张图像的名称['00000016.png', '00000019.png', '00000022.png']
#             imglist.sort(key=lambda x: float(x[:-4])) # 去掉扩展名后的字符串转换为浮点数进行排序。
#             self.blur_paths[subroot] = imglist # 子文件夹000做key，该子文件夹下的图像列表做值
#             self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1]) #最后一项就是最新的dataset number
#             self.dataset_name.append(subroot) # ['000','001','002']
#
#     def load_event(self):
#         self.event_paths = OrderedDict()
#         for subroot in sorted(os.listdir(self.root)):
#             eventroot = os.path.join(self.root, subroot)
#             eventlist = os.listdir(eventroot)
#             eventlist.sort(key=lambda x: float(x[5:-4]))
#             self.event_paths[subroot] = eventlist
#             self.dataset_acc_num_e.append(len(eventlist) + self.dataset_acc_num_e[-1])
#
#     def load_gt(self):  # 新增：加载 ground truth 图像
#         self.gt_paths = OrderedDict()
#         for subroot in sorted(os.listdir(self.root_gt)):
#             gtroot = os.path.join(self.root_gt, subroot)
#             gtlist = os.listdir(gtroot)
#             gtlist.sort(key=lambda x: float(x[:-4]))
#             self.gt_paths[subroot] = gtlist
#             self.dataset_acc_num_gt.append(len(gtlist) + self.dataset_acc_num_gt[-1])
#
#     def __len__(self):
#         return self.dataset_acc_num[-1]
#
#     def __getitem__(self, index):
#         # np.searchsorted 用于查找 index + 1 在 self.dataset_acc_num 中应该插入的位置，以保持数组的有序性。
#         # index + 1 的目的是为了确保当 index 恰好等于某个累积计数时，searchsorted 返回正确的子数据集索引。如果不加 1，可能会导致 index 落入错误的子数据集。
#         # - 1 是因为 searchsorted 返回的是插入位置，而我们想要的是最后一个累积计数小于等于 index + 1 的位置，即当前子数据集的索引。
#         dataset_idx = np.searchsorted(self.dataset_acc_num, index + 1)  # 这是子数据集的索引，表示 index 对应的子数据集。
#         img_idx = index - self.dataset_acc_num[dataset_idx - 1] # 这是在子数据集中图像的索引，表示 index 对应的具体图像。
#
#         dataname = self.dataset_name[dataset_idx - 1] # 子数据集名000、001
#         # blurred images
#         blur_paths = self.blur_paths.get(dataname) # imagelist['00000016.png', '00000019.png', '00000022.png']
#         # index对应blur图像的完整路径D:\ZQQ\111\test_data\gopro_test\blur_images\000\00000016.png
#         blur_path = os.path.join(self.root_blur, dataname, blur_paths[img_idx])
#         blur = cv_utils.read_cv2_img(blur_path, input_nc=1) # 读取这张模糊图像
#
#         # event images
#         event_paths = self.event_paths.get(dataname) # eventlist['event00000016.mat', 'event00000019.mat', 'event00000022.mat']
#         # index对应event的完整路径D:\ZQQ\111\test_data\gopro_test\eventstream_mat\000\event00000016.mat
#         event_path = os.path.join(self.root, dataname, event_paths[img_idx])
#         section_event_timestamp = cv_utils.read_mat_gopro(event_path, 'section_event_timestamp')
#         section_event_polarity = cv_utils.read_mat_gopro(event_path, 'section_event_polarity')
#         section_event_x = cv_utils.read_mat_gopro(event_path, 'section_event_x') - 1  # x,y exchange
#         section_event_y = cv_utils.read_mat_gopro(event_path, 'section_event_y') - 1  # x-->[1,m]  change  x-->[0,m-1]
#         start_timestamp = cv_utils.read_mat_gopro(event_path, 'start_timestamp')
#         end_timestamp = cv_utils.read_mat_gopro(event_path, 'end_timestamp')
#         section_event = np.concatenate(
#             (section_event_timestamp, section_event_polarity, section_event_x, section_event_y), axis=1)
#         # Timestamps:[[1.][2.][3.]]
#         # Polarities:[[1][0][1]]
#         # X coordinates:[[10][20][30]]
#         # Y coordinates:[[100][200][300]]
#         # Combined events:[[ 1.  1. 10. 100.][ 2.  0. 20. 200.][ 3.  1. 30. 300.]]
#
#         # if self.opt.num_frames_for_blur < 2:
#         #     num_frames = 2
#         # else:
#         #     num_frames = self.opt.num_frames_for_blur
#
#         event_img_lst = []
#         # split_events_by_time 函数将事件数据分为“中间时刻之前”（reversal）和“中间时刻之后”（shift）两部分
#         # 对于发生在中间时刻之前的事件 event_reversal 进行极性反转
#         # 根据指定的时间间隔split_num对两部分进行进一步的时间分割。
#         # 最终返回一个包含所有子事件列表的列表 events_lst --> split_num*2
#         events_lst = event2frame.split_events_by_time(section_event, split_num=16, start_ts=start_timestamp, end_ts=end_timestamp)
#         for e_idx, events_split in enumerate(events_lst):
#             # event_to_cnt_img 函数用于将事件数据转换为事件计数图像（Event Count Image）。
#             # 根据事件的极性（正或负）分别统计每个像素位置上的事件数量，并生成两个通道的图像：一个通道表示正事件的数量，另一个通道表示负事件的数量。
#             # 最终返回一个形状为 (2, height, width) 的三维数组，其中第一个维度表示极性通道。
#             event_img = event2frame.event_to_cnt_img(events_split, height=blur.shape[1], width=blur.shape[2])
#             event_img_lst.append(event_img) # split_num*2个元素的列表，每个元素都是形状为 (2, height, width) 的三维数组
#
#         event_img_bins = np.concatenate(event_img_lst) # 合并之后变成形状为(split_num*2*2(64), height(720), width(1280)) 的三维数组
#
#         sample = {
#             'event_bins': event_img_bins, # (64,720,1280)
#             'blurred': blur, # 一张模糊图像(3,720,1280)
#             'dataname': dataname, # 子数据集名000、001(字符串)
#             'img_idx': img_idx # 在子数据集中图像的索引(整数)
#         }
#
#         # 新增：加载 ground truth 图像
#         if self.is_for_train:
#             # ground truth images
#             gt_paths = self.gt_paths.get(dataname) # gtlist['00000016.png', '00000019.png', '00000022.png']
#             # index对应gt图像的完整路径D:\ZQQ\111\test_data\gopro_test\gt\000\00000016.png
#             gt_path = os.path.join(self.root_gt, dataname, gt_paths[img_idx])
#             gt = cv_utils.read_cv2_img(gt_path, input_nc=1)
#             sample['gt'] = gt
#
#         return sample