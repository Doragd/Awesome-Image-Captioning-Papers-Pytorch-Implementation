import torch
from torch.utils.data import Dataset
import h5py
import json
import os

'''
Thanks to: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
'''

class CaptionDataset(Dataset):
    """
    创建dataloader
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: 数据文件路径
        :param data_name: 数据文件前缀
        :param split: 类型, 'TRAIN', 'VAL', or 'TEST'
        :param transform: transform pipline 如果要自定义复杂的转换方法的话
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # 图片文件 两个字段：images和 captions_per_image
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # 每个图片采样的caption数量
        self.cpi = self.h.attrs['captions_per_image']

        # 将captions完全加载到内存
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # 将captions lens完全加载到内存
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # 可以使用Pytorch默认的transform 比如normalizing
        self.transform = transform

        # 数据集大小
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # 返回的一个样本的信息，在dataloader里面可以组装为一个batch
        # 这里换算关系: N号caption 对应的是 N // caption_per_image 号 图片
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.) #将图片归一化[0,1]，ToTensor可以实现
        if self.transform is not None: #应用transform,比如normalize到[-1,1]
            img = self.transform(img) #img shape: [3,224,224]

        caption = torch.LongTensor(self.captions[i]) #caption shape [max_len]

        caplen = torch.LongTensor([self.caplens[i]]) # shape [1]

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # 对于验证/测试，还需要返回该图片全部的captions用于BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
