import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

'''
Thanks to: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
'''

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    根据karpathy_json来得到测试集，训练集和验证集
    :param dataset: 数据集名称, 'coco', 'flickr8k', 'flickr30k'其中之一
    :param karpathy_json_path: Karpathy JSON文件路径
    :param image_folder: 图片数据集路径
    :param captions_per_image: 每张图片采样的caption数目(每张图片共有5个描述)
    :param min_word_freq: 最小词频
    :param output_folder: 存放文件的路径
    :param max_len: 不采样超过该长度的caption
    """
    # 验证数据集名称是否合法
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # 读取json文件 data.keys: ['images','dataset']
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # 保存训练集，验证集，测试集 
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()
    
    for img in data['images']: # img.keys: [sentids', 'imgid', 'sentences', 'split', 'filename']
        captions = []
        for c in img['sentences']:
            # c.keys: ['tokens', 'raw', 'imgid', 'sentid'],tokens是分词后的序列,raw是字符串
            word_freq.update(c['tokens']) # 统计词频,结果为字典{‘word’:word_freq}
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0: #如果该图片的所有caption都大于max_len,则跳过
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # 安全性检查
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # 创建字典
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)} #v+1是因为0代表pad字符
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 输出文件名前缀
    base_filename = dataset

    # 保存字典为json文件
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # 从每张图片的多个描述中采样captions_per_image个
    # 保存图片为HDF5文件, 标题和其长度存储为json文件
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # 在文件的属性中记录每张采样的caption数量
            h.attrs['captions_per_image'] = captions_per_image

            # 在文件中创建数据集,shape为(images_datasets_size, channels, height, width)
            images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)
            
            enc_captions = [] #编码后的captions, 保存数据集中所有的caption
            caplens = [] #对应的长度

            for i, path in enumerate(tqdm(impaths, ascii=True)):

                # 采样captions_per_image个caption保存到captions中
                if len(imcaps[i]) < captions_per_image: 
                    #该张图片对应的caption数小于采样数，则重复采样直到达到采样数要求
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    #利用sample函数随机选取k个
                    captions = sample(imcaps[i], k=captions_per_image)

                # 合法性检验
                assert len(captions) == captions_per_image

                # 读图片
                img = imread(impaths[i])
                if len(img.shape) == 2: #如果是灰度图的话，就插入新维度，每个channel的图片都一样
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (224, 224)) #resize到(224,224)
                img = img.transpose(2, 0, 1) #shape: (3,224,224)
                #安全性检查
                assert img.shape == (3, 224, 224)
                assert np.max(img) <= 255

                # 将图片存入HDF5
                images[i] = img

                for j, c in enumerate(captions):
                    # 转为词向量序列，同时做了pad操作
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # 长度加2,不含pad
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # 安全性检查
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # 保存captions及长度到json文件
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

if __name__ == "__main__":
    # 数据预处理
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='caption_datasets/dataset_flickr8k.json',
                       image_folder='Flicker8k_Dataset/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='output_data/',
                       max_len=50)    