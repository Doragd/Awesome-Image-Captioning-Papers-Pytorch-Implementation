import time
import os
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, Decoder
from datasets import CaptionDataset
from utils import accuracy, adjust_learning_rate, AverageMeter, save_checkpoint, clip_gradient
from nltk.translate.bleu_score import corpus_bleu


data_folder = 'output_data'  # 已处理的数据所在文件夹
data_name = 'flickr8k'  # 数据文件名前缀

# 模型参数
emb_dim = 512  # embeddeding层维数
decoder_dim = 512  # decoder hidden层维数
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 # 仅在输入数据维度或类型上变化不大时使用，否则会产生大量计算开销 
 # 输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，降低效率
cudnn.benchmark = True 

# 训练参数
start_epoch = 0
epochs = 20  # epoch数目，除非early stopping
epochs_since_improvement = 0  # 跟踪训练时的验证集上的BLEU变化，每过一个epoch没提升则加1
batch_size = 80
workers = 0  # 多处理器加载数据,但一般取0
encoder_lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
decoder_lr = 4e-4 
grad_clip = 5.  # 梯度裁剪阈值
best_bleu4 = 0.  # 目前验证集的BLEU-4 score
print_freq = 100  # 每隔print_freq个iteration打印状态
fine_tune_encoder = False  # 是否微调
checkpoint = None  # 模型断点, 无则None


def main():
    """
    训练和验证
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # 读入词典
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # 初始化/加载模型
    if checkpoint is None:
        decoder = Decoder(
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       num_layers=num_layers)
        # 这里仅优化需要求梯度的参数，尤其是encoder,设置了这个才能训练指定层
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        # 是否微调
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        #载入checkpoint
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            # 如果此时要开始微调,需要定义优化器
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # 移动到GPU
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) #ImageNet
    # pin_memory = True 驻留内存，不换进换出
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # 如果连续8个epoch效果没提升则学习率衰减, 连续20个epoch没提升则中止训练
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            # 如果设置了fine tune还需要调整encoder
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # 一个epoch的训练
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # 一个epoch的验证
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # 检查是否有提升
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # 保存模型
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    执行一个epoch的训练

    :param train_loader: DataLoader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 交叉熵loss
    :param encoder_optimizer:  encoder optimizer(if fine-tuning)
    :param decoder_optimizer:  decoder optimizer
    :param epoch: 执行到第几个epoch
    """
    # 切换模式(使用dropout)
    decoder.train() 
    encoder.train()

    batch_time = AverageMeter()  # 训练一个batch的平均时间
    data_time = AverageMeter()  # 加载一个batch的平均时间
    losses = AverageMeter()  # 平均到每个单词的loss
    top5accs = AverageMeter()  # 平均到每个句子的正确率

    start = time.time()

    # 迭代每个batch
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # 移动到GPU
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # 前向计算
        imgs = encoder(imgs)
        # predictions, encoded_captions, decode_lengths, alphas, sort_ind
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # 输入的第一个字符是<start> 所以输出应该是<start>以后的, 因此用来比较的target也是从下标1开始
        targets = caps_sorted

        # 计算loss时可以用pack_padded_sequence将pad移除掉或者不需要输入的(<end>)
        # 超出decode_lengths的则会被移除掉
        # targets shape [所有没有pad的元素]
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # 计算loss
        loss = criterion(scores, targets)

        # 反向传播
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # 更新参数
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # 更新指标
        top5 = accuracy(scores, targets, 5)
        # loss.item()是平均到每个元素的loss sum(decode_lengths)是元素总数
        # losser.update是更新到目前为止的所有batch的平均每个元素loss
        losses.update(loss.item(), sum(decode_lengths))
        # top5accs也更新到目前为止所有batch的平均每句的正确率
        top5accs.update(top5, sum(decode_lengths))
        # 训练一个batch的平均时间
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    执行一个epoch的验证(跑完整个验证集)

    :param val_loader: 验证集的DataLoader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: 交叉熵loss
    :return: BLEU-4 score
    """
    # 切换到评估模式(无dropout)
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) 用于计算BLEU-4
    hypotheses = list()  # hypotheses (predictions)

    # 设置不计算梯度
    with torch.no_grad():
        # 迭代每个batch
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # 移动到GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 前向计算
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # 输入的第一个字符是<start> 所以输出应该是<start>以后, 因此用来比较的target也是从下标1开始
            targets = caps_sorted
            print(scores.shape)
            print(decode_lengths.shape)
            print(scores[decode_lengths].shape)
            # pack_padded_sequence 移除不需要的元素
            vocab_size = len(word_map)
            scores_copy = scores[decode_lengths].view(batch_size, -1, vocab_size)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # 计算loss
            loss = criterion(scores, targets)


            # 更新指标
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))


            # References shape: [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...]
            # allcaps shape: [batch_size, captions_per_image, max_seq_len]
            allcaps = allcaps[sort_ind]  # 按照排了序的情况排列
            for j in range(allcaps.shape[0]): # 针对每个图片
                img_caps = allcaps[j].tolist() # [ref1a,ref1b,]
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # 移除 <start> 和 pads
                references.append(img_captions)

            # Hypotheses(predictions)
            # 未进行pad_pack操作的scores shape (batch_size, max_seq_len, vocab_size)
            # preds返回下标 (batch_size, max_seq_len, 1)
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # 移除 pads
            preds = temp_preds
            hypotheses.extend(preds) # shape [hyp1,hyp2,hyp3...]

            assert len(references) == len(hypotheses)

        # 计算整个验证集上的BLEU4
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
