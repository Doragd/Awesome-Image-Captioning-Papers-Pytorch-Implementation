import torch


def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪防止梯度爆炸

    :param optimizer: 需要梯度裁剪的优化器
    :param grad_clip: 裁剪阈值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                # inplace操作，直接修改这个tensor，而不是返回新的
                # 将梯度限制在(-grad_clip, grad_clip)间
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    衰减学习率
    :param optimizer: 指定要衰减学习率的优化器
    :param shrink_factor: 学习率衰减系数,(0,1)之间
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    保存模型

    :param data_name: 处理后的数据集名称(前缀)
    :param epoch: epoch number
    :param epochs_since_improvement: 自上次提升BLEU4后经过的epoch数
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: encoder optimizer if fine-tuning
    :param decoder_optimizer: decoder optimizer
    :param bleu4: 每个epoch的验证集上的bleu4
    :param is_best: 该模型参数是否是目前最优的
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth'
    torch.save(state, filename)
    # 如果目前的checkpoint是最优的，添加备份以防被重写
    if is_best:
        torch.save(state, 'BEST_' + filename)


def accuracy(scores, targets, k):
    """
    计算top-k正确率

    :param scores: 预测的label(one-hot) shape: (所有没有pad的元素, vocab_size)
    :param targets: 正确的label shape: (所有没有pad的元素)
    :param k: 前k个
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True) # 寻找前k个概率最大的,返回下标 shape(所有没有pad的元素,k)
    # 扩展到(所有没有pad的元素, 1),再广播(复制)至(所有没有pad的元素, k)
    # eq操作是元素级的, 对应位置进行比较,返回值为元素级的0,1,所以最后返回shape(所有没有pad的元素, k)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    #  correct.view(-1): (所有没有pad的元素*k) 再求和得到正确的个数
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size) #求得平均正确率(每句)



class AverageMeter(object):
    """
    跟踪指标的最新值,平均值,和,count
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 #value
        self.avg = 0 #average
        self.sum = 0 #sum
        self.count = 0 #count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count