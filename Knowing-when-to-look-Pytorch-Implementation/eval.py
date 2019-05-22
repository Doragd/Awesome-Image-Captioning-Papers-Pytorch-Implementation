import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import CaptionDataset
from utils import clip_gradient, adjust_learning_rate, save_checkpoint, accuracy, AverageMeter
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm


data_folder = 'output_data/'  # 预处理数据存放文件夹
data_name = 'flickr8k'  # 数据文件名前缀
checkpoint = 'BEST_checkpoint_flickr8k.pth'  # 模型断点
word_map_file = 'output_data/WORDMAP_flickr8k.json'  # 词典

# 模型参数
beam_size = 3
batch_size = 32
num_workers = 0 # 如果是Windows系统，则0，否则为1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


# 加载模型
checkpoint = torch.load(checkpoint, map_location='cpu')
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()


# 加载词典
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# 归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def beam_search_eval(encoder, decoder, image, beam_size, wrong):
    '''
    beam_search for batch_size=1 
    :param: wrong: 在生成过程中一直没遇到<end>且句子长度超过50时,wrong+=1
    :param: image: (1, 3, 224, 224)
    '''

    k = beam_size
    vocab_size = len(word_map)
    smth_wrong = False # 记录是否出错

    # encode
    image = image.to(device).unsqueeze(0) # (1, 3, 224, 224)
    spatial_f, global_f, enc_image = encoder(image)

    # enc_image: (1, 49, 2048) ==> (k, 49, 2048)
    num_pixels = enc_image.size(1)
    image_feature = enc_image.size(2)
    enc_image = enc_image.expand(k, num_pixels, image_feature)

    # k个输入,初始值为<start>的编码 (k, 1)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    # 保存句子 (k, 1)
    seqs = k_prev_words
    # 保存分数 (k, 1)
    top_k_scores = torch.zeros(k, 1).to(device)
    # 保存完整的句子和分数
    complete_seqs = list() # k个句子
    complete_seqs_scores = list() # k个分数
    # 开始decode
    step = 1
    # 初始化 h/c: (k, hidden_size)
    h, c = decoder.init_hidden_state(enc_image)
    while True:
        # 词嵌入 (k)
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        # xt: (k, embed_dim*2)
        inputs = torch.cat((embeddings,global_f.expand_as(embeddings)), dim = 1)
        # 生成h,c,st (k, hidden_size)
        h, c, st = decoder.LSTM(inputs, (h, c))
        # adaptive attention
        alpha_t, beta_t, c_hat = decoder.adaptive_attention(spatial_f,h,st)
        scores = decoder.fc(c_hat + h)
        # scores: (k, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        # 与之前生成的相加 logp1+logp2这样
        # (k, 1)广播为(k, vocab_size)+(k, vocab_size)==>(k, vocab_size)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        
        prev_word_inds = top_k_words / vocab_size
        next_word_inds = top_k_words % vocab_size
        # (k, step+1)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        enc_image = enc_image[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            # 超过50都没有遇到<end>
            smth_wrong = True
            break
        step += 1
    
    if smth_wrong is not True:
        # 取k个句子中概率最大的那个输出
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        sentence = [rev_word_map[seq[i]] for i in range(len(seq))]
    else:
        # 否则取长度为20，强制加end，生成1个句子, wrong+1
        seq = seqs[0][:20]
        sentence = [rev_word_map[seq[i].item()] for i in range(len(seq))]
        sentence = sentence + ['<end>']
        wrong += 1
    
    return sentence,wrong

def evaluate_batch(encoder, decoder, imgs, allcaps):
    '''
    批量进行evaluate
    imgs: (batch_size, 3, 224, 224)
    allcaps: (batch_size, captions_per_image=5, max_cap_len)
    '''
    wrong = 0
    batch_predictions = []
    for i in range(imgs.shape[0]):
        '''
        实际还是循环一个batch中的每个样本
        '''
        sent,wrong = beam_search_eval(encoder, decoder, imgs[i], beam_size, wrong)
        batch_predictions.append(sent)
    
    batch_references = []
    for batch in allcaps:
        captions = []
        for i in range(allcaps.shape[1]):
            current = batch[i]
            current_sen = [rev_word_map[current[j].item()] for j in range(current.shape[0])]
            # 去除<pad>
            filtered_sen = [word for word in current_sen if word != '<pad>']
            captions.append(filtered_sen)
        batch_references.append(captions)
    
    # batch_predictions: batch_size个句子
    # batch_references: batch_size*5个句子
    return batch_predictions, batch_references, wrong

def evaluate():

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    wrong_tot = 0
    references = list()
    hypotheses = list()

    for i, (imgs, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size), ascii=True)):

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        batch_predictions, batch_references, wrong = evaluate_batch(encoder, decoder, imgs, allcaps)
        wrong_tot += wrong
        assert len(batch_references) == len(batch_predictions)
        references.extend(batch_references)
        hypotheses.extend(batch_predictions)
    
    bleu4 = corpus_bleu(references, hypotheses)
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu4))


if __name__ == "__main__":
    evaluate()
    



    


