import math
import torch
import torchvision
import torch.utils.data
from torch import nn
from torch.nn import init
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    '''
    Encoder, Resnet101预训练模型最后一个卷积层
    Resnet标准输入尺寸224x224 最后一个卷积层2048x7x7
    '''
    def __init__(self, hidden_size, embed_size, encoded_image_size=7, dropout=0.5):
        super(Encoder, self).__init__()
        # 224/32 = 7
        self.enc_image_size = encoded_image_size 
        # 预训练模型ResNet-101
        resnet = torchvision.models.resnet101(pretrained=True)
        # 去掉avgpool和fc层
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # 匹配encoder和decoder的hidden_size
        # spatial_features即为encoder的hidden states
        self.spatial_features = nn.Linear(2048, hidden_size)
        # global_features维度转为embed_size
        # 在每个时间步参与decoder输入,论文中的xt=[wt;vg]
        self.global_features = nn.Linear(2048, embed_size)
        # 初始化the spatial and global features的权重
        self.avgpool = nn.AvgPool2d(7) # kernel size = 7 用于取均值操作
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        # finetune
        self.fine_tune()

    def init_weights(self):
        """
        Initialize the weights of the spatial and global features, since we are applying a transformation
        (copy code)
        """
        init.kaiming_uniform_(self.spatial_features.weight)
        init.kaiming_uniform_(self.global_features.weight)
        self.spatial_features.bias.data.fill_(0)
        self.global_features.bias.data.fill_(0)
    
    def forward(self, images):
        '''
        前向传播
        :param images: 
            shape:(batch_size, channel=3, h=224,w=224)
        :return: spatial_f, global_f, enc_image
        '''
        encoded_image = self.resnet(images) # (batch_size, 2048, 7, 7)


        # batch_size
        batch_size = encoded_image.shape[0]
        # encoder hidden_state数量
        num_pixels = encoded_image.shape[2] * encoded_image.shape[3]
        # encoded_image feature dim
        feature_dim = encoded_image.shape[1]

        # 生成global features并转换维度为embed_size
        global_f = self.avgpool(encoded_image) #(batch_size, 2048, 1, 1)
        global_f = global_f.view(global_f.size(0), -1) #(batch_size, 2048)
        global_f = F.relu(self.global_features(self.dropout(global_f)))

        # 生成spatial features并转换维度为hidden_size
        enc_image = encoded_image.permute(0, 2, 3, 1) # 转换axises(batch_size, 7, 7, 2048)
        enc_image = enc_image.view(batch_size,num_pixels,feature_dim) # (batch_size, num_pixels=49, feature_dim=2048)
        spatial_f = F.relu(self.spatial_features(self.dropout(enc_image))) # (batch_size, num_pixels, hidden_size)

        return spatial_f, global_f, enc_image
    
    def fine_tune(self, fine_tune=True):
        """
        是否进行微调，即encoder参数是否进行优化
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # 调整层数 1 layer only
        for c in list(self.resnet.children())[7:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
    

class AdaptiveAttention(nn.Module):
    '''
    AdaptiveAttention
    注意: num_pixels = attention_dim = 49
    '''
    def __init__(self, hidden_size, attention_dim=49, dropout=0.5):
        super(AdaptiveAttention, self).__init__()
        self.encoder_att = nn.Linear(hidden_size, attention_dim, bias=True)
        self.decoder_att = nn.Linear(hidden_size, attention_dim, bias=False)
        self.sentinel_att = nn.Linear(hidden_size, attention_dim, bias=False)
        self.attention_out = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.encoder_att.weight)
        init.xavier_uniform_(self.decoder_att.weight)
        init.xavier_uniform_(self.sentinel_att.weight)
        init.xavier_uniform_(self.attention_out.weight)

    def forward(self, spatial_f, decoder_hidden, st):

        # (batch_size, num_pixels, attention_dim)
        encoder_att_out = self.encoder_att(self.dropout(spatial_f))
        # (batch_size, attention_dim)
        decoder_hidden_out = self.decoder_att(self.dropout(decoder_hidden))
        # (batch_size, attention_dim)
        sentinel_att_out = self.sentinel_att(self.dropout(st))

        # 计算zt
        # 转置后的1向量 (batch_size, 1, attention_dim)
        ones_vec = torch.ones(decoder_hidden_out.shape[0], 1, decoder_hidden_out.shape[1]).to(device)
        # decoder_hidden_out * 1^T
        # 批量矩阵乘法 (batch_size, attention_dim, attention_dim)
        hidden_mul = torch.bmm(decoder_hidden_out.unsqueeze(2), ones_vec)
        # zt
        # (bacth_size, num_pixels)
        zt = self.attention_out(self.dropout(
            # (batch_size, num_pixels, attention_dim)
            torch.tanh(encoder_att_out + hidden_mul) 
        )).squeeze(2)

        # 计算注意力权重分布
        # (batch_size, num_pixels)
        alpha_t = F.softmax(zt, dim=1)

        # 计算ct: context vector,包含空间图像信息
        # spatial_f: (batch_size, num_pixels, hidden_size)
        # alpha_t: (batch_size, num_pixels)
        # ct: (batch_size, hidden_size)
        ct = (spatial_f * alpha_t.unsqueeze(2)).sum(dim=1)

        # 计算beta_t: 决定空间信息ct和语言模型信息st各自占的比重
        # 是alpha_hat最后一位 shape: (batch_size, 1)
        # 首先要计算alpha_hat: how much “attention” the network is placing on the sentinel
        temp_out = self.attention_out(self.dropout(
            torch.tanh(sentinel_att_out + decoder_hidden_out)
        ))
        score = torch.cat((zt, temp_out), dim=1)
        alpha_hat = F.softmax(score, dim=1)
        beta_t = alpha_hat[:, -1].unsqueeze(1)

        # c_hat: 包含空间图像信息的ct和语言信息的st
        # st: (batch_size, hidden_size)
        # ct: (batch_size, hidden_size)
        # c_hat: (batch_size, hidden_size)
        c_hat = beta_t * st + (1 - beta_t) * ct

        return alpha_t, beta_t, c_hat


class AdaptiveLSTMCell(nn.Module):
    '''
    改造后的LSTMCell
    主要加入了st和控制st的sentinel gate (sgate)
    '''
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.w_ih = nn.Parameter(torch.Tensor(5 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(5 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(5 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(5 * hidden_size))
        self.init_parameters()
    
    def init_parameters(self):
        '''
        copy code, and i don't know why
        '''
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.w_ih.data.uniform_(-stdv, stdv)
        self.w_hh.data.uniform_(-stdv, stdv)
        self.b_ih.data.fill_(0)
        self.b_hh.data.fill_(0)

    def forward(self, inp, states):
        ht,ct = states

        gates = F.linear(inp, self.w_ih, self.b_ih) + F.linear(ht, self.w_hh, self.b_hh)

        ingate, forgetgate, candidate_cellgate, outgate, sgate = gates.chunk(5, 1)

        ingate = torch.sigmoid(ingate)
        outgate = torch.sigmoid(outgate)
        forgetgate = torch.sigmoid(forgetgate)
        candidate_cellgate = torch.tanh(candidate_cellgate)
        sgate = torch.sigmoid(sgate)
        c_new = (forgetgate * ct) + (ingate * candidate_cellgate)
        h_new = outgate * torch.tanh(c_new)
        s_new = sgate * torch.tanh(c_new)

        return h_new, c_new, s_new

class DecoderWithAttention(nn.Module):
    '''
    Decoder, with Attention
    '''
    def __init__(self, hidden_size, vocab_size, attention_dim,embed_size, dropout=0.5, image_feature=2048):
        '''
        :param attention_dim: 用于计算score
        :param hidden_size: decoder hidden_size
        :param vocab_size: 词汇表长度
        :param embed_size: 词嵌入维度
        :param dropout: dropout  
        :param image_feature: 图片特征维度,用于初始化h0,c0    
        '''
        super(DecoderWithAttention, self).__init__()
        # 词汇表长度
        self.vocab_size = vocab_size
        # one-hot to word embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)   
        # 注意输入xt = [wt;v^g]是embed_size*2
        self.LSTM = AdaptiveLSTMCell(embed_size*2, hidden_size)
        # 生成y_t的全连接层
        self.fc = nn.Linear(hidden_size, vocab_size)
        # 注意力机制
        self.adaptive_attention = AdaptiveAttention(hidden_size, attention_dim)
        # 初始化h0,c0
        self.init_h = nn.Linear(image_feature, hidden_size)
        self.init_c = nn.Linear(image_feature, hidden_size)  
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        '''
        copy code
        '''
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)     

    def init_hidden_state(self, enc_image):
        '''
        see it in show and attend tell
        '''
        mean_enc_image = enc_image.mean(dim=1)
        h = self.init_h(mean_enc_image)
        c = self.init_c(mean_enc_image)
        return h,c

    def forward(self, spatial_f, global_f, enc_image, encoded_captions, caption_lengths):

        # batch_size
        batch_size = spatial_f.shape[0]
        # num_pixels(即attention_dim, 即encoder的hidden state数量)
        num_pixels = spatial_f.shape[1]
        # 词汇表长度
        vocab_size = self.vocab_size


        # 按句子长度由高到低排列
        # 这样一个batch中的句子都是从按照原长度长到短排列的
        # caption_lengths: (batch_size, 1)
        # sptial_f: (batch_size, num_pixels, hidden_size)
        # global_f: (batch_size, num_pixels, embed_size)
        # enc_image: (batch_size, num_pixels, image_feature=2048)
        # encoded_captions: (batch_size, max_cap_len)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        spatial_f = spatial_f[sort_ind]
        global_f = global_f[sort_ind]
        enc_image = enc_image[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # (batch_size, max_capt_len, embed_size)
        embeddings = self.embedding(encoded_captions)

        # 初始化h0和c0 (batch_size, hidden_size)
        h, c = self.init_hidden_state(enc_image)

        # decode_lengths是指用于输入的单词序列长度
        # 对于最后一个时间步: 输入是<end>前一个单词,输出是<end>
        # 所以<end>实际不参与输入
        decode_lengths = (caption_lengths - 1).tolist()        

        # 保存输出的概率和alphas,betas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        betas = torch.zeros(batch_size, max(decode_lengths), 1).to(device)

        # concat the embeddings and global image features
        # 作为LSTM的输入
        global_f = global_f.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((embeddings,global_f), dim = 2)

        #  对每个时间步
        for timestep in range(max(decode_lengths)):
            # 一个batch的captions的shape:(batch_size, max_cap_len)
            # 每个时间步，计算一列, 即所有seq的t列, 但是有些长度不够t,所以只计算l>t的
            # batch_size_t 就是 每列的有效单词数
            batch_size_t = sum([l > timestep for l in decode_lengths])

            # (batch_size_t, embed_size * 2)
            current_input = inputs[:batch_size_t, timestep, :]
            # (batch_size_t, hidden_size)
            h, c, st = self.LSTM(current_input, (h[:batch_size_t], c[:batch_size_t]))
            # adaptive attention model
            alpha_t, beta_t, c_hat = self.adaptive_attention(spatial_f[:batch_size_t], h, st)
            # 生成在词汇表上概率分布(batch_size, vocab_size)
            pt = self.fc(self.dropout(c_hat + h))

            # 保存结果
            predictions[:batch_size_t, timestep, :] = pt
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t

        return predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind         










