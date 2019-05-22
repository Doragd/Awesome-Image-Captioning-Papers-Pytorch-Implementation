import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size #feature map size 14*14*2048

        resnet = torchvision.models.resnet101(pretrained=True)  # 预训练模型ResNet-101

        # 去掉avgpool和fc层
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules) #迭代传入,生成Sequential

        # 对于任意形状输入可以转换为指定的形状输出
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # finetune微调
        self.fine_tune()

    def forward(self, images):
        """
        前向计算
        :param images: images, shape: (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # 5个卷积layer, 2^5=32,图像缩小32倍
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # 更换维度 (batch_size, encoded_image_size, encoded_image_size, 2048) 
        return out

    def fine_tune(self, fine_tune=True):
        """
        是否参与梯度计算,即是否训练参数

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # 如果要微调，则调整block2到4的
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 论文中的D,图片的特征维度,相当于encoder的hidden_dim
        :param decoder_dim: decoder的hidden_dim
        :param attention_dim: 计算score时的attention_dim
        """
        super(Attention, self).__init__()
        # score = f_att(encoder_dim,decoder_dim)
        # 计算score时要将encoder_dim 和 decoder_dim转为attention_dim
        # 相当于 wx1, wx2
        # 然后 relu(wx1+wx2)
        # 再用full_att将relu(wx1+wx2)转为一个value,即score
        # 然后用softmax得到注意力权值分布
        self.encoder_att = nn.Linear(encoder_dim, attention_dim) 
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1) 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """

        :param encoder_out:
             shape (batch_size, num_pixels, encoder_dim)
             即 batch_size * L * D, num_pixels是hidden_state的数量
             相当于(batch_size, seq_len, hidden_dim)
        :param decoder_hidden: 
            前一个 decoder hidden output,  (batch_size, decoder_dim)
        :return: 
            attention weighted encoding(z_t,(batch_size, encoder_dim)), 
            alpha(权重, (batch_size, num_pixels))
        """
        # 分别对encoder_out和decoder_hidden加线性层 相当于 wx1和wx2
        # 然后 relu(wx1+wx2) 就完成了一个简单的MLP
        # 最后由于得到的是分数，还要把最后的维度转为1维，然后把这一维压缩
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: 用于计算score
        :param embed_dim: embedding size
        :param decoder_dim: decoder hidden_size
        :param vocab_size: 词汇表长度
        :param encoder_dim: encoder hidden_size
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        # decoding LSTMCell 输入是[y_t-1;zt;h_t-1;c_t-1]
        # 对应到API上: y_t-1的维度+zt维度, hidden_dim/cell_dim
        # 注意这里 y_t-1是embed_dim 
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  
        # 初始化h,c,实质就是线性层
        self.init_h = nn.Linear(encoder_dim, decoder_dim) 
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # beta,输出维度是encoder_dim, 因为要和z_t相乘从而修正z_t,而z_t的维度就是encoder_dim
        # 实质上还是线性层 + 后面的sigmoid 组成一个简单MLP
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 
        self.sigmoid = nn.Sigmoid()
        # 输出one-hot编码的向量
        self.fc = nn.Linear(decoder_dim, vocab_size) 
        self.init_weights()  # 使用均匀分布进行权重初始化，加速收敛

    def init_weights(self):

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        对embedding采用预训练的词向量权重矩阵
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        如果使用预训练的词向量，是否微调
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        初始化h0和c0
        """
        mean_encoder_out = encoder_out.mean(dim=1) # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: encoded images
            shape: (batch_size, enc_image_size, enc_image_size, encoder_dim)
            此时还未打平成dim=3的
        :param encoded_captions: 
            encoded captions, shape (batch_size, max_caption_length)
        :param caption_lengths: 
            caption lengths, shape (batch_size, 1)
        :return: 
            predictions 整个batch的每个time step输出的词的概率(one-hot编码), 
            encoded_captions  按照长度排序的captions
            decode lengths 不包括end字符的captions长度, 
            alphas 整个batch的每个time step产生的权重系数
            sort indices 排序时原来的下标
        """

        batch_size = encoder_out.size(0) #获得batch_size
        encoder_dim = encoder_out.size(-1) #获得encoder_image特征维度，对应论文中的D
        vocab_size = self.vocab_size #词汇表长度

        # 将encoder_out打平为论文对应的(batch_size, L, D),这里记为(batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) 
        num_pixels = encoder_out.size(1) # 对应论文中的L

        # 按句子长度由高到低排列(encoder_out和encoded_captions都要同步变化)
        # 这样一个batch中的句子都是从按照原长度长到短排列的
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding shape (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # 按照论文中的做法初始化h0和c0
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # decode_lengths是指用于输入的单词序列长度,对于最后时间步输入是<end>前一个单词,输出是<end>
        # 所以<end>实际不参与输入
        decode_lengths = (caption_lengths - 1).tolist()

        #  保存输出的概率和alpha值(即每个decoder_hidden的权值向量)
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # 对每个时间步
        for t in range(max(decode_lengths)):
            # 一个batch的captions的shape:(batch_size, max_seq_len)
            # 每个时间步，计算一列, 即所有seq的t列, 但是有些长度不够t,所以只计算l>t的
            # batch_size_t 就是 每列的有效单词数，其实就是这时候假定新的batch_size
            # 从而 encoder_out和h也要取相应的batch_size
            # t = 0时 输入是<start>的词嵌入表示
            batch_size_t = sum([l > t for l in decode_lengths])
            # attention_weighted_encoding 即 zt
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            #修正系数beta
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            #这里用的是teacher_forcing
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
