import torch
from torch import nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # 预训练模型ResNet-101
        self.resnet = torchvision.models.resnet101(pretrained=True)
        # 去掉fc层, 将最后一层改成恒等映射
        del self.resnet.fc
        self.resnet.fc = lambda x:x

        # finetune微调
        self.fine_tune()

    def forward(self, images):
        """
        前向计算
        :param images: images, shape: (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048) 
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


class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size,num_layers):
        super(Decoder, self).__init__()
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # 将图片特征向量视作序列第一个词
        self.cnn_x = nn.Linear(2048, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, decoder_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(decoder_dim, vocab_size)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        '''
        :param: caption_lengths: (batch_size, 1) ==> (batch_size)
        :param: encoder_out: (batch_size, 2048)
        :param: encoded_captions: (batch_size, max_cap_len)
        '''
        # 按句子长度由高到低排列(encoder_out和encoded_captions都要同步变化) 
        # 这样一个batch中的句子都是从按照原长度长到短排列的
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # (batch_size, max_seq_len, embed_dim)
        embeddings = self.embedding(encoded_captions)
        # (batch_size, 1, embed_dim)
        encoder_out = self.cnn_x(encoder_out).unsqueeze(1)
        # (batch_size, max_cap_len+1, embed_dim)
        embeddings = torch.cat([encoder_out, embeddings], dim=1)
        # packed
        # 假设句子长度为N,实际输入是1到N-1,输出是2到N
        # 将图像特征向量看作第0个词后, 实际输入0到N-1,输出为1到N
        # 故输入序列长度为(caption_lengths-1)+1 = caption_lengths
        packed = pack_padded_sequence(embeddings, caption_lengths, batch_first=True)
        # (batch_size, max_cap_len, decoder_dim),(h_n, c_n)
        # hn: (batch_size, num_layers, decoder_dim)
        outputs, _ = self.LSTM(packed)
        # outputs是pack对象,取其data
        pred = self.out(outputs.data)
        return pred, encoded_captions, caption_lengths, sort_ind