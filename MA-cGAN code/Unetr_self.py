import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from utils.attentionScrip import se_block


class SingleDeconv3DBlock(nn.Module):
    '''
    使用转置卷积来实现上采样
    '''
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    '''
    decoder的三维卷积模块
    conv3x3x3,BN,Relu
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    '''
    反卷积上采样模块
    deconv2x2x2,conv3x3x3,BN,Relu
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Embeddings(nn.Module):
    '''
    embedded patches

    '''
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        #计算有多少个patch
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        # patch的大小
        self.patch_size = patch_size
        # 嵌入的尺寸大小，默认768
        self.embed_dim = embed_dim
        #使用3D卷积计算patch embedding
        # 在NLP中语言序列是1D的序列使用朋友torch中的nn.Embedding()
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        # 设置一个可以学习的嵌入位置参数
        #将一个固定不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.position_embeddings变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        #dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #[1,4,128,128,128]->[1,768,8,8,8]
        x = self.patch_embeddings(x)
        #从dim=2开始展平->[1,768,512]
        x = x.flatten(2)
        x = x.transpose(-1, -2) #[1,512,768]
        # 直接加上位置信息
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    '''
    transformer结构的核心模块:自注意力模块
    学习Wq,Wk,Wv矩阵
    # 输入和输出是相同的的尺寸[B,Seq_dim,embded_dim]
    '''

    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # query,key,value 具体实现是一个线性层(全量就层) 输入维度是K/n,输出维度是K
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        # x.shape=[1,512,768]
        # reshape tensor 到需要的维度[B,embded_dim,heads,head_size] torch.Size([1, 512, 12, 64])
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Q 和 K 计算出 scores，然后将 scores 和 V 相乘，得到每个patch的context vector

        # 1.SA(z) = Softmax( qk> √Ch )v,计算出 scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # torch.Size([1, 12, 512, 512])
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        # 2.scores 和 V 相乘
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # torch.Size([1, 12, 512, 64])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # torch.Size([1, 512, 768])
        context_layer = context_layer.view(*new_context_layer_shape)
        # 最后的一个线性输出层
        attention_output = self.out(context_layer)
        # 加了一个dropout层
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class PositionwiseFeedForward(nn.Module):
    '''
    位置级前馈网络
    除了注意子层外,我们的编码器和解码器中的每个层都包含一个完全连接的前馈网络.
    它分别和相同地应用于每个位置。这由两个线性变换组成.中间有一个ReLU激活。
    FFN(x) = max(0, xW1 + b1)W2 + b2 (2)
    '''

    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # Residual Dropout
        self.dropout = nn.Dropout(dropout)


class Mlp(nn.Module):
    '''
    MLP 层
    采用高斯误差线性单元激活函数GELU
    zi = MLP(Norm(z0i)) + z0i,
    '''

    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    '''
    可重复的transformer block
    Norm->MSA->Norm->MLP
    '''
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        #归一化，在一个样本上做归一化操作这里是laerNorm 而不是BatchNorm
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        #mlp dim
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        #1.NORM
        x = self.attention_norm(x)
        #2.MSA
        x, weights = self.attn(x)
        # 残差链接
        x = x + h
        h = x
        #3.MLP
        x = self.mlp_norm(x)
        x = self.mlp(x)
        #残差链接
        x = x + h
        return x, weights


# class TransformerBlock(nn.Module):
#     '''
#     可重复的transformer block
#     Norm->MSA->Norm->MLP
#     '''
#     def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
#         super().__init__()
#         #归一化，在一个样本上做归一化操作这里是laerNorm 而不是BatchNorm
#         self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
#         #mlp dim
#         self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
#         self.mlp = PositionwiseFeedForward(embed_dim, 2048)
#         self.attn = SelfAttention(num_heads, embed_dim, dropout)
#
#     def forward(self, x):
#         h = x
#         #1.NORM
#         x = self.attention_norm(x)
#         #2.MSA
#         x, weights = self.attn(x)
#         # 残差链接
#         x = x + h
#         h = x
#         #3.MLP
#         x = self.mlp_norm(x)
#         x = self.mlp(x)
#         #残差链接
#         x = x + h
#         return x, weights


class Transformer(nn.Module):
    """
    tansformer as the encoder:

    Args:
        input_dim:=4(MRI数据,多channel)
            输入数据的channel
        embed_dim:=768
            embedding 的尺寸
        cube_size:
            体数据的尺寸
        patch_size:=16
            补丁的个数
        num_heads:=12
            有多少个Multi-Head
        num_layers:
            layer的数目对应num_heads

        dropout:0.1
            随机dropout的概率
        extract_layers:=[3,6,9,12]
            提取特征的层

    """

    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers

class  UNETR(nn.Module):
    def __init__(self, img_shape=(128, 128, 128), in_channels=1, out_channels=2, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                in_channels,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(in_channels, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                #Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, out_channels, 1)
            )

    def forward(self, x):
        z = self.transformer(x)#z=[4,1,512,768]
        z0, z3, z6, z9, z12 = x, *z
        #[1,512,768]->[1,768,8,8,8]
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output


if __name__ == '__main__':
    a = torch.randn([1, 1, 128, 128, 128]).cuda()
    # a=torch.randn([1,768,768]).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=UNETR().to(device)
    b=model(a)
    print(b.shape)