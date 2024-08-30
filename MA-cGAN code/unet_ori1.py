import torch
import torch.nn as nn
from torch.nn import functional as F
from monai.networks.nets import ViT



def proj_feat(x, hidden_size, feat_size):
    x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(DoubleConv, self).__init__()
        channels = out_channels / 2
        if in_channels > out_channels:
            channels = in_channels / 2

        layers = [
            # in_channels：输入通道数
            # channels：输出通道数
            # kernel_size：卷积核大小
            # stride：步长
            # padding：边缘填充
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
        ]
        if bath_normal: # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            # 采用双线性插值的方法进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 采用反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + in_channels / 2, out_channels, batch_normal)

    # inputs1：上采样的数据（对应图中黄色箭头传来的数据）
    # inputs2：特征融合的数据（对应图中绿色箭头传来的数据）
    def forward(self, inputs1, inputs2):
        # 进行一次up操作
        inputs1 = self.up(inputs1)

        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class vit1(nn.Module):#1,1,48,256,256
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    def __init__(self):
        super(vit1, self).__init__()
        self.vit = ViT(
            in_channels=128, img_size=(48,256,256), patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=1,
            pos_embed='perceptron',
            classification=False,

            dropout_rate=0.0
        )
        self.conT1=nn.ConvTranspose3d(768, 256, 4, 4)
        self.conT2 = nn.ConvTranspose3d(256, 128, 2, 2)
        self.conT3 = nn.ConvTranspose3d(128, 64, 2, 2)



    def forward(self, x):
        x=self.vit(x)[0]
        x=self.proj_feat(x, 768, (3,16,16))#(1,768,3,16,16)
        x=self.conT1(x)
        x = self.conT2(x)
        x = self.conT3(x)
        return x

class vit2(nn.Module):#1,1,48,256,256
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    def __init__(self):
        super(vit2, self).__init__()
        self.vit = ViT(
            in_channels=128, img_size=(24,128,128), patch_size=(8, 8, 8),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=1,
            pos_embed='perceptron',
            classification=False,

            dropout_rate=0.0
        )
        self.conT1=nn.ConvTranspose3d(768, 256, 4, 4)
        self.conT2 = nn.ConvTranspose3d(256, 128, 2, 2)
        # self.conT3 = nn.ConvTranspose3d(64, 32, 2, 2)



    def forward(self, x):
        x=self.vit(x)[0]
        x=self.proj_feat(x, 768, (3,16,16))#(1,768,3,16,16)
        x=self.conT1(x)
        x = self.conT2(x)
        # x = self.conT3(x)
        return x
class vit3(nn.Module):#1,256,12,64,64
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    def __init__(self):
        super(vit3, self).__init__()
        self.vit = ViT(
            in_channels=128, img_size=(12,64,64), patch_size=(4, 4, 4),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=1,
            pos_embed='perceptron',
            classification=False,

            dropout_rate=0.0
        )
        self.conT1=nn.ConvTranspose3d(768, 256, 2, 2)
        self.conT2 = nn.ConvTranspose3d(256, 128, 2, 2)
        # self.conT3 = nn.ConvTranspose3d(64, 32, 2, 2)



    def forward(self, x):
        x=self.vit(x)[0]
        x=self.proj_feat(x, 768, (3,16,16))#(1,768,3,16,16)
        x=self.conT1(x)
        x = self.conT2(x)
        # x = self.conT3(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear

        self.inputs = DoubleConv(in_channels, 64, self.batch_normal)
        self.down_1 = DownSampling(64, 128, self.batch_normal)
        self.down_2 = DownSampling(128, 256, self.batch_normal)
        self.down_3 = DownSampling(256, 512, self.batch_normal)

        self.up_1 = UpSampling(512, 256, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.batch_normal, self.bilinear)
        self.outputs = LastConv(64, num_classes)

        # self.vit1 = ViT(
        #     in_channels=128, img_size=(48, 256, 256), patch_size=(16, 16, 16),
        #     hidden_size=512,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     num_layers=4,
        #     pos_embed='perceptron',
        #     classification=False,
        #
        #     dropout_rate=0.0
        # )
        # self.vit1_1=nn.Sequential(
        #     nn.ConvTranspose3d(768, 256, 2, 2),
        #     nn.ConvTranspose3d(256, 128, 1, 1)
        # )




    def forward(self, x):


        # down 部分
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x1=vit1(x1)
        x2=vit2(x2)
        x3=vit3(x3)


        # up部分
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x = self.outputs(x7)

        return x
