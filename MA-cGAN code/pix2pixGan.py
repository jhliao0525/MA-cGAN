import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model as creat
# from monai.networks.blocks.transformerblock import TransformerBlock
# from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
#
from monai.networks.nets import ViT
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
# from utils.attentionScrip import scSE
# from torch.utils import data
# import torchvision
# from torchvision import transforms
#
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# from PIL import Image

#创建模型
class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Downsample, self).__init__()
        self.conv_relu=nn.Sequential(
                        nn.Conv3d(in_channels,out_channels,
                                  kernel_size=3,stride=2,padding=1),
                        nn.LeakyReLU(inplace=True)#inplace=True,将计算得到的值直接覆盖之前的值
        )
        self.bn=nn.BatchNorm3d(out_channels)

    def forward(self,x,is_bn=True):
        x=self.conv_relu(x)
        if is_bn:
            x=self.bn(x)
        return x

#上采样
class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu=nn.Sequential(
                            nn.ConvTranspose3d(in_channels,out_channels,
                                               kernel_size=3,stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.LeakyReLU(inplace=True)
        )
        self.bn=nn.BatchNorm3d(out_channels)

    def forward(self,x,is_drop=False):
        x=self.upconv_relu(x)
        x=self.bn(x)
        if is_drop:
            x=F.dropout3d(x)
        return x

#定义判别器
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.down1=Downsample(6,64)
#         self.down2=Downsample(64,128)
#         self.down3=Downsample(128,256)
#         self.conv=nn.Conv3d(256,512,3,1,1)
#         self.bn=nn.BatchNorm3d(512)
#         self.last=nn.Conv3d(512,1,3,1)
#     def forward(self,img,musk):
#         x=torch.cat([img,musk],axis=1)
#         x=self.down1(x,is_bn=False)
#         x=self.down2(x)
#         x=F.dropout2d(self.down3(x))
#         x=F.dropout2d(F.leaky_relu(self.conv(x)))
#         x=F.dropout2d(self.bn(x))
#         x=torch.sigmoid(self.last(x))
#         return x

#输入生成器的图片是（[1, 1, 48, 256, 256]）
#由生成器（unet）传入判别器的  （1,2,48,256,256）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1=Downsample(3,64)
        self.down2=Downsample(64,128)
        self.down3=Downsample(128,256)
        self.conv=nn.Conv3d(256,512,3,1,1)
        self.bn=nn.BatchNorm3d(512)
        self.last=nn.Conv3d(512,1,3,1)
    def forward(self,img,musk):
        x=torch.cat([img,musk],axis=1)
        x=self.down1(x,is_bn=False)
        x=self.down2(x)
        x=F.dropout3d(self.down3(x))
        x=F.dropout3d(F.leaky_relu(self.conv(x)))
        x=F.dropout3d(self.bn(x))
        # x = self.down3(x)
        # x = F.leaky_relu(self.conv(x))
        # x = self.bn(x)
        x=torch.sigmoid(self.last(x))
        return x

#由生成器（unet）传入判别器的  （1,2,48,256,256）
class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.down1=Downsample(3,64)
        self.down2=Downsample(64,128)
        self.bn = nn.BatchNorm3d(128)
        #self.down3=Downsample(128,256)
        self.conv=nn.Conv3d(128,1,3,1,1)

        #self.last=nn.Conv3d(512,1,3,1)
    def forward(self,img,musk):
        x=torch.cat([img,musk],axis=1)
        x=self.down1(x,is_bn=False)

        x=F.dropout3d(self.down2(x))
        #x=F.dropout3d(F.leaky_relu(self.conv(x)))
        x=F.dropout3d(self.bn(x))
        x=torch.sigmoid(self.conv(x))
        return x
# 定义论文中的判别器
class Discriminator_paper(nn.Module):
    def __init__(self):
        super(Discriminator_paper, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3,stride=2,padding=1)
        #self.scse=scSE(64)
        self.conv2 = nn.Conv3d(64, 128,kernel_size=3,stride=2,padding=1)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=128)

        self.conv3 = nn.Conv3d(128, 1, kernel_size=1,stride=1)


    def forward(self, img, musk):
        # print("img{}".format(img.shape))
        # print("musk{}".format(musk.shape))
        x = torch.cat([img, musk], axis=1)
        x = self.conv1(x)
       # x=self.scse(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout3d(x)
        x = self.conv2(x)
        x=self.gn(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x=F.dropout3d(x)
        x=torch.sigmoid(self.conv3(x))
        return x
# 使用WGann中的判别器
class Discriminator_gan(nn.Module):
    def __init__(self):
        super(Discriminator_gan, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3,stride=2,padding=1)
        #self.scse=scSE(64)
        self.conv2 = nn.Conv3d(64, 128,kernel_size=3,stride=2,padding=1)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=128)

        self.conv3 = nn.Conv3d(128, 1, kernel_size=1,stride=1)
        #self.fc = nn.Linear(128 *48* 256 * 256, 1)

    def forward(self, img, musk):
        # print("img{}".format(img.shape))
        # print("musk{}".format(musk.shape))
        x = torch.cat([img, musk], axis=1)
        x = self.conv1(x)
       # x=self.scse(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = F.dropout3d(x)
        x = self.conv2(x)
        x=self.gn(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x=F.dropout3d(x)
        #x=self.fc(x)
        x=torch.sigmoid(self.conv3(x))
        return x

# 使用trans中的判别器
class Discriminator_trans(nn.Module):
    def __init__(self):
        super(Discriminator_trans, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3,stride=2,padding=1)
        #self.scse=scSE(64)
        self.conv2 = nn.Conv3d(64, 128,kernel_size=3,stride=2,padding=1)
        # self.gn = nn.GroupNorm(num_groups=2, num_channels=128)

        # self.conv3 = nn.Conv3d(128, 1, kernel_size=1,stride=1)

        self.vit = ViT(
            in_channels=128,
            img_size=(12,64,64),
            patch_size=4,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=1,
            num_heads=12,
            pos_embed="perceptron",
            classification="classification_head",
            dropout_rate=0.0,
            num_classes=1,
            post_activation=None
        )
        # self.vit = creat('vit_base_patch16_224', pretrained=True, num_classes=1)

    #     self.trans=TransformerBlock(hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate=0.0, qkv_bias=False)
    #     self.patch_embedding = PatchEmbeddingBlock(
    #     in_channels=128,
    #
    #     img_size=(12, 64, 64),
    #     patch_size=6,
    #     hidden_size=768,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     dropout_rate=0.0,
    #
    # )
    #     self.fc = nn.Linear(1 * 768 * 768, 1)


    def forward(self, img, musk):
        # print("img{}".format(img.shape))
        # print("musk{}".format(musk.shape))
        x = torch.cat([img, musk], axis=1)
        x = self.conv1(x)
       # x=self.scse(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        # x = F.dropout3d(x)
        x = self.conv2(x)
        # x=self.gn(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        # x=F.dropout3d(x)
        #x=self.fc(x)
        # x=self.conv3(x)
        # x=self.trans(self.patch_embedding(x))
        # x = x.view(-1, 1 * 768 * 768)  # (batch, 128, 6, 6）-->  (batch, 128*6*6)
        # x = self.fc(x)
        x,h=self.vit(x)
        x = torch.sigmoid(x)

        return x

# 使用trans中的判别器
###128，128，128
class Discriminator_trans128(nn.Module):
    def __init__(self):
        super(Discriminator_trans128, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3,stride=2,padding=1)
        # #self.scse=scSE(64)
        self.conv2 = nn.Conv3d(64, 128,kernel_size=3,stride=2,padding=1)
        # self.conv=nn.Sequential(
        #     nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.GroupNorm(num_groups=2, num_channels=64),
        #     nn.ReLU(True),
        #     nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.GroupNorm(num_groups=2, num_channels=128),
        #     nn.ReLU(True)
        # )
        # self.gn = nn.GroupNorm(num_groups=2, num_channels=128)

        # self.conv3 = nn.Conv3d(128, 1, kernel_size=1,stride=1)

        self.vit = ViT(
            in_channels=128,
            img_size=(32,32,32),
            patch_size=4,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=1,
            num_heads=12,
            pos_embed="perceptron",
            classification="classification_head",
            dropout_rate=0.0,
            num_classes=1,
            post_activation=None
        )
        # self.vit = creat('vit_base_patch16_224', pretrained=True, num_classes=1)

    #     self.trans=TransformerBlock(hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate=0.0, qkv_bias=False)
    #     self.patch_embedding = PatchEmbeddingBlock(
    #     in_channels=128,
    #
    #     img_size=(12, 64, 64),
    #     patch_size=6,
    #     hidden_size=768,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     dropout_rate=0.0,
    #
    # )
    #     self.fc = nn.Linear(1 * 768 * 768, 1)


    def forward(self, img, musk):
        # print("img{}".format(img.shape))
        # print("musk{}".format(musk.shape))
        x = torch.cat([img, musk], axis=1)
        x = self.conv1(x)
        # x=self.scse(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        # x = F.dropout3d(x)
        x = self.conv2(x)
        # x=self.gn(x)
        x = F.leaky_relu(x, negative_slope=0.2)
       #  x = self.conv1(x)
       # # x=self.scse(x)
       #  x = F.relu(x)
       #  # x = F.dropout3d(x)
       #  x = self.conv2(x)
       #  # x=self.gn(x)
       #  x = F.relu(x)
        # x=F.dropout3d(x)
        #x=self.fc(x)
        # x=self.conv3(x)
        # x=self.trans(self.patch_embedding(x))
        # x = x.view(-1, 1 * 768 * 768)  # (batch, 128, 6, 6）-->  (batch, 128*6*6)
        # x = self.fc(x)
        # x=self.conv(x)
        x,h=self.vit(x)
        x = torch.sigmoid(x)

        return x

# 使用trans中的判别器，先vit 再卷积
class Discriminator_trans2(nn.Module):
    def __init__(self):
        super(Discriminator_trans2, self).__init__()
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=3,stride=2,padding=1)
        # #self.scse=scSE(64)
        # self.conv2 = nn.Conv3d(64, 128,kernel_size=3,stride=2,padding=1)
        # self.gn = nn.GroupNorm(num_groups=2, num_channels=128)

        # self.conv3 = nn.Conv3d(128, 1, kernel_size=1,stride=1)

        self.vit = ViT(
            in_channels=3,
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=4,
            num_heads=12,
            pos_embed="perceptron",
            classification=None,
            dropout_rate=0.0,
            num_classes=1,
            post_activation=None
        )
        # self.conv=nn.Sequential(
        #     nn.ConvTranspose3d(768, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
        #     nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
        #           padding=((kernel_size - 1) // 2))

        # )
        self.t1 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=768,
            out_channels=16,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=False,
            res_block=True,
        )
        self.t2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=16,
            out_channels=1,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=False,
            res_block=True,
        )
        self.map1 = nn.Sequential(
            nn.Conv3d(768, 16, 1, 1),
            ##测试时候改回上一行
            # nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=False),
            # 训练时采用0.5倍
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            # nn.Softmax(dim=1)
        )
        self.map2 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            ##测试时候改回上一行
            # nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=False),
            # 训练时采用0.5倍
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            # nn.Softmax(dim=1)
        )
        self.gn = nn.GroupNorm(num_groups=2, num_channels=16)


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
        # self.vit = creat('vit_base_patch16_224', pretrained=True, num_classes=1)

    #     self.trans=TransformerBlock(hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate=0.0, qkv_bias=False)
    #     self.patch_embedding = PatchEmbeddingBlock(
    #     in_channels=128,
    #
    #     img_size=(12, 64, 64),
    #     patch_size=6,
    #     hidden_size=768,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     dropout_rate=0.0,
    #
    # )
    #     self.fc = nn.Linear(1 * 768 * 768, 1)



    def forward(self, img, musk):
        # print("img{}".format(img.shape))
        # print("musk{}".format(musk.shape))
        x = torch.cat([img, musk], axis=1)
        x, h = self.vit(x)
        x = self.proj_feat(x, 768, (8, 8, 8))
        # x = self.t1(x)
        # x = self.t2(x)
        x=self.map1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x=self.gn(x)
        x=self.map2(x)

        x = torch.sigmoid(x)

        return x


# input_img=torch.randn([1,2,48,256,256]).cuda()
# mask=torch.randn([1,2,48,256,256]).cuda()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # model=UNet3D(in_channels=1,num_classes=2).to(device)
# model=Discriminator().to(device)
# out=model(input_img,mask)
# print(out.shape)