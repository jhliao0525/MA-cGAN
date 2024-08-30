import torch

import Unter
from Unter import *
from timm import create_model as creat
# from vit_pytorch import ViT
from monai.networks.nets import ViT,SwinUNETR
# from utils.attentionScrip import *
# from unet3d_ori import UNet3D
#from models.UNet import UNet
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # self.conT1 = nn.Conv3d(1, 32, kernel_size=7,padding=3)
        # self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout=0.1))
        # self.fc=nn.Linear(589824,1)
        # self.vit = creat('vit_base_patch16_224', pretrained=True, num_classes=1)
        self.vit = ViT(
            in_channels=2,
            img_size=(48, 256, 256),
            patch_size=(16,16,16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=1,
            num_heads=12,
            pos_embed="perceptron",
            classification=None,
            dropout_rate=0.0,
            num_classes=1,
            post_activation=None
        )

        self.t1= UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=768,
            out_channels=16,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=False,
            res_block=False,
        )
        self.t2= UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=16,
            out_channels=1,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name="instance",
            conv_block=False,
            res_block=False,
        )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x



    def forward(self, x):

        x,h=self.vit(x)
        x = self.proj_feat(x, 768,(3,16,16) )
        x=self.t1(x)
        x = self.t2(x)
        return x


a=torch.randn([1,1,128,128,128]).cuda()
# a=torch.randn([1,768,768]).cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# model=UNet().to(device)
# model=Unter.UNETR(in_channels=2,out_channels=1,img_size=(48,256,256),feature_size=8,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed='perceptron',
#     norm_name='instance',
#     conv_block=True,
#     res_block=True,
#     dropout_rate=0.0).to(device)

model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=1,
        out_channels=1,
        feature_size=12,

    ).to(device)

# trans1=TransformerBlock(hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate = 0.0, qkv_bias = False).to(device)
# patch_embedding = PatchEmbeddingBlock(
#     in_channels=1,
#
#     img_size=(48,256,256),
#     patch_size=8,
#     hidden_size=768,
#     num_heads=12,
#     pos_embed='perceptron',
#     dropout_rate=0.0,
#
# ).to(device)
# out=trans1(patch_embedding(a))
out=model(a)
print(out.shape)
