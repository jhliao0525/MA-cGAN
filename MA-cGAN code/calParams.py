# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from ptflops import get_model_complexity_info

from models.unet3d_ori import UNet3D
import Unter
from models.transunet3d import transUNet3D
from models.trans_bts_ori import transbts
from monai.networks.nets import ViT,SwinUNETR
from models.resUnet_apaatt import resUnet_apa
from models.utnetv2.dim3.utnetv2 import UTNetV2
from models.Vnet import VNet
from pix2pixGan import Discriminator_gan
# Model
print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
# model = Unter.UNETR(in_channels=1, out_channels=2, img_size=(128, 128, 128), feature_size=16,
#                     hidden_size=768,
#                     mlp_dim=3072,
#                     num_heads=12,
#                     pos_embed='perceptron',
#                     norm_name='instance',
#                     conv_block=True,
#                     res_block=False,
#                     dropout_rate=0.2)

# model=UNet3D(in_channels=1, num_classes=2, batch_normal=True, bilinear=True)
# model =transUNet3D(in_channels=1,num_classes=2)
# model = transbts(in_channel=1, out_channel=2, training=True)

# model = SwinUNETR(
#         img_size=(128, 128, 128),
#         in_channels=1,
#         out_channels=2,
#         feature_size=12,
#
#     )
# model=resUnet_apa(in_channel=1, out_channel=2, training=True)
model = Discriminator_gan()
# model=VNet(n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=False)

# model=UTNetV2(in_chan=1, num_classes=2)

dummy_input = torch.randn(1,2, 128, 128, 128)
musk=torch.randn(1,1, 128, 128, 128)
flops, params = profile(model, (dummy_input,musk))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

flops, params = get_model_complexity_info(model, (1, 128, 128, 128), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)


