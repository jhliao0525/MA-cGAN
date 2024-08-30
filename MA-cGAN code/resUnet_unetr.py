import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
# from utils.attentionScrip import se_block

#################采用transformer做编码层，返回3，6，9，12层


def proj_feat(x, hidden_size, feat_size):
    x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x
class resUnet_unetr(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(resUnet_unetr, self).__init__()
        self.training = training
        ###############################################################
        ##################  双重卷积的编码层和解码层  ###############################
        # self.encoder1 = nn.Sequential(
        #             nn.Conv3d(in_channel, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
        #
        #            # nn.Conv3d(32, 32, 3, stride=1, padding=1),
        #
        #             nn.GroupNorm(num_groups=2, num_channels=32)
        # )
        #
        # self.encoder2=   nn.Sequential(
        #             nn.Conv3d(32, 64, 3, stride=1, padding=1), # b, 8, 3, 3
        #
        #
        #          #   nn.Conv3d(64, 64, 3, stride=1, padding=1),  # b, 8, 3, 3
        #
        #             nn.GroupNorm(num_groups=2, num_channels=64)
        # )
        #
        # self.encoder3=  nn.Sequential(
        #             nn.Conv3d(64, 128, 3, stride=1, padding=1),
        #
        #
        #          #   nn.Conv3d(128, 128, 3, stride=1, padding=1),
        #
        #              nn.GroupNorm(num_groups=2, num_channels=128)
        #
        # )
        # self.encoder4=   nn.Sequential(
        #             nn.Conv3d(128, 256, 3, stride=1, padding=1),
        #
        #             nn.GroupNorm(num_groups=2, num_channels=256)
        #
        # )
        self.decoder2 =nn.Sequential(
            nn.Conv3d(256, 128, 3, stride=1, padding=1),  # b, 8, 15, 1


           # nn.Conv3d(128, 128, 3, stride=1, padding=1),  # b, 8, 15, 1
            nn.GroupNorm(num_groups=2, num_channels=128)

        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128, 64, 3, stride=1, padding=1),  # b, 8, 15, 1


            #nn.Conv3d(64, 64, 3, stride=1, padding=1),  # b, 8, 15, 1
            nn.GroupNorm(num_groups=2, num_channels=64)

        )
        self.decoder4 = nn.Sequential(
            nn.Conv3d(64, 32, 3, stride=1, padding=1),  # b, 8, 15, 1


            # nn.Conv3d(32, 32, 3, stride=1, padding=1),  # b, 8, 15, 1
            nn.GroupNorm(num_groups=2, num_channels=32)

        )
        self.decoder5 = nn.Sequential(
            nn.Conv3d(32, 16, 3, stride=1, padding=1),  # b, 8, 15, 1
            nn.GroupNorm(num_groups=2, num_channels=16)

        )



        # self.map4 = nn.Sequential(
        #     nn.Conv3d(2, out_channel, 1, 1),
        #     # nn.ConvTranspose3d(out_channel, out_channel, 1, 1)
        #     nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
        #     # nn.Softmax(dim =1)
        # )
        #
        # # 128*128 尺度下的映射
        # self.map3 = nn.Sequential(
        #     nn.Conv3d(64, out_channel, 1, 1),
        #     nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=True),
        #     # nn.Softmax(dim =1)
        # )
        #
        # # 64*64 尺度下的映射
        # self.map2 = nn.Sequential(
        #     nn.Conv3d(128, out_channel, 1, 1),
        #     nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=True),
        #     # nn.Softmax(dim =1)
        # )

        # 32*32 尺度下的映射
        # self.map1 = nn.Sequential(
        #     nn.Conv3d(256, out_channel, 1, 1),
        #     nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=True),
        #     # nn.Softmax(dim =1)
        # )

        self.vit = ViT(
            in_channels=1, img_size=(48, 256, 256), patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            pos_embed='perceptron',
            classification=False,

            dropout_rate=0.0
        )

        self.conv1=nn.Sequential(
            nn.ConvTranspose3d(768, 256, 2, 2),
            nn.GroupNorm(num_groups=2,num_channels=256),
            nn.PReLU(256),
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.GroupNorm(num_groups=2,num_channels=128),
            nn.PReLU(128),
            nn.ConvTranspose3d(128, 32, 2, 2),
            nn.GroupNorm(num_groups=2,num_channels=32),
            nn.PReLU(32)
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(768, 256, 2, 2),
            nn.GroupNorm(num_groups=2,num_channels=256),
            nn.PReLU(256),
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.GroupNorm(num_groups=2,num_channels=128),
            nn.PReLU(128),
            nn.ConvTranspose3d(128, 64, 1, 1),
            nn.GroupNorm(num_groups=2,num_channels=64),
            nn.PReLU(64)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(768, 256, 2, 2),
            nn.GroupNorm(num_groups=2,num_channels=256),
            nn.PReLU(256),
            nn.ConvTranspose3d(256, 128, 1, 1),
            nn.GroupNorm(num_groups=2,num_channels=128),
            nn.PReLU(128)
            # nn.ConvTranspose3d(128, 48, 2, 2),
            # nn.PReLU(48)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(768, 256, 1, 1),
            nn.GroupNorm(num_groups=2,num_channels=256),
            nn.PReLU(256)
            # nn.ConvTranspose3d(256, 128, 1, 1),
            # nn.PReLU(128)
            # nn.ConvTranspose3d(128, 48, 2, 2),
            # nn.PReLU(48)
        )

        self.xc=nn.Sequential(#原图像x传过来拼接
            nn.Conv3d(1,8, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=2,num_channels=8),
            nn.PReLU(8),
            nn.Conv3d(8, 16,  3, stride=1, padding=1),
            nn.GroupNorm(num_groups=2,num_channels=16),
            nn.PReLU(16)
            # nn.ConvTranspose3d(256, 128, 1, 1),
            # nn.PReLU(128)
            # nn.ConvTranspose3d(128, 48, 2, 2),
            # nn.PReLU(48)
        )

        self.outconv=nn.Sequential(
            nn.Conv3d(16, 4, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=2,num_channels=4),
            nn.PReLU(4),
            nn.Conv3d(4, out_channel, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=2,num_channels=out_channel),


        )





        # self.conT1 = nn.ConvTranspose3d(768, 256, 2, 2)
        # self.conT2 = nn.ConvTranspose3d(256, 128, 1, 1)



        # class vits(nn.Module):  # 1,128,6,32,32
        #     def proj_feat(self, x, hidden_size, feat_size):
        #         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        #         x = x.permute(0, 4, 1, 2, 3).contiguous()
        #         return x
        #
        #     def __init__(self):
        #         super(vits, self).__init__()
        #         self.vit = ViT(
        #             in_channels=128, img_size=(6, 32, 32), patch_size=(2, 2, 2),
        #             hidden_size=768,
        #             mlp_dim=3072,
        #             num_heads=12,
        #             num_layers=12,
        #             pos_embed='perceptron',
        #             classification=False,
        #
        #             dropout_rate=0.0
        #         )
        #         self.conT1 = nn.ConvTranspose3d(768, 256, 2, 2)
        #         self.conT2 = nn.ConvTranspose3d(256, 128, 1, 1)
        #         # self.conT3 = nn.ConvTranspose3d(64, 32, 2, 2)
        #
        #     def forward(self, x):
        #         x = self.vit(x)[0]
        #         x = self.proj_feat(x, 768, (3, 16, 16))  # (1,768,3,16,16)
        #         x = self.conT1(x)
        #         x = self.conT2(x)
        #         # x = self.conT3(x)
        #         return x

    def forward(self, x):

        x0, hidden_states_out = self.vit(x)
        x3=hidden_states_out[2]
        x3=self.conv1(proj_feat(x3,768,(3,16,16)))
        x6 = hidden_states_out[5]
        x6 = self.conv2(proj_feat(x6, 768, (3, 16, 16)))
        x9 = hidden_states_out[8]
        x9 = self.conv3(proj_feat(x9, 768, (3, 16, 16)))
        x12 = hidden_states_out[11]
        x12 = self.conv4(proj_feat(x12, 768, (3, 16, 16)))
        t1=x3
        t2=x6
        t3=x9
        t4=x12
        out=F.interpolate(self.decoder2(t4),scale_factor=(2,2,2),mode ='trilinear',align_corners=True)





       #  #####################################################################
       #  # ###########################采用插值做上采样##############################
       #  out = F.max_pool3d(self.encoder1(x), 2, 2)
       #
       #
       #  ### #残差连接###
       #  # down_conv1=self.down_conv1(x)
       #  # out=down_conv1+out
       #  ###############
       #
       #  out=F.relu(out)
       #  t1 = out
       #
       #  out = F.max_pool3d(self.encoder2(out), 2, 2)
       #
       #  ### #残差连接###
       #  # down_conv2 = self.down_conv2(t1)
       #  # out = down_conv2 + out
       #  ###############
       #  out=F.relu(out)
       #
       #  t2 = out
       #
       #  out = F.max_pool3d(self.encoder3(out), 2, 2)
       #  ### #残差连接###
       #  # down_conv3 = self.down_conv3(t2)
       #  # out = down_conv3 + out
       #  ###############
       #  out = F.relu(out)
       #  t3 = out
       #  # out=F.max_pool3d(self.encoder4(out), 2, 2)
       #  #
       #  # #残差连接
       #  # # down_conv4=self.down_conv4(t3)
       #  # # out  =down_conv4+out
       #  #
       #  # out = F.relu(out)
       #  #
       #  # ## 残差连接
       #  # #conect_conv2=self.conect_conv2(out)
       #  #
       #  # # t4 = out
       #  # # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))
       #  #
       #  # # t2 = out
       #  # # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
       #  # # print(out.shape,t4.shape)
       #  # #output1 = self.map1(out)
       #  #
       #  # # agatt
       #  # # t3=self.att1(out,t3)
       #  #
       #  # # up_conv1=self.up_conv1(out)
       #  # out = self.decoder2(out)
       #  #
       #  # # scse注意力
       #  # # out = self.scse1(out)
       #  #
       #  # out=F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
       #  #####残差连接######
       #  #out = conect_conv2 + out
       #  # out =up_conv1+out
       #  ##############
       #  # out = F.relu(out)
       #
       #  # #seblock
       # # t3=self.se128(t3)







        out = torch.add(out, t3)

        ####upconv1
        #upconv1=self.up_conv1(out)
        # up_conv2=self.up_conv2(out)




        # agatt
        # t2 = self.att2(out, t2)

       # output2 = self.map2(out)
        out = self.decoder3(out)

        # scse注意力
        # out = self.scse2(out)
        out=F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        ###残差连接#######
        # out=up_conv2+out
        ######
        out = F.relu(out)

        # # seblock
        #t2 = self.se64(t2)

        out = torch.add(out, t2)
        # up_conv3=self.up_conv3(out)


        #output3 = self.map3(out)

        # agatt
        # t1 = self.att3(out, t1)

        out = self.decoder4(out)

        # scse注意力
        # out = self.scse3(out)
        out=F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

        ############残差连接
        # out =up_conv3+out
        ############
        out = F.relu(out)

        # #seblock
        #t1=self.se32(t1)

        #(32,48,256,256)
        out =torch.add(out, t1)
        # up_conv4=self.up_conv4(out)

        #(16,48,256,256)
        out=F.relu(self.decoder5(out))
        out=F.interpolate(out, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        x=self.xc(x)
        x=torch.add(x,out)
        output=self.outconv(x)

        # out=up_conv4+out
        # out = F.relu(out)

        # output4 = self.map4(x)
        # output4=output+0.4*(output1+output2+output3)
        #  output4=torch.cat((output,output1,output2,output3),1)
        #  output4=self.F2OConv(output4)
        sof = nn.Softmax(dim=1)
        outputSof = sof(output)
        outTanh = torch.tanh(output)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            # return output1, output2, output3, output4
            return outputSof, outTanh
        else:
            return outputSof