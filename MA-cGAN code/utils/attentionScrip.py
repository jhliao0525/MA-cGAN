import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class se_block(nn.Module):  # seNet 模型，通道注意力机制
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d,h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1,1)
        return x * y




# 下面是通道注意力机制和空间注意力机制的结合（CBAM），空间注意力机制是每个通道内的看哪个比较重要
# 空间注意力机制他对应的将通道这个维度进行卷积或者池化，对通道进行压缩操作使其变成两个通道，对图片的长宽不进行改变


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b,c,d,h,w=x.size()
        avg_pool=nn.AdaptiveAvgPool3d((d,1,1))
        max_pool=nn.AdaptiveMaxPool3d((d,1,1))
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x



#下面是ECA，其就是将通道注意力机制中的两个全连接操纵变成了一个一维卷积
class eca_block(nn.Module):
    def __init__(self,channel,d,b=1,gamma=2):
        super(eca_block, self).__init__()
        kernel_size=int(abs((math.log(channel,2)+b)/gamma))
        kernel_size=kernel_size if kernel_size%2 else kernel_size+1
        padding=kernel_size//2
        #上面相当于对每个不同的图片，卷积核自适应的进行改变

        #self.avg_pool =nn.AdaptiveAvgPool3d((d,1,1))#参数是否该改成（d,1,1）
        self.conv=nn.Conv2d(1,1,kernel_size=kernel_size,padding=padding,bias=False)
        #padding=(kernel_size-1)//2) 相当于padding=same 即保持输出图片大小不变得到操作
        #为啥这里进入的通道数是1呢，是因为前面有个自适应层，将图片变成了1*1*channel的样子，在下面经过维度变换，此时将维度变成了b*1*c，
        #然后conv1d是对最后一维进行卷积的（同理conv2d是对最后两维进行卷积的）因此就是对channel这个维度进行了一个卷积，
        #此时就可以相当于把一个长方体横过来看（或者说换成了channel和长这个面）此时相当于宽为以前的通道数即1.
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        #print(x.size())
        b,c,d,h,w=x.size()#batch,通道数，深度，高，宽
        avg_pool=nn.AdaptiveAvgPool3d((d,1,1))
        avg=avg_pool(x)
        avg= avg.view([b,1,c,d])
        out=self.conv(avg)
        out=self.sigmoid(out)
        out=out.view([b,c,d,1,1])
        #print(out)
        return out*x

        # y.expand_as(x)是将y的size于x的size进行一个统一，可以看成将y像x一样扩展

####Coordinate Attention Block
class h_sigmoid(nn.Module):
    def __init__(self,inplace=True):# inplace=True：不创建新的对象，直接对原始对象进行修改
        super(h_sigmoid, self).__init__()
        self.relu=nn.ReLU6(inplace=inplace)
    def forward(self,x):
        return self.relu(x+3)/6

class h_swish(nn.Module):
    def __init__(self,inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid=h_sigmoid(inplace=inplace)

    def forward(self,x):
        return x*self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((1,None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1,1, None))
        self.pool_d= nn.AdaptiveAvgPool3d((None,1,1))
        oup=inp

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c,d, h, w = x.size()
        x_d = self.pool_d(x)
        x_h = self.pool_h(x).permute(0, 1, 3,2,4)
        x_w = self.pool_w(x).permute(0, 1, 4,3,2)



        y = torch.cat([x_d,x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)#归一化处理
        y = self.act(y)

        x_d,x_h, x_w = torch.split(y, [d,h, w], dim=2)#按照第三个维度分割成长度为h和w的两个块
        x_h = x_h.permute(0, 1, 3,2,4)
        x_w = x_w.permute(0, 1, 4,3,2)
        a_d=self.conv_d(x_d).sigmoid()
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h*a_d
        #print("coordatt")
        return out


#####scse注意力模块
import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,d,h,w] to q:[bs,1,d,h,w]
        q = self.sigmoid(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool=nn.AdaptiveMaxPool3d(1)
        self.Conv_Squeeze = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.Conv_Excitation = nn.Conv3d(in_channels//2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, U):
        avg = self.avgpool(U)# shape: [bs, c,d, h, w] to [bs, c,1, 1, 1]
        avg = self.Conv_Squeeze(avg) # shape: [bs, c/2]
        avg=self.relu(avg)
        avg = self.Conv_Excitation(avg) # shape: [bs, c]
        max = self.maxpool(U)  # shape: [bs, c,d, h, w] to [bs, c,1, 1, 1]
        max = self.Conv_Squeeze(max)  # shape: [bs, c/2]
        max=self.relu(max)
        max = self.Conv_Excitation(max)  # shape: [bs, c]

        out=self.sigmoid(avg+max)
        #############################gai
        return out*U
        #return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

########ag unet 的注意力
def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

class Agatt_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Agatt_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        g1=F.interpolate(g1,scale_factor=(2,2,2),mode ='trilinear',align_corners=True)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



#####   APA-att   #####
class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class InnerTransBlock(nn.Module):
    '''
    parameters:
        x: low-resolution features from decoder
        y: high-resolution features from encoder
    '''

    def __init__(self, dim, kernel_size, project_dim=2, isCat=False):
        super(InnerTransBlock, self).__init__()

        self.isCat = isCat
        # current output dimension for decoder
        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size

        # kxk group convolution
        self.key_embed = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, kernel_size=2, stride=2, groups=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.key_embed_isCat = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=1, stride=1, groups=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        # two sequential 1x1 convolution
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.ConvTranspose3d(2 * dim, dim, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )
        self.conv1x1_isCat = nn.Sequential(
            nn.ConvTranspose3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.bn = nn.BatchNorm3d(dim)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x, y):
        '''
            x: [B,C,H,W,D]
            y: [B,C/2,2H,2W,2D]
        '''
        k = torch.max(x, self.project_dim)[0] + torch.mean(x, self.project_dim)
        if self.isCat:
            k = self.key_embed_isCat(k)
        else:
            k = self.key_embed(k)
        q = torch.max(y, self.project_dim)[0] + torch.mean(y, self.project_dim)
        qk = torch.cat([q, k], dim=1)

        w = self.embed(qk)
        w = w.unsqueeze(self.project_dim)
        fill_shape = w.shape[-1]
        repeat_shape = [1, 1, 1, 1, 1]
        repeat_shape[self.project_dim] = fill_shape
        w = w.repeat(repeat_shape)

        if self.isCat:
            v = self.conv1x1_isCat(x)
        else:
            v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        y = y.view(B, C, 1, H, W, D)
        y = torch.cat([y, v], dim=2)

        y_gap = y.sum(dim=2)
        y_gap = y_gap.mean((2, 3, 4), keepdim=True)
        y_attn = self.se(y_gap)
        y_attn = y_attn.view(B, C, self.radix)
        y_attn = F.softmax(y_attn, dim=2)
        out = (y * y_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)

        return out.contiguous()

class InnerBlock(nn.Module):
    def __init__(self, dim, kernel_size, project_dim=2):
        super(InnerBlock, self).__init__()

        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.bn = nn.BatchNorm3d(dim)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            # nn.GroupNorm(num_groups=2, num_channels=attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        k = torch.mean(x, self.project_dim) + torch.max(x, self.project_dim)[0]
        k = self.key_embed(k)
        q = torch.mean(x, self.project_dim) + torch.max(x, self.project_dim)[0]
        qk = torch.cat([q, k], dim=1)

        w = self.embed(qk)
        w = w.unsqueeze(self.project_dim)
        fill_shape = w.shape[-1]
        repeat_shape = [1, 1, 1, 1, 1]
        repeat_shape[self.project_dim] = fill_shape
        w = w.repeat(repeat_shape)

        v = self.conv1x1(x)
        # print("v{0},w{1}".format(v.shape,w.shape))
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        s= x_attn.reshape((B, C, self.radix, 1, 1, 1))
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)

        return out.contiguous()


class apa_att(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(apa_att, self).__init__()
        self.sh_att=InnerBlock(dim=in_ch, kernel_size=3, project_dim=2)
        self.sw_att=InnerBlock(dim=in_ch, kernel_size=3, project_dim=3)
        self.sd_att= InnerBlock(dim=in_ch, kernel_size=3, project_dim=4)
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, feat):
        # feat = self.conv1(input)
        sh_attn = self.sh_att(feat)
        sw_attn = self.sw_att(feat)
        sd_attn = self.sd_att(feat)
        a1=self.beta[0]
        attn = (sh_attn * self.beta[0] + sw_attn * self.beta[1] + sd_attn * self.beta[2]) / self.beta.sum()
        attn = self.conv(attn)

        return attn


class apa_decoder(nn.Module):
    def __init__(self, out_ch,isCat):
        super(apa_decoder, self).__init__()
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.sh_att=InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=2,isCat=isCat)
        self.sw_att = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=3,isCat=isCat)
        self.sd_att = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=4,isCat=isCat)
        # self.conv = nn.Sequential(
        #     nn.Conv3d( out_ch,  out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm3d( out_ch),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, input, high_res_input):
        sh_attn = self.sh_att(input, high_res_input)
        sw_attn = self.sw_att(input, high_res_input)
        sd_attn = self.sd_att(input, high_res_input)
        attn = (sh_attn * self.beta[0] + sw_attn * self.beta[1] + sd_attn * self.beta[2]) / self.beta.sum()
        return attn


class apa_decoder_1(nn.Module):
    def __init__(self, out_ch,isCat):
        super(apa_decoder_1, self).__init__()
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.sh_att=InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=2)
        self.sw_att = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=3)
        self.sd_att = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=4)
        # self.conv = nn.Sequential(
        #     nn.Conv3d( out_ch,  out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm3d( out_ch),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, input, high_res_input):
        sh_attn = self.sh_att(input, high_res_input)
        sw_attn = self.sw_att(input, high_res_input)
        sd_attn = self.sd_att(input, high_res_input)
        attn = (sh_attn * self.beta[0] + sw_attn * self.beta[1] + sd_attn * self.beta[2]) / self.beta.sum()
        return attn



#######   apaaa-att 2 ,三个维度不一样  #####
class InnerBlock2(nn.Module):
    def __init__(self, dim, kernel_size, project_dim=2, isD=False):
        super(InnerBlock2, self).__init__()

        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size
        self.isD = isD

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.bn = nn.BatchNorm3d(dim)

        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            # nn.GroupNorm(num_groups=2,num_channels=attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        k = torch.mean(x, self.project_dim) + torch.max(x, self.project_dim)[0]
        k = self.key_embed(k)
        q = torch.mean(x, self.project_dim) + torch.max(x, self.project_dim)[0]
        qk = torch.cat([q, k], dim=1)

        w = self.embed(qk)
        w = w.unsqueeze(self.project_dim)
        # fill_shape = w.shape[-1]
        # if self.isD:
        #     fill_shape = w.shape[-1] // 2
        # else:
        #     fill_shape = w.shape[-1]
        fill_shape=x.shape[self.project_dim]
        repeat_shape = [1, 1, 1, 1, 1]
        repeat_shape[self.project_dim] = fill_shape
        w = w.repeat(repeat_shape)

        v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)

        return out.contiguous()

class apa_att2(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(apa_att2, self).__init__()
        self.sd_att=InnerBlock2(dim=in_ch, kernel_size=3, project_dim=2,isD=True)
        self.sh_att=InnerBlock2(dim=in_ch, kernel_size=3, project_dim=3)
        self.sw_att= InnerBlock2(dim=in_ch, kernel_size=3, project_dim=4)
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, feat):
        # feat = self.conv1(input)
        sd_attn = self.sd_att(feat)
        sh_attn = self.sh_att(feat)
        sw_attn = self.sw_att(feat)

        attn = (sd_attn * self.beta[0] + sh_attn * self.beta[1] + sw_attn * self.beta[2]) / self.beta.sum()
        attn = self.conv(attn)

        return attn