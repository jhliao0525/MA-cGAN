## This is the code of ECA
import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)#[32,3,1,1]

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))#[32,3,1,1]->[32,3,1]->[32,1,3]->[32,1,3]
        y=y.transpose(-1, -2).unsqueeze(-1)#[32,1,3]->[32,3,1,1]

        # Multi-scale information fusion
        y = self.sigmoid(y) ## y为每个通道的权重值

        return x * y.expand_as(x) ##将y的通道权重一一赋值给x的对应通道
# 生成batch_size=32，channel=3, hight=128, width=128大小的图片 (注意数据要为tensor类型)
input_x = torch.rand(32, 3, 128, 128, dtype=torch.float32)
print(input_x)  #打印结果看看
model = eca_layer()
output_y = model(input_x)
print(output_y)
