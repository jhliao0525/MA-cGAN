
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.resUnet4 import resUnet4
from models.resUnet_Gai3 import resUnet_Gai3

device =  'cuda'
model = resUnet_Gai3(in_channel=1, out_channel=2, training=True).to(device)

a=torch.randn((1,1,186,256,256),device=device)
b=model(a)
print(a.shape)