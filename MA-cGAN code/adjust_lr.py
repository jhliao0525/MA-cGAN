import matplotlib.pyplot as plt
import math
import torch
from torchvision.models import resnet50
from math import cos, pi


def adjust_learning_rate(optimizer, current_epoch, max_epoch,scheduler, lr_max=0.01, warmup_epoch=50,warmup=True):
    warmup_epoch = warmup_epoch if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif current_epoch < max_epoch:
        scheduler.step()




# model = resnet50(pretrained=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# lr_max = 0.1
# lr_min = 0.00001
# max_epoch = 10 * 5
# lrs = []
# for epoch in range(10 * 40):
#     adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=max_epoch, lr_min=lr_min, lr_max=lr_max,
#                          warmup=True)
#     print(optimizer.param_groups[0]['lr'])
#     lrs.append(optimizer.param_groups[0]['lr'])
#     optimizer.step()
#
# plt.plot(lrs)
# plt.show()
