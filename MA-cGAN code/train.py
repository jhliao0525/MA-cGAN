from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models import UNet, ResUNet, KiUNet_min, SegNet
from models.ResUNet_ori import ResUNet_ori
from models.transunet3d import transUNet3D
from models.unet3d_ori import UNet3D
from models.Vnet import VNet
from models.utnetv2.dim3.utnetv2 import UTNetV2
from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from models.unetr_ori import UNETR_ori
from collections import OrderedDict
from models.trans_bts_ori import transbts
from monai.networks.nets import ViT,SwinUNETR
from models.SegNet import SegNet
from models.UNet1 import UNet1
from models.resunet_apa_UD import resUnet_apa_ud
from models.resUnet_apaatt import resUnet_apa

from models.resUnet6 import resUnet6
from models.resUnet_Gai import resUnet_Gai

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()

        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        s = data.size(0)

        optimizer.zero_grad()

        output = model(data)
        # loss0 = loss_func(output[0], target)
        # loss1 = loss_func(output[1], target)
        # loss2 = loss_func(output[2], target)
        # loss3 = loss_func(output[3], target)
        # loss = loss3 + alpha * (loss0 + loss1 + loss2)

        loss3 = loss_func(output, target)
        loss=loss3


        loss = loss.requires_grad_()
        loss.backward()
        optimizer.step()

        train_loss.update(loss3.item(), data.size(0))
        # train_dice.update(output[3], target)
        train_dice.update(output, target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels == 3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    # model info
    # model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True, phi=4).to(device)
    # model = ResUNet_ori(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = UNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = SegNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model =transUNet3D(in_channels=1,num_classes=2).to(device)
    model=transbts(in_channel=1, out_channel=2, training=True).to(device)
    # model=SegNet(in_channel=1, out_channel=2, training=True).to(device)
    # model = SwinUNETR(
    #     img_size=(128, 128, 128),
    #     in_channels=1,
    #     out_channels=2,
    #     feature_size=12,
    #
    # ).to(device)
    # model=UTNetV2(in_chan=1, num_classes=2).to(device)
    # model=UNETR_ori(in_channels=1,out_channels=2,img_size=(128,128,128),feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=False,
    #     dropout_rate=0.0).to(device)
    # model=UNet3D(in_channels=1, num_classes=2, batch_normal=True, bilinear=True).to(device)
    # model=VNet(n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=False).to(device)
    # model.apply(weights_init.init_model)
    # model=UNet1(in_channel=1, out_channel=2, training=True).to(device)
    # model=resUnet_apa_ud(in_channel=1, out_channel=2, training=True).to(device)
    # model = resUnet6(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model=resUnet_Gai(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet_apa(in_channel=1, out_channel=args.n_labels, training=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    common.print_network(model)  # 打印网络参数
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    model = torch.nn.DataParallel(model, device_ids=[0])  # 将使用的gpu_id为0
    print(args.resume)

    if args.resume:
        print("继续训练")
        path_checkpoint = 'experiments/823_tooth_transunet3d/best_model.pth'
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        #########################临时修改学习率##########################
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 1e-4

    #     #########################################################


    loss = loss.TverskyLoss()  # 损失函数

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值

    if args.resume:
        epochNum = checkpoint['epoch']
        epoch_1 = 0
    else:
        epochNum=0
    for epoch in range(epochNum, args.epochs + 1):
        epochNum += 1

        # common.adjust_learning_rate(optimizer, epoch, args)  # 动态调整学习率
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= 50:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()