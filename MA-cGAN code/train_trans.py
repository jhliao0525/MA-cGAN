###########806model + wgan
import Unter
from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import torch.autograd as autograd
from networks.UXNet_3D.network_backbone import UXNET
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from torch.autograd import Variable
from monai.networks.nets import ViT,SwinUNETR
from models import UNet, ResUNet, KiUNet_min, SegNet
from models.resUnet_Gai import resUnet_Gai
from models.resUnet4 import resUnet4
from Unter import *

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict

from pix2pixGan import Discriminator, Discriminator_paper, Discriminator_gan,Discriminator_trans,Discriminator_trans2


transforms = transforms = transforms.Compose([
    # transforms.ToTensor(), #0-1; 格式转为channel,high,width
    transforms.Normalize(mean=0.5, std=0.5)  # 均值和方差均采用0.5，将0-1的范围转变为-1到1的范围
])

Tensor = torch.cuda.FloatTensor

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
            # outputSof, outTanh = model(data)
            # loss = loss_func(outputSof, target)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log

# 1.3 定义函数compute_gradient_penalty()完成梯度惩罚项
# 惩罚项的样本X_inter由一部分Pg分布和一部分Pr分布组成，同时对D(X_inter)求梯度，并计算梯度与1的平方差，最终得到gradient_penalties
lambda_gp = 10
# 计算梯度惩罚项
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(model, train_loader, dis, d_optimizer, optimizer, loss_func, n_labels, alpha):
    LAMBDA = 10  # l1损失的系数
    D_epoch_loss = 0  # 每一轮的损失
    G_epoch_loss = 0
    count = len(train_loader)

    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)
    loss_b = torch.nn.BCELoss()


    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):


        data, target = data.float(), target.long()
        # print("data原图：{}".format(data))
        # print("没处理的target标签：{}".format(target))
        target = common.to_one_hot_3d(target, n_labels)
        #target=transforms(target)############################
        # print("target真实标签{}".format(target))
        # print("target*2-1{}".format(target*2-torch.ones_like(target)))
        # target=target*2-torch.ones_like(target)
        data, target = data.to(device), target.to(device)




       #  d_optimizer.zero_grad()
       #  disc_real_output = dis( data,target)  # 判别器输入真实图片
       # # print("disc_real_output{}:".format(disc_real_output))
       #  d_real_loss = loss_b(disc_real_output,
       #                        torch.ones_like(disc_real_output, device=device))
       #  d_real_loss.backward()
       #
       #  # 生成器输入随机张量得到生成图片
       # # gen_output = optimizer(data)
       #  gen_outputSof, gen_output = model(data)#####此处得到gen_out是outtanh
       #  # 判别器输入生成图像，注意此处的detach方法
       #  disc_gen_output = dis(gen_output.detach(),data)
       #  d_fake_loss = loss_b(disc_gen_output,
       #                        torch.zeros_like(disc_gen_output, device=device))
       #  d_fake_loss.backward()
       #
       #  disc_loss = d_real_loss + d_fake_loss  # 判别器的总损失
       #  d_optimizer.step()
       #
       #  optimizer.zero_grad()
       #  disc_gen_output = dis( gen_output,data)  # 判别器输入生成图像
       #  gen_loss_crossentropy = loss_b(disc_gen_output,
       #                                  torch.ones_like(disc_gen_output, device=device))
       #  #gen_l1_loss = torch.mean(torch.abs(target - gen_output))  # L1损失，torch.mean()取均值
       #  gen_l1_loss = loss_func(target, gen_outputSof)
       #  gen_loss = gen_loss_crossentropy + (LAMBDA * gen_l1_loss)
       #  gen_loss.backward()
       #  optimizer.step()

        for i in range(1):
            # 训练判别器##############################
            for p in dis.parameters():
                p.data.clamp_(-0.01, 0.01)
            d_optimizer.zero_grad()
            disc_real_output = dis(data, target).mean() # 判别器输入图片和真实标签

            # d_real_loss = loss_b(disc_real_output, torch.ones_like(disc_real_output, device=device))
            # d_real_loss.backward()
            gen_outputSof, outTanh = model(data) # 暂时只返回一个结果

            # gen_output=output[3]#返回output4作为生成虚假标签

            # print("gen_output虚假标签{}".format(gen_output))

            # 判别器输入生成图像，注意此处的detach方法
            disc_gen_output = dis(data, outTanh.detach()).mean()  # 判别器输入原图和虚假标签

            # 判别器输入生成图像，注意此处的detach方法
            # disc_gen_output = dis(data, outTanh.detach()).mean()  # 判别器输入原图和虚假标签
            # d_fake_loss = loss_b(disc_gen_output,
            #                      torch.zeros_like(disc_gen_output, device=device))

            disc_loss = -disc_real_output + disc_gen_output

            ####################################################
            disc_loss.backward()

            # disc_loss = d_real_loss + d_fake_loss  # 判别器的总损失
            # print("disc_loss:{}".format(disc_loss))
            d_optimizer.step()

            ##################################################

        ############################################
        # 训练生成器
        optimizer.zero_grad()
        # output = model(data)#生成器得到的标签

        ###訓練生成器使用深監督
        # disc_gen1=[]
        # for i in range(4):
        #     disc_gen_output = dis(data, outTanh[i])  # 将原图和生成的虚假标签输入判别器，此处没有detach
        #     disc_gen1.append(disc_gen_output)
        # disc_gen_total1=0.4*(disc_gen1[0]+disc_gen1[1]+disc_gen1[2])+disc_gen1[3]
        #######
        # 訓練生成器bu使用深監督
        disc_gen_total1 = dis(data, outTanh)
        gen_loss1 = -torch.mean(disc_gen_total1)

        # gen_loss_crossentropy = loss_b(disc_gen_output,
        #                                torch.ones_like(disc_gen_output, device=device))

        # gen_l1_loss = torch.mean(torch.abs(target - gen_outputSof)) # (target是0/1, genout是（-1，1）)
        gen_dice_loss = loss_func(target, gen_outputSof)
        # print("gen_loss:{}".format(gen_loss))
        # gen_loss = gen_loss_crossentropy + (LAMBDA * gen_l1_loss)+(LAMBDA * gen_dice_loss)
        gen_loss = gen_loss1 + (LAMBDA * gen_dice_loss)
        gen_loss.backward()




        # for i in range(1):
        #     # 训练判别器##############################
        #     # for p in dis.parameters():
        #     #     p.data.clamp_(-0.01, 0.01)
        #     d_optimizer.zero_grad()
        #     disc_real_output = dis(data, target)  # 判别器输入图片和真实标签
        #
        #
        #
        #     d_real_loss = loss_b(disc_real_output, torch.ones_like(disc_real_output, device=device))
        #     d_real_loss.backward()
        #
        #     gen_outputSof, outTanh = model(data)  # 暂时只返回一个结果
        #
        #     # gen_output=output[3]#返回output4作为生成虚假标签
        #
        #     # print("gen_output虚假标签{}".format(gen_output))
        #
        #
        #         # 判别器输入生成图像，注意此处的detach方法
        #     disc_gen_output = dis(data, outTanh.detach())  # 判别器输入原图和虚假标签
        #     #disc_gen_output = dis(data, gen_outputSof.detach()).mean()  # 判别器输入原图和虚假标签
        #
        #     # 判别器输入生成图像，注意此处的detach方法
        #     # disc_gen_output = dis(data, outTanh.detach()).mean()  # 判别器输入原图和虚假标签
        #     d_fake_loss = loss_b(disc_gen_output,
        #                           torch.zeros_like(disc_gen_output, device=device))
        #     d_fake_loss.backward()
        #
        #     #disc_loss = -disc_real_output + disc_gen_output
        #
        #     ####################################################
        #     #disc_loss.backward()
        #
        #     disc_loss = d_real_loss + d_fake_loss  # 判别器的总损失
        #     # print("disc_loss:{}".format(disc_loss))
        #     d_optimizer.step()
        #
        #     ##################################################
        #
        #
        #
        # ############################################
        # # 训练生成器
        # optimizer.zero_grad()
        # # output = model(data)#生成器得到的标签

        ###訓練生成器使用深監督
        # disc_gen1=[]
        # for i in range(4):
        #     disc_gen_output = dis(data, outTanh[i])  # 将原图和生成的虚假标签输入判别器，此处没有detach
        #     disc_gen1.append(disc_gen_output)
        # disc_gen_total1=0.4*(disc_gen1[0]+disc_gen1[1]+disc_gen1[2])+disc_gen1[3]
        #######
        #訓練生成器bu使用深監督
       #  disc_gen_total1 = dis(data, outTanh)
       # # disc_gen_total1 = dis(data, gen_outputSof)
       #  #gen_loss1 = -torch.mean(disc_gen_total1)
       #
       #  gen_loss_crossentropy = loss_b(disc_gen_output,
       #                                  torch.ones_like(disc_gen_output, device=device))
       #
       #
       #  gen_l1_loss = torch.mean(torch.abs(target - gen_outputSof)) # (target是0/1, genout是（-1，1）)
       #  #gen_dice_loss=loss_func(target,gen_outputSof)
       #
       #  #bceloss = loss_func(gen_outputSof, target)
       #  #diceloss = loss_func(gen_outputSof, target)
       #  # print("gen_loss:{}".format(gen_loss))
       # # gen_loss = gen_loss_crossentropy + (LAMBDA * gen_l1_loss)+(LAMBDA * gen_dice_loss)
       #  #gen_loss = gen_loss1  + (LAMBDA * gen_dice_loss)
       #  gen_loss = gen_loss_crossentropy + (LAMBDA * gen_l1_loss)
       #  gen_loss.backward()


        loss3 = loss_func(gen_outputSof, target)



        ###################################
        optimizer.step()  # 生成器更新
        #scheduler.step()  # 余弦退火
        train_loss.update(loss3.item(), data.size(0))  # 原代码的损失函数
        train_dice.update(gen_outputSof, target)  # 原代码的损失函数

        # print('（1）DICE计算结果，      DSI       = {0:.4}'.format(calDSI(target, gen_outputSof)))  # 保留四位有效数字
        # print('（2）VOE计算结果，       VOE       = {0:.4}'.format(calVOE(target, gen_outputSof)))
        # print('（3）RVD计算结果，       RVD       = {0:.4}'.format(calRVD(target, gen_outputSof)))

        with torch.no_grad():
            D_epoch_loss += disc_loss.item()
            G_epoch_loss += gen_loss.item()
    with torch.no_grad():
        D_epoch_loss /= count
        G_epoch_loss /= count
        print("D_epoch_loss:{}".format(D_epoch_loss))
        print("G_epoch_loss:{}".format(G_epoch_loss))
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        # 训练完一个Epoch，打印提示并绘制生成的图片
        # print("Epoch:", epoch)


    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels == 3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cuda')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads,
                            shuffle=False)  ###原shuffle=False

    # model info
    #model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    #model = UNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    #model = resUnet4(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = Unter.UNETR(in_channels=1, out_channels=2, img_size=(48, 256, 256), feature_size=16,
    #                     hidden_size=768,
    #                     mlp_dim=3072,
    #                     num_heads=12,
    #                     pos_embed='perceptron',
    #                     norm_name='instance',
    #                     conv_block=True,
    #                     res_block=True,
    #                     dropout_rate=0.0)
    # model.to(device)
    # model = SwinUNETR(
    #     img_size=(128, 128, 128),
    #     in_channels=1,
    #     out_channels=1,
    #     feature_size=12,
    #
    # ).to(device)
    model = UXNET(
        in_chans=1,
        out_chans=1,
        depths=[2, 2, 2, 2],
        feat_size=[6,12,24,48],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)

    if args.pretrain:
        print("load pretrain")
        path_checkpoint = 'preModel/UNETR_model_best_acc/UNETR_model_best_acc.pth'
        checkpoint = torch.load(path_checkpoint)
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
        # model.load_state_dict(model_dict)

    #model=UNet_gai().to(device)
    dis = Discriminator_gan()

    # trans1 = TransformerBlock(hidden_size=768, mlp_dim=3072, num_heads=12, dropout_rate=0.0, qkv_bias=False).to(device)
    # patch_embedding = PatchEmbeddingBlock(
    #     in_channels=64,
    #
    #     img_size=(12, 64, 64),
    #     patch_size=6,
    #     hidden_size=768,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     dropout_rate=0.0,
    #
    # ).to(device)
    #



    # 初始化网络中的参数
    #model.apply(weights_init.init_model)

    # optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))  # 生成器优化器
    #optimizer = optim.AdamW(model.parameters(), lr=0.0002)  # 生成器优化器
    # d_optimizer = optim.AdamW(dis.parameters(), lr=0.0002)  # 生成器优化器
    # d_optimizer = optim.Adam(dis.parameters(), lr=0.00002, betas=(0.5, 0.999))  # 判别器优化器
    optimizer =torch.optim.RMSprop(model.parameters(), lr=2e-4)
    d_optimizer =torch.optim.RMSprop(dis.parameters(), lr=1e-5)
    # 余弦退火
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=4, eta_min=1.0e-6)
    common.print_network(model)
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    model = torch.nn.DataParallel(model, device_ids=[0])  # 将使用的gpu_id为0
    dis = torch.nn.DataParallel(dis, device_ids=[0])


    if args.resume:
        print("继续训练")
        path_checkpoint = 'experiments/106trans_gan/best_model.pth'
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    #########################临时修改学习率##########################
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 0.0002
    # #     # print(param_group['lr'])
    # for param_group in d_optimizer.param_groups:
    #      param_group['lr'] = 1e-6
         # print(param_group['lr'])
    #     #########################################################
    loss = loss.TverskyLoss()
   # lossDice=
    # loss = torch.nn.BCELoss()  # 损失函数

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值
    ##############
    D_loss = []  # 判别器和生成器的损失变化
    G_loss = []
    ################
    if args.resume:
        epochNum = checkpoint['epoch']
    else:
        epochNum=0
    for epoch in range(epochNum, args.epochs + 1):
        epochNum += 1
       # if epochNum%100==0:
       #     for param_group in optimizer.param_groups:
       #         param_group['lr'] = param_group['lr']*0.5
       #     for param_group in d_optimizer.param_groups:
       #         param_group['lr'] = param_group['lr']*0.5

        # common.adjust_learning_rate(optimizer, epoch, args) #调节学习率
        train_log = train(model, train_loader, dis, d_optimizer, optimizer, loss, args.n_labels, alpha)
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
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()

    ########存储训练器和判别器的loss
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(os.path.join('./experiments', args.save) + "/DaG_loss")

    for i in range(epochNum):
        writer.add_scalars('D_', {'loss': D_loss[i]}, i)
        writer.add_scalars('G_', {'loss': G_loss[i]}, i)
    writer.close()
