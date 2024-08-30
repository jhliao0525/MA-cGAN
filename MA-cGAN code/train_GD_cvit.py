
from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
from tqdm import tqdm
import config
import Unter

# from models import UNet, ResUNet, KiUNet_min, SegNet
# from models.resUnet_Gai import resUnet_Gai
# from models.resUnet4 import resUnet4
from models.resUnet5 import resUnet5
from models.resUnet6 import resUnet6
from models.unet3d_ori_apa import UNet3D_ori_apa
from utils import logger, weights_init, metrics, common, loss
import os
# import numpy as np
from collections import OrderedDict
from models.resUnet_bts import resUnet_bts
from unet3d_ori import UNet3D

# from models.resUnet7 import resUnet_claw
from pix2pixGan import Discriminator, Discriminator_paper, Discriminator_gan,Discriminator_trans,Discriminator_trans128
from models.resUnet_apaatt import resUnet_apa
from models.resUnet_apa_256 import resUnet_apa_256
# from TI_Loss import getTILoss
# from models.unetr2_add import UNETR_max256
# from models.unetr2_cat import UNETR_512add
# from models.unetr_256_add import UNETR_256add
from models.unetr_256_add_del import UNETR_256add_del
# # from resUnet_unetr import resUnet_unetr
from models.apauent import APAUNet
from models.resunet_apa_UD import resUnet_apa_ud
from models.resunet_apa_UD_1 import resUnet_apa_ud_1
from adjust_lr import adjust_learning_rate

transforms = transforms = transforms.Compose([
    # transforms.ToTensor(), #0-1; 格式转为channel,high,width
    transforms.Normalize(mean=0.5, std=0.5)  # 均值和方差均采用0.5，将0-1的范围转变为-1到1的范围
])


def val(model, val_loader, loss_func, n_labels):
    model.eval()

    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            # print("val{}".format(data.shape))
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def train(model, train_loader, dis, d_optimizer, optimizer, loss_func, n_labels, alpha):
    LAMBDA = 10  # l1损失的系数
    D_epoch_loss = 0  # 每一轮的损失
    G_epoch_loss = 0
    tiloss_s=0
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
        # print("train{}".format(data.shape))
        for i in range(1):
            # 训练判别器##############################
            for p in dis.parameters():
                p.data.clamp_(-0.01, 0.01)
            d_optimizer.zero_grad()
            disc_real_output = dis(data, target).mean()  # 判别器输入图片和真实标签

            # d_real_loss = loss_b(disc_real_output, torch.ones_like(disc_real_output, device=device))
            # d_real_loss.backward()
            gen_outputSof, outTanh = model(data)  # 暂时只返回一个结果

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
        #訓練生成器bu使用深監督
        disc_gen_total1 = dis(data, outTanh)
        gen_loss1 = -torch.mean(disc_gen_total1)

        # gen_loss_crossentropy = loss_b(disc_gen_output,
        #                                torch.ones_like(disc_gen_output, device=device))


        #gen_l1_loss = torch.mean(torch.abs(target - gen_outputSof)) # (target是0/1, genout是（-1，1）)
        gen_dice_loss=loss_func(target,gen_outputSof)
        # print("gen_loss:{}".format(gen_loss))
       # gen_loss = gen_loss_crossentropy + (LAMBDA * gen_l1_loss)+(LAMBDA * gen_dice_loss)

        # #tiloss
        # tiloss=getTILoss(gen_outputSof,target,dim=3)
        # tiloss_s+=tiloss.item()
        gen_loss = gen_loss1  + (LAMBDA * gen_dice_loss)
        gen_loss.backward()


        loss3 = loss_func(gen_outputSof, target)



        ###################################
        optimizer.step()  # 生成器更新
        # scheduler.step()  # 余弦退火
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
        print("tiloss:{}".format(tiloss_s))
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        # 训练完一个Epoch，打印提示并绘制生成的图片
        # print("Epoch:", epoch)




    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels == 3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log,G_epoch_loss


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cuda')

    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=args.batch_size, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads,
                            shuffle=False)  ###原shuffle=False

    # model info
    #model = ResUNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = UNet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet4(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet5(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model=UNETR_max256(training=True).to(device)
    # model= UNETR_512add(training=True).to(device)
    # model =  UNETR_256add(training=True).to(device)
    # model = resUnet_bts(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model=resUnet_apa(in_channel=1, out_channel=args.n_labels, training=True).to(device)#      小论文
    # model = resUnet_apa_ud(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet_apa_ud_1(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model=resUnet_apa_256(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model=UNet3D(in_channels=1).to(device)
    # model=APAUNet(1,2,True).to(device)
    # model=UNet3D_ori_apa(in_channels=1,training=True).to(device)
    # model = resUnet6(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet_claw(in_channel=1, out_channel=2, training=True).to(device)
    # model = resUnet_unetr(in_channel=1, out_channel=2, training=True).to(device)
    model = UNETR_256add_del(training=True).to(device)
    # model=Unter.UNETR(in_channels=1,out_channels=2,img_size=(128,128,128),feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     pos_embed='perceptron',
    #     norm_name='instance',
    #     conv_block=True,
    #     res_block=False,
    #     dropout_rate=0.0).to(device)
    #model=UNet_gai().to(device)
    dis = Discriminator_gan()
    # dis=Discriminator_trans128()
    # dis=Discriminator_trans()

    # 初始化网络中的参数
    #model.apply(weights_init.init_model)

    # optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))  # 生成器优化器
    # optimizer = optim.AdamW(model.parameters(), lr=0.0002)  # 生成器优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)  # 生成器优化器
    d_optimizer = optim.AdamW(dis.parameters(), lr=2e-4)  # 判别器优化器
    # optimizer =torch.optim.RMSprop(model.parameters(), lr=2e-4)
    # d_optimizer =torch.optim.RMSprop(dis.parameters(), lr=2e-4)

    # optimizer=torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    # d_optimizer = torch.optim.SGD(dis.parameters(), lr=2e-4,momentum=0.9)
    # 余弦退火
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=8, eta_min=1.0e-6)
    common.print_network(model)
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    model = torch.nn.DataParallel(model, device_ids=[0])  # 将使用的gpu_id为0
    dis = torch.nn.DataParallel(dis, device_ids=[0])

    if args.resume:
        print("继续训练")

        # path_checkpoint = 'experiments/23_6_7_paperAddGan /best_model.pth'
        path_checkpoint = os.path.join('./experiments', args.save) + '/best_model.pth'
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    #########################临时修改学习率##########################
    for param_group in optimizer.param_groups:
        param_group['lr'] = 2e-4
    for param_group in d_optimizer.param_groups:
         param_group['lr'] =2e-4
    #     #########################################################
    loss = loss.TverskyLoss()
   # lossDice=
    # loss = torch.nn.BCELoss()  # 损失函数

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]  # 初始化最优模型的epoch和performance
    min_loss=[0,10]# 初始化最优模型的epoch和loss

    trigger = 0  # early stop 计数器
    loss_trigger=0
    minloss_com=False


    alpha = 0.4  # 深监督衰减系数初始值
    ##############
    D_loss = []  # 判别器和生成器的损失变化
    G_loss = []
    ################
    epoch_1=0

    if args.resume:
        epochNum = checkpoint['epoch']
        epoch_1 = 0
    else:
        epochNum=0
    for epoch in range(epochNum, args.epochs + 1):
        epochNum += 1
       # if epochNum%100==0:
       #     for param_group in optimizer.param_groups:
       #         param_group['lr'] = param_group['lr']*0.5
       #     for param_group in d_optimizer.param_groups:
       #         param_group['lr'] = param_group['lr']*0.5

        ####lr  调整
        # adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=args.epochs + 1,
        #                      scheduler=scheduler,lr_max=0.01,warmup=True)
        # common.adjust_learning_rate(optimizer, epoch, args) #调节学习率




        train_log,G_epoch_loss = train(model, train_loader, dis, d_optimizer, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        loss_trigger+=1
        if loss_trigger>=50:
            minloss_com=True
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0

        if G_epoch_loss < min_loss[1] and minloss_com==False:
            print('Saving min loss model')
            torch.save(state, os.path.join(save_path, 'minloss_model.pth'))
            min_loss[0] = epoch
            min_loss[1] = G_epoch_loss
            loss_trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
        print('The minimun loss at Epoch: {} | {}'.format(min_loss[0],min_loss[1]))




        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop and  minloss_com:
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
