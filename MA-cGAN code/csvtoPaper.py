import pandas as pd
import os
import matplotlib.pyplot as plt

import time

import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['font.family']='sans-serif'

#解决负号'-'显示为方块的问题

plt.rcParams['axes.unicode_minus'] = False

def main( ):
    file = r'D:\ljhProjects\3DUNet429\3DUNet-Pytorch-master\log_out\liver'
    # file = r'D:\ljhProjects\3DUNet429\3DUNet-Pytorch-master\log_out\tumor'
    fileList=[]
    for root, dirs, files in os.walk(file):
        if root != file:
            break
        for file in files:
            path = os.path.join(root, file)
            fileList.append(path)
            print(path)















    path_ours = "experiments/112_cvit128/train_log_1.csv"
    # path_unet = "experiments/1214_unet/train_log.csv"
    # path_unet = "log_out/1115_untbts_.csv"
    path_ours = "log_out/liver/MA-cGAN_9843.csv"
    path_unet = "log_out/liver/UTNet_9753.csv"

    ydata = []

    xdata = []











#使用python下pandas库读取csv文件

    data_unet=pd.read_csv(path_unet,encoding='gbk')
    data_ours = pd.read_csv(path_ours,encoding='gbk')

####################距离误差

#读取列名为距离误差和时间点的所有行数据

    ydata_ours_loss = data_ours.loc[:500,'Val_Loss']
    # ydata_ours_dice = data_ours.loc[:300, 'Train_dice_liver']
    #
    ydata_unet_loss = data_unet.loc[:300, 'Val_Loss']
    # # ydata_unet_dice = data_unet.loc[:300, 'Train_dice_liver']
    #
    # xdata = data_ours.loc[:300,'epoch']
    #
    #
    xdata2 = data_ours.loc[:300, 'epoch']
    xdata1 = data_ours.loc[:500, 'epoch']

#读取列名为距离误差的前1000行数据

    # ydata = data.loc[:1000,'Train_Loss']

    plt.figure(1)

#点线图
    index = np.arange(0, 300, 3)
    # plt.plot(xdata,ydata,'bo-',label=u'cte_误差',linewidth=0.5,markersize=0)
    # plt.plot(xdata1, ydata_ours_loss,'r',label=u'ours',linewidth=0.8)
    # plt.plot(xdata, ydata_ours_dice,label=u'ours',linewidth=0.8)
    #
    # plt.plot(xdata2, ydata_unet_loss, label=u'unet_Train_Loss',linewidth=0.8)
    # plt.plot(xdata2, ydata_unet_dice, label=u'unet_dice_liver')

    a=fileList[2]
    fileList[2]=fileList[3]
    fileList[3]=a
    for i in fileList:
        name=i.split("\\")[-1].split("_")[0]
        data_unet = pd.read_csv(i, encoding='gbk')
        ydata_ours_dice = data_unet.loc[:400, 'Val_Loss']
        # ydata_ours_dice = data_unet.loc[:500, 'Train_dice_liver']

        xdata = data_unet.loc[:500, 'epoch']
        # plt.plot(xdata, ydata_ours_dice, label=name,linewidth=0.8)
        # plt.plot(xdata, ydata_ours_dice, label=name, linewidth=0.8)
        plt.plot(xdata[index], ydata_ours_dice[index], label=name, linewidth=0.8)


#点图

    # plt.scatter(xdata,ydata,s=0.1)

    # plt.title(u"CTE误差",size=10)

    plt.legend()

    plt.xlabel(u'Epoch',size=10)

    plt.ylabel(u'Tversky Loss',size=10)

#在展示图片前可以将画出的曲线保存到自己路径下的文件夹中

    # plt.savefig('C:\\Users\\yangyukuan\\Desktop\\data_nyj\\12.11\\cte误差.jpg')

    plt.show()

    print ("all picture is starting")

if __name__ == "__main__":

    main()