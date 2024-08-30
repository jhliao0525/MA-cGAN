# 通过连通成分分析，移除小区域
import SimpleITK as sitk
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import config
from resultAve import procResult


def get_dices(logits, targets):  # logits预测结果, targets真实标签
    # ##################################
    # #将预测值重新回归0,1
    # softmax=nn.Softmax(dim=1)
    # logits=softmax(logits)
    # ##################################
    logits=torch.from_numpy(logits)
    targets = torch.from_numpy(targets)
    dices = []
    inter = torch.sum(logits[:, :, :] * targets[:, :, :])
    union = torch.sum(logits[:, :, :]) + torch.sum(targets[:, :, :])
    dice = (2. * inter + 1) / (union + 1)
    dices.append(dice.item())

    return np.asarray(dices)
def RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5):
    '''
    two steps:
        step 1: Connected Component analysis: 将输入图像分成 N 个连通域
        step 2: 假如第 N 个连通域的体素小于最大连通域 * rate，则被移除
    :param sitk_maskimg: input binary image 使用 sitk.ReadImage(path, sitk.sitkUInt8) 读取，
                        其中sitk.sitkUInt8必须注明，否则使用 sitk.ConnectedComponent 报错
    :param rate: 移除率，默认为0.5， 小于 1/2最大连通域体素的连通域被移除
    :return:  binary image， 移除了小连通域的图像
    '''

    # step 1 Connected Component analysis
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0  # 获取最大连通域的索引
    maxsize = 0  # 获取最大连通域的体素大小

    # 遍历每一个连通域， 获取最大连通域的体素大小和索引
    for l in stats.GetLabels():  # stats.GetLabels()  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        size = stats.GetPhysicalSize(l)  # stats.GetPhysicalSize(5)=75  表示第5个连通域的体素有75个
        if maxsize < size:
            maxlabel = l
            maxsize = size

    # step 2 获取每个连通域的大小，保留 size >= maxsize * rate 的连通域
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size >= maxsize * rate:
            not_remove.append(l)

    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 1
    # 保存图像
    #outmask = outmask.astype('float32')

    out = sitk.GetImageFromArray(outmask)
    out.SetDirection(sitk_maskimg.GetDirection())
    out.SetSpacing(sitk_maskimg.GetSpacing())
    out.SetOrigin(sitk_maskimg.GetOrigin())  # 使 out 的层厚等信息同输入一样

    return out  # to save image: sitk.WriteImage(out, 'largecc.nii.gz')

def test_dice_compare(segStr, gtStr):
    seg = sitk.ReadImage(segStr, sitk.sitkUInt8)
    gt = sitk.ReadImage(gtStr, sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    gt_array = sitk.GetArrayFromImage(gt)

    return get_dices(seg_array, gt_array)

def procResults(proc_model):
    if proc_model=="bestdice":
        ori_path='{}/bestdice_result/result-'.format(save_path)
        out_path='{}/bestdice_result_proc'.format(save_path)
        out_txt_path='{}/bestdice_result_proc.txt'.format(save_path)
    elif proc_model=="minloss":
        ori_path='{}/minloss_result/result-'.format(save_path)
        out_path='{}/minloss_result_proc'.format(save_path)
        out_txt_path = '{}/minloss_result_proc.txt'.format(save_path)

    j = 0
    k = 0
    outfile = open( out_txt_path, "w", encoding='utf-8')
    if not os.path.exists(out_path): os.mkdir(out_path)
    for i in range(130):
        i = i + 1
        if os.path.exists(ori_path + str(i) + '.nii.gz'):
            k += 1
            input = ori_path+ str(i) + '.nii.gz'

            output = out_path +'/result-'+ str(i) + '-proc.nii.gz'
            print("procing result-" + str(i) + "...")
            sitk_maskimg = sitk.ReadImage(input, sitk.sitkUInt8)
            out = RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5)  # 可以设置不同的比率
            sitk.WriteImage(out, output)

            gtStr = "D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/label/segmentation-" + str(i) + ".nii.gz"
            #     segStr='D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/experiments/1116_apa/result_proc/result-' +str(l) + '-proc.nii.gz'
            #
            test_dice=test_dice_compare(output,gtStr)
            outfile.write("{}.nii,{}\n".format(i, float(test_dice)))
            print("{}.nii,{}".format(i, float(test_dice)))


            # gt = sitk.ReadImage('fixed_data_256_2t/label/' + 'segmentation-' + str(i) + '.nii.gz', sitk.sitkUInt8)
            # seg_array = sitk.GetArrayFromImage(out)
            # gt_array = sitk.GetArrayFromImage(gt)
            # j += float(get_dices(seg_array, gt_array))
            # print(get_dices(seg_array, gt_array))
    outfile.close()  # 关闭读写txt文件
    # print(j / k)
    procResult(out_txt_path)##计算平均值


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    print(save_path)
    # parser = argparse.ArgumentParser(description="remove small connected domains")
    # # parser.add_argument('--input', type=str, default="result-71.nii.gz")
    # # parser.add_argument("--output", type=str, default='result-71-proc-1.nii.gz')
    # parser.add_argument('--input', type=str, default="")
    # parser.add_argument("--output", type=str, default='')
    # args = parser.parse_args()

    # for single image

    # sitk_maskimg = sitk.ReadImage(args.input, sitk.sitkUInt8)
    # out = RemoveSmallConnectedCompont(sitk_maskimg, rate=0.5)  # 可以设置不同的比率
    # sitk.WriteImage(out, args.output)

    procResults(args.test_model)


