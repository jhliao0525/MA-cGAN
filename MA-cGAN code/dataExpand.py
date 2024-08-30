# import imgaug.augmenters as iaa  # 导入iaa
# import cv2

# # 导入原图和分割金标准
# images = cv2.imread(filename='img.png')[:,:,:1]
#
# images = np.expand_dims(images, axis=0).astype(np.float32)  # 尤其注意这里数据格式 (batch_size, H, W, C)
# segmaps = cv2.imread(filename='seg.png')[:,:,:1]/255
#
# segmaps = np.expand_dims(segmaps, axis=0).astype(np.int32)  # segmentation 需要时 int 型
# # 定义数据增强策略
# seq = iaa.Sequential([
#     iaa.Fliplr(p=0.5),  # 这里写按照顺序执行的数据增强策略 (这里是依次进行水平和垂直翻转)
#     iaa.Flipud(p=0.5),
# ])
# # 同时对原图和分割进行数据增强
# images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)
from dataset.transforms import RandomFlip_LR

import numpy as np
import SimpleITK as sitk
import torch
import os
# ori_ct_path='D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/raw_dataset/train/ct/'
# new_ct_path='D:/ljhProjects/3DUNet429/expand_data/ct/'
# ori_seg_path='D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/raw_dataset/train/label/'
# new_seg_path='D:/ljhProjects/3DUNet429/expand_data/label/'
ori_ct_path='testdata/'
new_ct_path='ss/'
ori_seg_path='testdata/'
new_seg_path='ss/'
for j in range(4):
    t = "-"+str(j + 1)+"."
    for i in os.listdir(ori_ct_path):
        print("processing ct:{}".format(os.path.join(ori_ct_path, i)))
        ct = sitk.ReadImage(os.path.join(ori_ct_path, i), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        # ct_array = ct_array / 200
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array)
        if j==0:#进行4种数据增强
            ct1 = ct_array.flip(2)
        else:
            ct1=torch.rot90(ct_array,j,[2,0])

        new_ct = sitk.GetImageFromArray(ct1)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        a = i.split('.')

        sitk.WriteImage(new_ct, os.path.join(new_ct_path, a[0] + t + a[1]))
        print("save newct:{}".format(a[0] + t + a[1]))

    for i in os.listdir(ori_seg_path):
        print("processing seg:{}".format(os.path.join(ori_ct_path, i)))
        seg = sitk.ReadImage(os.path.join(ori_seg_path, i), sitk.sitkInt8)

        seg_array = sitk.GetArrayFromImage(seg)
        seg_array = torch.FloatTensor(seg_array)
        if j==0:#进行4种数据增强
            seg1 = seg_array.flip(2)
            #continue
        else:
            seg1=torch.rot90(seg_array,j,[2,0])

        new_seg = sitk.GetImageFromArray(seg1)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        a = i.split('.')
        sitk.WriteImage(new_seg, os.path.join(new_seg_path, a[0] + t + a[1]))
        print("save newseg:{}".format(a[0] + t + a[1]))

#ct_array = sitk.GetArrayFromImage(ct)






# ct1,seg1=RandomFlip_LR._flip(ct_array,seg_array,prob=1.0)




