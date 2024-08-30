
from scipy import ndimage
import numpy as np
import SimpleITK as sitk
from torchvision import transforms
import torch
import os

from sklearn import preprocessing
transform=transforms.Compose([
    transforms.ToTensor(),                  #取值范围会归一化到（0,1）之间
    #transforms.Normalize(mean=0.5,std=0.5)  #设置均值和方差均为0.5
])

# ct = sitk.ReadImage('fixed_data/ct/volume-2.nii', sitk.sitkInt16)
#
# ct_array = sitk.GetArrayFromImage(ct)
#
# #ct_array = ct_array.astype(np.float32)
#
# min_max_scaler = preprocessing.MinMaxScaler()
# minmax_x = min_max_scaler.fit_transform(ct_array)
#ct_array=transform(ct_array)



#seg_array[seg_array>0]=1
# seg_array = seg_array.astype(np.float32)
# for i in range(seg_array.shape[0]):
#     for j in range(seg_array.shape[1]):
#         for k in range(seg_array.shape[2]):
#             # if seg_array[i][j][k]!=0:
#                 print(seg_array[i][j][k])
# print(seg_array.shape)


def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newing = (ct_array - minWindow)/float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1
    #将值域转到0-255之间,例如要看头颅时， 我们只需将头颅的值域转换到 0-255 就行了
    if not normal:
        newing = (newing *255).astype('uint16')

    return newing


def saved_preprocessed(savedImg,origin,direction,xyz_thickness,saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    # newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)



if __name__ == '__main__':
    # ct = sitk.ReadImage('raw_dataset/train/ct/volume-1.nii')
    # ct_array = sitk.GetArrayFromImage(ct)
    # ct_new=window_transform(ct_array,170,60)
    #
    # origin = ct.GetOrigin()
    # direction = ct.GetDirection()
    # xyz_thickness = ct.GetSpacing()
    # save_name="volumeWindow.nii"
    # saved_preprocessed(ct_new,origin, direction, xyz_thickness,save_name)

    ct_path = "raw_dataset/train/ct/volume-"
    newct_path = "fixed_data/ct/volume-"
    newseg_path = "fixed_data/label/segmentation-"
    seg_path = "raw_dataset/train/label/segmentation-"
    # new_path="experiments/823_10val/liverXori/volume-"

    for i in range(130):
        j = i + 1
        ct_p = ct_path + str(j) + '.nii'
        seg_p = seg_path + str(j) + '.nii.gz'
        new_cp = newct_path + str(j) + '.nii'
        new_seg = newseg_path + str(j) + '.nii.gz'
        # print(segStr)

        if os.path.exists(ct_p):
            print(ct_p)

            ct = sitk.ReadImage(ct_p, sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)
            origin = ct.GetOrigin()
            direction = ct.GetDirection()
            xyz_thickness = ct.GetSpacing()

            seg = sitk.ReadImage(seg_p, sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            seg_array[seg_array > 0] = 1

            ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, 0.25, 0.25), order=3)
            seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / 1, 0.25, 0.25), order=0)

            z = np.any(seg_array, axis=(1, 2))  # 返回沿着slice方向的是否有掩膜存在的列表
            start_slice, end_slice = np.where(z)[0][[0, -1]]
            start_slice=start_slice-20
            end_slice=end_slice+20
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

            ct_new = window_transform(ct_array, 170, 60)
            saved_preprocessed(ct_new, origin, direction, xyz_thickness, new_cp)
            saved_preprocessed(seg_array, origin, direction, xyz_thickness, new_seg)




