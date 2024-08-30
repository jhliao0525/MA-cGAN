import numpy as np
import os
import SimpleITK as sitk



ct_path="ct_window/ct/volume-"
seg_path="ct_window/label/segmentation-"
new_path="experiments/823_10val/ct/volume-"


for i in range(130):
    # print(i)
    j = i + 1
    ct_p = ct_path + str(j) + '.nii'
    seg_p = seg_path + str(j) + '-proc.nii.gz'
    new_cp=new_path+str(j) + '.nii'
    # print(segStr)

    if os.path.exists(seg_p):
        print(ct_p)

        ct = sitk.ReadImage(ct_p, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_p, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)


       # ct_array = 0 + (ct_array - 42) * (255 - 0) / (232 - 42)
       #  ct_array = 0 - ct_array

        newct = ct_array * seg_array
        newct[newct == 0] = -200
        # newct[newct >= 200] = 200

        newct = sitk.GetImageFromArray(newct)
        newct.SetDirection(ct.GetDirection())
        newct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        sitk.WriteImage(newct, new_cp)
