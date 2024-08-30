from __future__ import division
import os
import SimpleITK as sitk
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy.spatial.distance import pdist
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt
from mindspore import Tensor
from mindspore.nn.metrics import HausdorffDistance
from sklearn.metrics import roc_curve

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import utils.metrics
import GeodisTK
from scipy import ndimage
import torch


liverSt = 'experiments/1121_apa_128/bestdice_result_proc/result-'
tumorSt = 'experiments/1203_apa_toumr/bestdice_result/result-'
mixSt='experiments/1203_apa_toumr/mix_result/result-'
for i in range(130):
    # print(i)
    j = i + 1
    liverStr = liverSt + str(j) + '-proc.nii.gz'
    tumorStr = tumorSt + str(j) + '.nii.gz'
    mixStr=mixSt+ str(j) + '.nii.gz'
    # print(segStr)

    if os.path.exists(tumorStr):
        print(tumorStr)

        liver = sitk.ReadImage(liverStr, sitk.sitkUInt8)
        tumor = sitk.ReadImage(tumorStr, sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver)
        tumor_array = sitk.GetArrayFromImage(tumor)

        liver_array[tumor_array == 1] = 2

        new_seg = sitk.GetImageFromArray(liver_array)
        new_seg.SetDirection(liver.GetDirection())
        new_seg.SetOrigin(liver.GetOrigin())
        new_seg.SetSpacing((liver.GetSpacing()[0] * 1, liver.GetSpacing()[1] * 1,
                            1))
        sitk.WriteImage(new_seg, mixStr)