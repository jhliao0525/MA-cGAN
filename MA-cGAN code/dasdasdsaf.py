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


liverSt = 'o_new/liver/att-9652.nii.gz'
tumorSt = 'o_new/tumor1/att-0.78.nii.gz'
mixSt='o_new/mix/att-9652-0.78.nii.gz'
# for i in range(130):
#     # print(i)
#     j = i + 1
#     liverStr = liverSt + str(j) + '-proc.nii.gz'
#     tumorStr = tumorSt + str(j) + '.nii.gz'
#     mixStr=mixSt+ str(j) + '.nii.gz'
#     # print(segStr)
#
#     if os.path.exists(tumorStr):
#         print(tumorStr)

liver = sitk.ReadImage(liverSt, sitk.sitkUInt8)
tumor = sitk.ReadImage(tumorSt, sitk.sitkUInt8)
liver_array = sitk.GetArrayFromImage(liver)
tumor_array = sitk.GetArrayFromImage(tumor)

liver_array[tumor_array == 1] = 2

new_seg = sitk.GetImageFromArray(liver_array)
new_seg.SetDirection(liver.GetDirection())
new_seg.SetOrigin(liver.GetOrigin())
new_seg.SetSpacing((liver.GetSpacing()[0] * 1, liver.GetSpacing()[1] * 1,
                            1))
sitk.WriteImage(new_seg, mixSt)