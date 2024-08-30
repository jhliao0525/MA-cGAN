###计算论文中测试样本的肿瘤的分割准确值
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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import utils.metrics
import GeodisTK
from scipy import ndimage
import torch


# seg = sitk.ReadImage('experiments/806model/result-proc/result-10-proc.nii.gz', sitk.sitkUInt8)
# #seg = sitk.ReadImage('result-71.nii.gz', sitk.sitkUInt8)
# gt = sitk.ReadImage('fixed_data/label/segmentation-10.nii.gz', sitk.sitkUInt8)
# seg_array = sitk.GetArrayFromImage(seg)
# gt_array = sitk.GetArrayFromImage(gt)
# # for i in range(gt_array.shape[0]):
# #     for j in range(gt_array.shape[1]):
# #         for k in range(gt_array.shape[2]):
# #             if gt_array[i][j][k]!=0:
# #                 print(seg_array[i][j][k])
# # print(sum(gt_array))
# # print(sum(seg_array))
#
#
# m = seg_array.tolist()
# m.remove(m[0])
# m = np.array(m)
# m=m.flatten()
# #print(m.shape)
#
# n = gt_array.tolist()
# n.remove(n[0])
# n = np.array(n)
# n=n.flatten()

###########手动计算IOU
# s=0
# b=0
# for i in range(16187392):
#     if m[i]==n[i] and m[i]==1:
#         s+=1
#     if  m[i]==1 or n[i]==1:
#         b+=1
# print(float(s/b))
##############################


#1.IOU
##########求评价指标jaccard(IOU)交并比##################################
def jaccard(seg,gt):
    seg = sitk.GetArrayFromImage(seg)
    gt = sitk.GetArrayFromImage(gt)
    #将array转成numpy
    seg = seg.tolist()
    seg.remove(seg[0])
    seg = np.array(seg)
    seg = seg.flatten()
    # 将array转成numpy
    gt = gt.tolist()
    gt.remove(gt[0])
    gt = np.array(gt)
    gt = gt.flatten()

    X = np.vstack([seg, gt])
    d2 = pdist(X, 'jaccard')
    return(1 - d2)

# print("jaccard(IOU):{}".format(jaccard(seg,gt)))
################################################################################


#2.召回率
############################召回率计算#############################################
def recallCompute(y_true,y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred,  labels=None, pos_label=1,average='binary', sample_weight=None,zero_division="warn")

# print("召回率(recall)：{}".format(recallCompute(n,m)))
##################################################################################

#3.准确率
##################准确率计算############################################################
def accuaryCompute(y_true,y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred,  normalize=True, sample_weight=None)

# print("准确率(accuary):{}".format(accuaryCompute(n,m)))
##############################################################################

#########  jaccard1计算   ###########################################################
# def jaccard1 (GT, SEG)  :
#     # SEG, GT are the binary segmentation and ground truth areas, respectively.
#     # jaccard index
#     return jaccard_score(GT,SEG)
# print("jaccard1:{}".format(jaccard1(n,m)))
################################################################################

#4.f1
##############  f1计算   ##############################################################
def f1Compute(y_true,y_pred):
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    # print("\nF1 score (F-measure): " + str(F1_score))
    return (F1_score)
# print("f1:{}".format(f1Compute(n,m)))
##################################################################################

#5.查准率
############### 查准率（Precision，P）###################################
def precesionCompute(y_true,y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred,  labels=None, pos_label=1, average='binary', sample_weight=None,zero_division="warn")

# print("查准率：{}".format(precesionCompute(n,m)))
##################################################################################

#6.豪斯多夫距离
##################豪斯多夫距离计算#########################################
def get_hausdoff(label,predict):


    labelPred=sitk.GetImageFromArray(predict)
    labelTrue=sitk.GetImageFromArray(label)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue,labelPred)
    avg_hausd = hausdorffcomputer.GetAverageHausdorffDistance()
    hausd = hausdorffcomputer.GetHausdorffDistance()
    return hausd

# print("豪斯多夫：{}".format(get_hausdoff(gt_array,seg_array)))
########################################################################

#7.dice计算
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
# print("dice:{}".format(get_dices(seg_array,gt_array)))

def hasd1(gt,my_mask):



    gt = sitk.GetImageFromArray(gt, isVector=False)
    my_mask = sitk.GetImageFromArray(my_mask, isVector=False)

    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(gt > 0.5, my_mask > 0.5)
    AvgHD = hausdorffcomputer.GetAverageHausdorffDistance()
    HD = hausdorffcomputer.GetHausdorffDistance()

    # dice_dist = sitk.LabelOverlapMeasuresImageFilter()
    # dice_dist.Execute(gt > 0.5, my_mask > 0.5)
    # dice = dice_dist.GetDiceCoefficient()
    return HD
# print(hasd1(gt_array,seg_array))
#
#
#
# x = Tensor(np.array(seg_array))
# y = Tensor(np.array(gt_array))
# metric = HausdorffDistance()
# metric.clear()
# metric.update(x, y, 0)
# distance = metric.eval()
# print(distance)
#



# 豪斯多夫距离
# def HausdorffCompute(y_true,y_pred):
#     return directed_hausdorff(y_pred,y_true)
# print(HausdorffCompute(gt_array,seg_array))
#
# #自定义豪斯多夫距离
#
# def Hausdorff3dCompute(y_true,y_pred):
#     # 豪斯多夫python实现过程
#     import numpy as np
#     import operator
#
#     A = [[1, 3, 3],
#          [4, 5, 6]]
#     B = [[1, 2, 3],
#          [4, 8, 7]]
#
#     asize = [len(A), len(A[0])]
#     bsize = [len(B), len(B[0])]
#
#     if asize[1] != bsize[1]:
#         print('The dimensions of points in the two sets are not equal')
#
#     fhd = 0
#     for i in range(asize[0]):
#         mindist = float("inf")
#         for j in range(bsize[0]):
#             tempdist = np.linalg.norm(list(map(operator.sub, A[i], B[j])))
#             if tempdist < mindist:
#                 mindist = tempdist
#         fhd = fhd + mindist
#     fhd = fhd / asize[0]
#     fhd
#
#     rhd = 0
#     for j in range(bsize[0]):
#         mindist = float("inf")
#         for i in range(asize[0]):
#             tempdist = np.linalg.norm(list(map(operator.sub, A[i], B[j])))
#             if tempdist < mindist:
#                 mindist = tempdist
#         rhd = rhd + mindist
#     rhd = rhd / bsize[0]
#
#     mhd = max(fhd, rhd)
#
#     fhd, rhd, mhd



# print(len(A))
# print(A.shape)

# from scipy import ndimage
#
# def voxelToReal(pt, affine_matrix=None):
# 	affine_matrix = np.array([[spacing[0], 0, 0, origin[0],
#     						 [0, spacing[1], 0, origin[1],
#                              [0, 0, spacing[1], origin[2],
#                              [0, 0, 0, 1]])
#
# 	real = affine_matrix * pt
#     return real[:3]
#
# def distance_A_to_B(A, B):
# 	tree_B = KDTree(np.array(B))
#     distance_A_to_B, indices = tree_B.query(np.array(A))
#     return distance_A_to_B, indices
#
# def ASSD(seg, gt):
# 	struct = ndimage.generate_binary_structure(3, 1)
#
#     ref_border = gt ^ ndimage.binary_erosion(gt, struct, border_value=0)
#     ref_border_voxels = np.array(np.where(ref_border))  # 获取gt边界点的坐标,为一个n*dim的数组
#
#     seg_border = seg ^ ndimage.binary_erosion(seg, struct, border_value=0)
#     seg_border_voxels = np.array(np.where(seg_border)) # 获取seg边界点的坐标,为一个n*dim的数组
#
# 	# 将边界点的坐标转换为实数值,单位一般为mm
#     ref_real = voxelToReal(seg_border_voxels, affine_matrix)
#     gt_real = voxelToReal(ref_border_voxels, affine_matrix)
#
#     tree_ref = KDTree(np.array(ref_border_voxels_real))
#     dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real, k=1)
#     tree_seg = KDTree(np.array(seg_border_voxels_real))
#     dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real, k=1)
#
#     assd = (dist_seg_to_ref.sum() + dist_ref_to_seg.sum()) / (len(dist_seg_to_ref) + len(dist_ref_to_seg))
#
#     return assd


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


def resultCompute(trainlist):#计算所有结果求平均
    gtSt = 'D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/label/segmentation-'
    segSt = 'D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/experiments/23_6_6_paperAddMa_1/bestdice_result_proc/result-'


    jaccardL_test = []
    recall_test = []
    accuary_test = []
    precision_test = []
    dice_test = []
    hausdorff95_test = []
    asd_test = []

    for j in trainlist:
        # print(i)

        gtStr = gtSt + str(j) + '.nii.gz'
        segStr = segSt + str(j) + '-proc.nii.gz'
        # print(segStr)

        if os.path.exists(segStr):
            # print(gtStr)


            seg = sitk.ReadImage(segStr, sitk.sitkUInt8)
            gt = sitk.ReadImage(gtStr, sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            gt_array = sitk.GetArrayFromImage(gt)
            m = seg_array.tolist()
            m.remove(m[0])
            m = np.array(m)
            m = m.flatten()
            # print(m.shape)

            n = gt_array.tolist()
            n.remove(n[0])
            n = np.array(n)
            n = n.flatten()

            jac=float(jaccard(seg, gt))
            # rec=float(recallCompute(n, m))
            # acc=float(accuaryCompute(n, m))
            # has=float(HausdorffCompute(gt_array,seg_array))
            # pre=float(precesionCompute(n, m))
            # hau=float(get_hausdoff(gt_array, seg_array))
            dic=float(get_dices(seg_array, gt_array))
            print("{}:{}".format(j,dic))
            hau95=float(binary_hausdorff95(seg_array, gt_array))
            ass=float(binary_assd(seg_array,gt_array))
            jaccardL_test.append(jac)
            # recall_test.append(rec)
            # accuary_test.append(acc)
            # HausdorffComputelist_test.append(has)
            # precision_test.append(pre)
            # hausdorff_test.append(hau)
            dice_test.append(dic)

            hausdorff95_test.append(hau95)

            asd_test.append(ass)






    print("#########################################################")
    print("测试集指标：")
    print('jaccardsum：{}'.format(sum(jaccardL_test)))
    print(len(jaccardL_test))
    print("jaccard:{}".format(sum(jaccardL_test) / len(jaccardL_test)))
    # print("recall:{}".format(sum(recall_test) / len(recall_test)))
    # print("accuary:{}".format(sum(accuary_test) / len(accuary_test)))
    # print("has:{}".format(sum(HausdorffComputelist_test) / len(HausdorffComputelist_test)))
    # print("precision:{}".format(sum(precision_test) / len(precision_test)))
    # print("hausdorff:{}".format(sum(hausdorff_test) / len(hausdorff_test)))
    print("dice:{}".format(sum(dice_test) / len(dice_test)))
    print("hausdorff95:{}".format(sum(hausdorff95_test) / len(hausdorff95_test)))
    print("asd:{}".format(sum(asd_test) / len(asd_test)))



if __name__ == '__main__':

    ######   加载训练集   ####

    # train_f = open("D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/train_path_list.txt", "r",
    #                encoding='utf-8')
    # train_lines = train_f.readlines()

    train_list = [56,109,16,104,123,126,128,2,16,81,82,4,7,17,81,13,85,92]

    #
    #
    # #######  加载验证集   #####
    # val_f = open("D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/val_path_list.txt", "r", encoding='utf-8')
    # val_lines = val_f.readlines()
    #
    # val_list = []
    # for line in val_lines:
    #     a = line.split('.')[1].split('-')[1]
    #     val_list.append(a)



    resultCompute(trainlist=train_list)



