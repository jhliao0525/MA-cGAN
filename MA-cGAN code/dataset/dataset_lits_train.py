from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize,RandomRotate,RandomResize,Center_Crop
from dataset import augmentation
from multiprocessing.dummy import Pool as ThreadPool

##############################################
###############         apa   #############
def padding(sample, target, input_shape):
    HWD = np.array(input_shape)
    hwd = np.array(target.shape)
    tmp = np.clip((HWD - hwd) / 2, 0, None)
    rh, rw, rd = np.floor(tmp).astype(int)
    lh, lw, ld = np.ceil(tmp).astype(int)

    if sample.ndim == 3:
        sample = np.pad(sample, ((rh, lh), (rw, lw), (rd, ld)), 'constant', constant_values=-3)
        target = np.pad(target, ((rh, lh), (rw, lw), (rd, ld)), 'constant')
    else:
        sample = np.pad(sample, ((0, 0), (rh, lh), (rw, lw), (rd, ld)), 'constant', constant_values=-3)
        target = np.pad(target, ((rh, lh), (rw, lw), (rd, ld)), 'constant')
    return sample, target


def random_crop(sample, target, input_shape):
    H, W, D = input_shape
    h, w, d = target.shape

    x = np.random.randint(0, h - H + 1)
    y = np.random.randint(0, w - W + 1)
    z = np.random.randint(0, d - D + 1)

    if sample.ndim == 3:
        return sample[x:x + H, y:y + W, z:z + D], target[x:x + H, y:y + W, z:z + D]
    else:
        return sample[:, x:x + H, y:y + W, z:z + D], target[x:x + H, y:y + W, z:z + D]


def bounding_crop(sample, target, input_shape):
    H, W, D = input_shape
    source_shape = list(target.shape)
    xyz = list(np.where(target > 0))

    lower_bound = []
    upper_bound = []
    for A, a, b in zip(xyz, source_shape, input_shape):
        lb = max(np.min(A), 0)
        ub = min(np.max(A) - b + 1, a - b + 1)

        if ub <= lb:
            lb = max(np.max(A) - b, 0)
            ub = min(np.min(A), a - b + 1)

        lower_bound.append(lb)
        upper_bound.append(ub)
    x, y, z = np.random.randint(lower_bound, upper_bound)

    if sample.ndim == 3:
        return sample[x:x + H, y:y + W, z:z + D], target[x:x + H, y:y + W, z:z + D]
    else:
        return sample[:, x:x + H, y:y + W, z:z + D], target[x:x + H, y:y + W, z:z + D]


def random_mirror(sample, target, prob=0.5):
    p = np.random.uniform(size=3)
    axis = tuple(np.where(p < prob)[0])
    sample = np.flip(sample, axis)
    target = np.flip(target, axis)
    return sample, target

def totensor(data):
    return torch.from_numpy(np.ascontiguousarray(data))


# def batch_and_aug(index):
#     ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
#     seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)
#
#     ct_array = sitk.GetArrayFromImage(ct)
#     seg_array = sitk.GetArrayFromImage(seg)
#
#     ct_array = ct_array / self.args.norm_factor
#     ct_array = ct_array.astype(np.float32)
#
#     ct_array, seg_array = padding(ct_array, seg_array, (64, 64, 64))
#     if np.random.random() < 0.5:
#         ct_array, seg_array = random_crop(ct_array, seg_array, (64, 64, 64))
#     else:
#         ct_array, seg_array = bounding_crop(ct_array, seg_array, (64, 64, 64))
#
#     ct_array, seg_array = random_mirror(ct_array, seg_array)
#     ct_array = totensor(ct_array).unsqueeze(0)  # 1,d,w,h
#     seg_array = totensor(seg_array).unsqueeze(0)  # 1,d,w,h
#     return ct_array, seg_array.squeeze(0)
######################################################

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                # RandomFlip_LR(prob=0.5),

                # RandomFlip_UD(prob=0.5),
                # RandomRotate(),####
                # Center_Crop(16,896),
                # RandomResize([0.5,1.5],[0.5,1.5],[0.5,1.5])####
            ])

    def __getitem__(self, index):

        # pool = ThreadPool(6)
        # batch = pool.map(batch_and_aug, idxs)
        # pool.close()
        # pool.join()
        # samples, targets = zip(*batch)
        # samples = torch.cat(samples)
        # targets = torch.cat(targets)
        # return samples, targets



        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        # ct_array,seg_array=padding(ct_array, seg_array, (64,64,64))
        # if np.random.random() < 0.5:
        #     ct_array, seg_array = random_crop(ct_array, seg_array, (64, 64, 64))
        # else:
        #     ct_array, seg_array = bounding_crop(ct_array, seg_array, (64, 64, 64))

        # ct_array, seg_array = random_mirror(ct_array, seg_array)
        # ct_array = totensor(ct_array).unsqueeze(0)  # 1,d,w,h
        # seg_array = totensor(seg_array).unsqueeze(0)  # 1,d,w,h


        #################################
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)#1,d,w,h
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)#1,d,w,h

        if self.transforms:
            ct_array,seg_array = self.transforms(ct_array, seg_array)

        ####################################################

        ###################augmentation

        ct_array = ct_array.unsqueeze(0)
        seg_array = seg_array.unsqueeze(0)

        if np.random.random() < 0.15:
            std = np.random.random() * 0.1
            ct_array = augmentation.gaussian_noise(ct_array, std=std)
            # print("Gaussian Noise")

        if np.random.random() < 0.15:
            ct_array = augmentation.brightness_multiply(ct_array, multiply_range=[0.7, 1.3])
            # print("brightness_multiply")
        if np.random.random() < 0.15:
            ct_array = augmentation.gamma(ct_array, gamma_range=[0.7, 1.5])
            # print("gamma")
        if np.random.random() < 0.15:
            ct_array = augmentation.contrast(ct_array, contrast_range=[0.65, 1.5])
            # print("contrast")
        if np.random.random() < 0.3:
            ct_array = augmentation.mirror(ct_array, axis=2)
            seg_array = augmentation.mirror(seg_array, axis=2)
            # print("mirror2")
        if np.random.random() < 0.2:
            ct_array = augmentation.mirror(ct_array, axis=1)
            seg_array = augmentation.mirror(seg_array, axis=1)
            # print("mirror1")
        if np.random.random() < 0.05:
            ct_array = augmentation.mirror(ct_array, axis=1)
            seg_array = augmentation.mirror(seg_array, axis=1)
            # print("mirror3")

        # if np.random.random() < 0.3:
        #     # crop trick for faster augmentation
        #     # crop a sub volume for scaling and rotation
        #     # instead of scaling and rotating the whole image
        #     d = self.args.crop_size
        #     h = w = 128
        #     ct_array, seg_array = augmentation.crop_3d(ct_array, seg_array, [d + 70, h + 70, w + 70],
        #                                                mode='random')
        #     # print("crop_3d")
        #     ct_array, seg_array = augmentation.random_scale_rotate_translate_3d(ct_array, seg_array,
        #                                                                         [0.3, 0.3, 0.3], [30, 30, 30],
        #                                                                         [0, 0, 0])
        #     # print("random_scale_rotate_translate_3d")
        #     ct_array, seg_array = augmentation.crop_3d(ct_array, seg_array, [128, 128, 128],
        #                                                mode='center')
        #     # print("crop_3d2")
        # else:
        #     ct_array, seg_array = augmentation.crop_3d(ct_array, seg_array, [128, 128, 128],
        #                                                mode='random')
        #     print("crop_3d3")
        # seg_array = seg_array.squeeze(0)
        # ct_array = ct_array.squeeze(0)
        # ###########################################################################
        # ct_array, seg_array = augmentation.crop_3d(ct_array, seg_array, [128, 128, 128], mode='random')
        ct_array=ct_array.squeeze(0)
        seg_array=seg_array.squeeze(0)
        #
        # # b=seg_array.squeeze(0)
        #
        #     ##################################################
        #     ########存照片
        # new_ct = sitk.GetImageFromArray(ct_array.squeeze(0))
        #     # new_ct.SetDirection(ct.GetDirection())
        #     # new_ct.SetOrigin(ct.GetOrigin())
        # sitk.WriteImage(new_ct, 'ttttt.nii')
        #
        # new_seg = sitk.GetImageFromArray(seg_array.squeeze(0))
        #     # new_seg.SetDirection(ct.GetDirection())
        #     # new_seg.SetOrigin(ct.GetOrigin())
        # sitk.WriteImage(new_seg, 'sss.nii.gz')
        #
        # var = 1
        #
        # while var == 1:  # 表达式永远为 True
        #
        #     print("var = 1")

            ##################################################


        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())