from resultCompare import get_dices
import SimpleITK as sitk


def test_dice_compare(segStr, gtStr):
    seg = sitk.ReadImage(segStr, sitk.sitkUInt8)
    gt = sitk.ReadImage(gtStr, sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    gt_array = sitk.GetArrayFromImage(gt)

    return get_dices(seg_array, gt_array)


f = open("experiments/1121_apa_128/1.txt","r",encoding='utf-8')



######   加载训练集   ####

train_f=open("D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/train_path_list.txt","r",encoding='utf-8')
train_lines=train_f.readlines()


train_list=[]

for line in train_lines:
    a=line.split('.')[1].split('-')[1]
    train_list.append(a)


#######  加载验证集   #####

val_f=open("D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/val_path_list.txt","r",encoding='utf-8')
val_lines=val_f.readlines()

val_list=[]
for line in val_lines:
    a=line.split('.')[1].split('-')[1]
    val_list.append(a)



# lines = f.readlines()      #读取全部内容 ，并以列表方式返回
# s_all=0
# s_test=0
# s_train=0
# s_val=0


# n_all=0
# n_test=0
# n_train=0
# n_val=0
#
# test_dice_sum=0
# test_num=0
# for line in lines:
#     l=line.split('.')[0]
#     n=line.split(',')[1]
#     gtStr = "D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/label/segmentation-" + str(l) + ".nii.gz"
#     segStr = 'D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/experiments/1121_apa_128/result_proc/result-' + str(
#         l) + '-proc.nii.gz'
#
#     test_dice = test_dice_compare(segStr, gtStr)
#     test_dice_sum += test_dice
#     test_num += 1
#     print("{}.nii,{}".format(l, float(test_dice)))
    # if l in train_list:
    #     s_train+=float(n)
    #     n_train+=1
    #
    # elif l in val_list:
    #     s_val+=float(n)
    #     n_val+=1
    # else:
    #     s_test+=float(n)
    #     n_test+=1
    #
    #     gtStr="D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/label/segmentation-"+str(l)+".nii.gz"
    #     segStr='D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/experiments/1116_apa/result_proc/result-' +str(l) + '-proc.nii.gz'
    #
    #     test_dice=test_dice_compare(segStr,gtStr)
    #     test_dice_sum+=test_dice
    #     test_num+=1
    #     print("{}:{}".format(l,test_dice))


#
#     s_all+=float(n)
#     n_all+=1
# #
# print("test_dice_ave:{}".format(test_dice_sum/test_num))
#
#
# acc_all=s_all/n_all
# acc_test=s_test/n_test
# acc_train=s_train/n_train
#
# print("总个数：{0},总准确率：{1}".format(n_all,acc_all))
# print("训练集个数：{0},训练集准确率：{1}".format(n_train,acc_train))
# print("测试集个数：{0},测试集准确率：{1}".format(n_test,acc_test))


def procResult(resTxt,train_list=train_list,val_list=val_list):
    resTxt = open(resTxt, "r",
                   encoding='utf-8')
    lines = resTxt.readlines()  # 读取全部内容 ，并以列表方式返回
    # print(lines)
    train_sum=train_num=val_sum=val_num=test_sum=test_num=all_sum=all_num=0

    for line in lines:
        # print(line)
        # print(line)
        l = line.split('.')[0]
        n = line.split(',')[1]
        if l in train_list:
            train_sum += float(n)
            train_num += 1

        elif l in val_list:
            val_sum += float(n)
            val_num += 1
        else:
            print("{0}:{1}".format(l,n))
            test_sum += float(n)
            test_num += 1
        all_sum+=float(n)
        all_num+=1

    acc_all = all_sum / all_num
    print("总个数：{0},总准确率：{1}".format(all_num, acc_all))


    acc_val=val_sum/val_num
    print("验证集个数：{0},验证集准确率：{1}".format(val_num, acc_val))

    acc_train = train_sum / train_num

    print("训练集个数：{0},训练集准确率：{1}".format(train_num, acc_train))

    acc_test = test_sum / test_num
    print("测试集个数：{0},测试集准确率：{1}".format(test_num, acc_test))

            # gtStr = "D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/label/segmentation-" + str(l) + ".nii.gz"
            # segStr = 'D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/experiments/1116_apa/result_proc/result-' + str(l) + '-proc.nii.gz'
            #
            # test_dice = test_dice_compare(segStr, gtStr)
            # test_dice_sum += test_dice
            # test_num += 1
            # print("{}:{}".format(l, test_dice))


if __name__=='__main__':

    resTxt="experiments/23_6_8_paperAddGanTumor/bestdice_result.txt"
    procResult(resTxt,train_list=train_list,val_list=val_list)