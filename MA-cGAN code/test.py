from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger,common
from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d
from models.resUnet_apa_256 import resUnet_apa_256
from models.unet3d_ori import UNet3D
import SimpleITK as sitk
import os
import numpy as np

from resultAve import procResult

import Unter

# from models.resUnet4 import resUnet4
# from models import ResUNet,UNet
# from models.resUnet_Gai import resUnet_Gai
# from models.ResUNet_ori import ResUNet_ori
from utils.metrics import DiceAverage
from collections import OrderedDict
from models.unetr_256_add import UNETR_256add
from models.resUnet5 import resUnet5
from models.resUnet6 import resUnet6
#from models.resUnet_Gai2 import resUnet_Gai2
# from models.resUnet_Gai3 import resUnet_Gai3
# from unet3d_ori import UNet3D
from models.unet3d_ori import UNet3D
from models.resUnet7 import resUnet_claw
from models.resUnet_apaatt import resUnet_apa
from models.resunet_apa_UD import resUnet_apa_ud
from models.resunet_apa_UD_1 import resUnet_apa_ud_1
from models.transunet3d import transUNet3D
from models.apauent import APAUNet
from models.unetr_ori import UNETR_ori
from models.trans_bts_ori import transbts
from models.Vnet import VNet



def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)



    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.to(device)
            # print("dataShape{}".format(data.shape))
            output = model(data)
            #print("outputShape{}".format(output.shape))
            output = torch.nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred,dim=1)#tensor

    # ct_array = ndimage.zoom(ct_array,
    #                         (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
    #                         order=3)
    pred_img = common.to_one_hot_3d(pred,args.n_labels)#tensor


    # pred = np.asarray(pred.numpy(), dtype='uint8')
    # pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))
    # sitk.WriteImage(pred, "zzz.nii.gz")




    test_dice.update(pred_img, target)
    
    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels==3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    
    pred = np.asarray(pred.numpy(),dtype='uint8')



    if args.postprocess:
        pass # TO DO

    # label_1 = img_dataset.label
    # target_label = np.asarray(label_1.numpy(), dtype='uint8')
    # target_label = sitk.GetImageFromArray(np.squeeze(target_label, axis=0))

    # sitk.WriteImage(target_label, "sggh.nii.gz")
    #
    # var = 1
    #
    # while var == 1:  # 表达式永远为 True
    #
    #     print("var = 1")
    target = torch.argmax(target, dim=1)
    target = np.asarray(target.data.cpu().numpy(),dtype='uint8')




    pred = sitk.GetImageFromArray(np.squeeze(pred,axis=0))
    target= sitk.GetImageFromArray(np.squeeze(target,axis=0))

    # sitk.WriteImage(target, "target.nii.gz")
    pred.SetDirection(target.GetDirection())
    pred.SetOrigin(target.GetOrigin())
    pred.SetSpacing((target.GetSpacing()[0] * int(1 / 1),
                     target.GetSpacing()[1] * int(1 / 1),target.GetSpacing()[2]))

    # pred.SetDirection( target_label.GetDirection())
    # pred.SetOrigin( target_label.GetOrigin())
    # pred.SetSpacing(( target_label.GetSpacing()[0] * int(1 / 1),
    #                   target_label.GetSpacing()[1] * int(1 / 1), 1))


    return test_dice, pred

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    # model info
    device = torch.device('cpu' if args.cpu else 'cuda')
    #model = UNet(in_channel=1, out_channel=args.n_labels,training=False).to(device)
    #model = ResUNet(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    # model = transbts(in_channel=1, out_channel=2, training=True).to(device)
    # model = resUnet_apa(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    #model = ResUNet_ori(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    #model = resUnet_Gai(in_channel=1, out_channel=args.n_labels, training=False).to(device)
   # model = resUnet_Gai3(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    #model = resUnet4(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU

    # model = Unter.UNETR(in_channels=1, out_channels=2, img_size=(128, 128, 128), feature_size=16,
    #                     hidden_size=768,
    #                     mlp_dim=3072,
    #                     num_heads=12,
    #                     pos_embed='perceptron',
    #                     norm_name='instance',
    #                     conv_block=True,
    #                     res_block=False,
    #                     dropout_rate=0.0).to(device)
    # # model.to(device)
    # model = UNETR_ori(in_channels=1, out_channels=2, img_size=(128, 128, 128), feature_size=16,
    #                   hidden_size=768,
    #                   mlp_dim=3072,
    #                   num_heads=12,
    #                   pos_embed='perceptron',
    #                   norm_name='instance',
    #                   conv_block=True,
    #                   res_block=False,
    #                   dropout_rate=0.0).to(device)
    # model = transbts(in_channel=1, out_channel=2, training=True).to(device)
    # model = resUnet5(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet6(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = UNETR_256add(training=True).to(device)
    # model = resUnet_apa(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = transUNet3D(in_channels=1, num_classes=2).to(device)
    # model = resUnet_apa_ud(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet_apa_ud_1(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet_apa_256(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = UNet3D(in_channels=1, num_classes=2, batch_normal=True, bilinear=True).to(device)
    # model = APAUNet(1, 2, True).to(device)
    # model = VNet(n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=False).to(device)
    model = resUnet_apa(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    # model = resUnet_claw(in_channel=1, out_channel=2, training=True).to(device)
    # model = UNet3D(in_channels=1, num_classes=2, batch_normal=True, bilinear=True).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])#将使用的gpu_id为0
    outTextPath=""
    if args.test_model=="bestdice":
        loadPath='{}/best_model.pth'.format(save_path)
        result_save_path = '{}/bestdice_result'.format(save_path)
        outTextPath='{}/bestdice_result.txt'.format(save_path)
        print("load best dice model......")
    else:
        loadPath = '{}/minloss_model.pth'.format(save_path)
        result_save_path = '{}/minloss_result'.format(save_path)
        outTextPath = '{}/minloss_result.txt'.format(save_path)
        print("load min loss model.......")
    ckpt = torch.load(loadPath)
    model.load_state_dict(ckpt['net'],False)

    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    # result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    outfile=open(outTextPath, "w", encoding='utf-8')

    print(args.test_data_path)
    datasets = Test_Datasets(args.test_data_path,args=args)
    for img_dataset,file_idx in datasets:
        #print("img_dataset.shape:{}".format(img_dataset))

        test_dice,pred_img = predict_one_img(model, img_dataset, args)
       # print("pred_img.shape{}".format(type(pred_img)))
        test_log.update(file_idx, test_dice)

        outfile.write("{},{}\n".format(file_idx,str(test_dice).split(' ')[1].split(')')[0]))
        # print(file_idx)
        # print(str(test_dice).split(',')[1].split(')')[0])

        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.gz'))

    outfile.close()#关闭读写txt文件

procResult(outTextPath)
