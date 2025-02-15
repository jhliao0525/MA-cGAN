import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[0,1], help='use cpu only')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes') # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）
parser.add_argument('--upper', type=int, default=200, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--expand_slice', type=int, default=20, help='')
parser.add_argument('--min_slices', type=int, default=48, help='')
parser.add_argument('--xy_down_scale', type=float, default=1, help='')
parser.add_argument('--slice_down_scale', type=float, default=1, help='')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')

# data in/out and dataset
#parser.add_argument('--dataset_path',default = '/ssd/lzq/dataset/fixed_lits',help='fixed trainset root path')
parser.add_argument('--dataset_path',default = './fixed_data',help='fixed trainset root path')
#parser.add_argument('--test_data_path',default = '/ssd/lzq/dataset/LiTS/test',help='Testset path')
parser.add_argument('--test_data_path',default = 'fixed_data',help='Testset path')
parser.add_argument('--save',default='1216_unet_tumor',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=1,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=800, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=150, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=128,help='default=48')
parser.add_argument('--val_crop_max_size', type=int, default=128,help='default=48')

# test
parser.add_argument('--test_cut_size', type=int, default=128, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=12, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=False, help='post process')
parser.add_argument('--resume', type=bool, default=False, help='resume train')
parser.add_argument('--pretrain', type=bool, default=False, help='pre train')
parser.add_argument('--test_model', type=str, default="bestdice", help='bestdice or minloss')


args = parser.parse_args()


