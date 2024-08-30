f = open("experiments/1113_all128_cvit/1.txt","r",encoding='utf-8')
# train_f=open("D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/train_path_list.txt","r",encoding='utf-8')
train_lines=f.readlines()


vloume_id=[]
vloume_score=[]
n_train=0
for line in train_lines:
    a=line.split('.')[0]
    b=line.split(',')[1]
    vloume_id.append(a)
    vloume_score.append(b)

for i in range(30):
    max_num=max(vloume_score)
    index=vloume_score.index(max_num)
    vloumeid=vloume_id[index]
    print("./fixed_data/ct/volume-{0}.nii ./fixed_data/label/segmentation-{1}.nii".format(vloumeid,vloumeid))
    # print("{0}--- id:{1},score:{2}".format(i,vloumeid,max_num))
    del vloume_id[index]
    del vloume_score[index]
    if i==9:
        print("val_list:")

print("train_list")

for i in vloume_id:
    print("./fixed_data/ct/volume-{0}.nii ./fixed_data/label/segmentation-{1}.nii".format(i, i))