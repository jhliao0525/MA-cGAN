f = open("experiments/1121_apa_ud/test_proc_res.txt","r",encoding='utf-8')
# train_f=open("D:/ljhProjects/3DUNet429/3DUNet-Pytorch-master/fixed_data/train_path_list.txt","r",encoding='utf-8')
train_lines=f.readlines()


# train_list=[]
n_train=0
s=0
for line in train_lines:
    a=line.split(',')[1]
    s+=float(a)
    n_train+=1
print(s/n_train)
