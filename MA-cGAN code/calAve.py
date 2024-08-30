train_f=open("experiments/1121_apa_128/val_test.txt","r",encoding='utf-8')
train_lines=train_f.readlines()

n_train=0
n=0
for line in train_lines:
    a=line.split(',')[1]
    a=float(a)
    n_train+=a
    n+=1

print(n_train/n)