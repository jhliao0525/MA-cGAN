f = open("fixed_data/train_path_list.txt","r")
lines = f.readlines()
f.close()
f = open("fixed_data/train_path_list.txt","w")
#lines = f.readlines()
i=0
for line in lines:
    if "-3.nii" not in line and "-4.nii" not in line:
        f.write(line)




f.close()