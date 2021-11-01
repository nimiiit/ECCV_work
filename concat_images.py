import os
import shutil
import cv2
import numpy as np

src='/home/ipcv101/Praveen/KT/KT_Project/KT_LUA/Data/Test_patches'

dest='/home/ipcv101/Praveen/KT/KT_Project/KT_LUA/Data_concatenated/10'

src_1=os.path.join(src,'10')
src_2=os.path.join(src,'10')
src_in_1=os.listdir(os.path.join(src,'10'))
src_in_2=os.listdir(os.path.join(src,'10'))
i=0

for file_name in src_in_1:
	
	img_1=os.path.join(src_1,file_name)
	img_2=os.path.join(src_2,file_name)

	dest_name=os.path.join(dest,str(i)+".jpg")
	
	img_1=cv2.imread(img_1)
	img_1=cv2.resize(img_1,(128,128))
	img_2=cv2.imread(img_2)

	img_12 = np.concatenate([img_1, img_2],1)
	cv2.imwrite(dest_name,img_12)
	i+=1
