import argparse
import h5py
import numpy as np
import PIL.Image as pil_image
from skimage import io



#root=args.dataset
root = 'x2/Lifeact_all/train'
output = 'Lifeact_train_step2_VSR_centernorm.h5'
#output=args.output
#trainlist=np.genfromtxt(root+'seq_trainlist.txt',dtype='str')
#pre_hr=root+'sequences/'


h5_file = h5py.File(output, 'w')

lr_group = h5_file.create_group('lr')
hr_group = h5_file.create_group('hr')

patch_idx = 0
n_frame = 7
for i in range(1,41):
    #if i != 9 and i != 11:

    for seq_num in range(1):
        hr=np.zeros((n_frame,1,1536,1536)).astype(np.float32)
        lr=np.zeros((n_frame,1,512,512)).astype(np.float32)
        for k in range(n_frame):
            if i > 0 or k < 8:
                hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(k+seq_num*7+1).rjust(4, '0') +'.tif')
                lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_step2'+'/' + str(k+seq_num*7+1).rjust(3, '0') +'.tif')
            else:
                hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(6).rjust(4, '0') +'.tif')
                lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_step2'+'/' + str(6).rjust(3, '0') +'.tif')
            # hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr/' + str(k+2*seq_num*n_frame+1) +'.tif')
            # lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_x2_BI'+'/' + str(k+2*seq_num*n_frame+1) +'.tif')
            hr_temp = np.array(hr_temp, dtype=np.float32)
            lr_temp = np.array(lr_temp, dtype=np.float32)
            hr_temp = np.maximum(hr_temp, 0)
            # lr_temp = (lr_temp - np.min(lr_temp)) / (np.max(lr_temp) - np.min(lr_temp))
            #hr_temp = (hr_temp - np.min(hr_temp)) / (np.max(hr_temp) - np.min(hr_temp))
            hr_temp = hr_temp[:, :, np.newaxis]
            lr_temp = lr_temp[:, :, np.newaxis]
            hr[k,:,:,:]=np.asarray(hr_temp).astype(np.float32).transpose(2,0,1)
            lr[k,:,:,:]=np.asarray(lr_temp).astype(np.float32).transpose(2,0,1)
    
        lr = (lr - np.min(lr[3,:,:,:])) / (np.max(lr[3,:,:,:]) - np.min(lr[3,:,:,:]))
        hr = (hr - np.min(hr[3,:,:,:])) / (np.max(hr[3,:,:,:]) - np.min(hr[3,:,:,:]))
        # lr = lr[3:4,:,:,:]
        # hr = hr[3:4,:,:,:]
        # print(np.max(lr))
        # print(np.max(hr))
        lr_group.create_dataset(str(patch_idx), data=lr)
        hr_group.create_dataset(str(patch_idx), data=hr)
        
        patch_idx += 1
        print(patch_idx)

h5_file.close()