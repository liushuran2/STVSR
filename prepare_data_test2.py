import argparse
import h5py
import numpy as np
import PIL.Image as pil_image
from skimage import io



#root=args.dataset
root = 'x2/Lifeact_all/test_Aug'
output = 'Lifeact_test_step2_VSR_Aug2.h5' 
#output=args.output
#trainlist=np.genfromtxt(root+'seq_trainlist.txt',dtype='str')
#pre_hr=root+'sequences/'


h5_file = h5py.File(output, 'w')

lr_group = h5_file.create_group('lr')
hr_group = h5_file.create_group('hr')

patch_idx = 0
n_frame = 7
valid = [1,3,5]
test = [2,4,6]
patchsize = 178
factor = 3
for i in range(1,41):
    for seq_num in range(1):
        hr=np.zeros((n_frame,1,patchsize*factor,patchsize*factor)).astype(np.float64)
        lr=np.zeros((n_frame,1,patchsize,patchsize)).astype(np.float64)
        for k in range(n_frame):
            if k < 10:
                hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(k+seq_num+1).rjust(4, '0') +'.tif')
                lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_step2'+'/' + str(k+seq_num+1).rjust(3, '0') +'.tif')
            else:
                hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(6).rjust(4, '0') +'.tif')
                lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_step4'+'/' + str(6).rjust(3, '0') +'.tif')
            # hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(k+seq_num+1).rjust(4, '0') +'.tif')
            # lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr_step2'+'/' + str(k+seq_num+1).rjust(3, '0') +'.tif')
            hr_temp = np.array(hr_temp, dtype=np.float32)/65535
            lr_temp = np.array(lr_temp, dtype=np.float32)/65535
            hr_temp = np.maximum(hr_temp, 0)
            # lr_temp = (lr_temp - np.min(lr_temp)) / (np.max(lr_temp) - np.min(lr_temp))
            #hr_temp = (hr_temp - np.min(hr_temp)) / (np.max(hr_temp) - np.min(hr_temp))
            hr_temp = hr_temp[:, :, np.newaxis]
            lr_temp = lr_temp[:, :, np.newaxis]
            hr[k,:,:,:]=np.asarray(hr_temp).astype(np.float32).transpose(2,0,1)
            lr[k,:,:,:]=np.asarray(lr_temp).astype(np.float32).transpose(2,0,1)

        # lr = (lr - np.min(lr)) / (np.max(lr) - np.min(lr))
        # hr = (hr - np.min(hr)) / (np.max(hr) - np.min(hr))
        # lr = lr[3:4,:,:,:]
        # hr = hr[3:4,:,:,:]
        # print(np.max(lr))
        # print(np.max(hr))
        lr_group.create_dataset(str(patch_idx), data=lr)
        hr_group.create_dataset(str(patch_idx), data=hr)
        
        patch_idx += 1
        print(patch_idx)

h5_file.close()