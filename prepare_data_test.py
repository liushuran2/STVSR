
import h5py
import numpy as np
from skimage import io

root = 'data/test'
output = 'Lifeact_test_step2_VSR_Aug2.h5' 
patchsize = 512
factor = 3
n_frame = 7
total_count = 40
num_clip = 1

h5_file = h5py.File(output, 'w')

lr_group = h5_file.create_group('lr')
hr_group = h5_file.create_group('hr')

patch_idx = 0
for i in range(1,total_count+1):
    for seq_num in range(num_clip):
        hr=np.zeros((n_frame,1,patchsize*factor,patchsize*factor)).astype(np.float64)
        lr=np.zeros((n_frame,1,patchsize,patchsize)).astype(np.float64)
        for k in range(n_frame):
            hr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/hr'+'/GT' + str(k+seq_num+1).rjust(4, '0') +'.tif')
            lr_temp=io.imread(root+'/' + str(i).rjust(3, '0') +'/lr'+'/' + str(k+seq_num+1).rjust(3, '0') +'.tif')
            hr_temp = np.array(hr_temp, dtype=np.float32)
            lr_temp = np.array(lr_temp, dtype=np.float32)
            hr_temp = np.maximum(hr_temp, 0)
            hr_temp = hr_temp[:, :, np.newaxis]
            lr_temp = lr_temp[:, :, np.newaxis]
            hr[k,:,:,:]=np.asarray(hr_temp).astype(np.float32).transpose(2,0,1)
            lr[k,:,:,:]=np.asarray(lr_temp).astype(np.float32).transpose(2,0,1)

        lr = (lr - np.min(lr)) / (np.max(lr) - np.min(lr))
        hr = (hr - np.min(hr)) / (np.max(hr) - np.min(hr))
        lr_group.create_dataset(str(patch_idx), data=lr)
        hr_group.create_dataset(str(patch_idx), data=hr)
        
        patch_idx += 1
        print(patch_idx)

h5_file.close()