import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os
import io_
from io_ import *
import nibabel as nib

img_root = '/data/userdisk1/qjzeng/semi_seg/BCP-main/code/data/img/'

img_path = os.listdir(img_root)

img_path.sort()

train_list = []
test_list = []

for item in img_path[:48]:
    train_list.append(item[:8])

for item in img_path[48:]:
    test_list.append(item[:8])
    
with open('test.txt','w') as f:
    for name in test_list:
        f.write(name)
        f.write('\n')
        
with open('train.txt','w') as f:
    for name in train_list:
        f.write(name)
        f.write('\n')