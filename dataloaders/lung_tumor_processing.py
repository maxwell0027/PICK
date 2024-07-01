import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os
import io_
from io_ import *
import nibabel as nib

output_size =[96, 96, 96]
img_root = '/data/userdisk1/qjzeng/semi_seg/code/data_lung/img/'
lab_root = '/data/userdisk1/qjzeng/semi_seg/code/data_lung/lab/'
save_reimg_root = '/data/userdisk1/qjzeng/semi_seg/code/data_lung/img_respacing/'
save_relab_root = '/data/userdisk1/qjzeng/semi_seg/code/data_lung/lab_respacing/'

save_h5_root = '/data/userdisk1/qjzeng/semi_seg/code/data_lung/lung_h5/'

img_path = os.listdir(img_root)
lab_path = os.listdir(lab_root)

img_path.sort()
lab_path.sort()

def normalize(data):
    # normalized_data = (data - data.mean()) / (data.std() + 1e-10)
    normalized_data = (data - data.min()) / (data.max() - data.min())
    normalized_data = normalized_data  # * 2 - 1
    return normalized_data
    
    


def save_to_h5(img, mask, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('image', data=img)
    hf.create_dataset('label', data=mask)
    hf.close()
    
    
    
    

for i in range(len(img_path)):
    img, spacing, affine_pre = io_.read_nii(img_root + img_path[i])
    mask, _, _ = io_.read_nii(lab_root + lab_path[i])
    
    assert mask.shape == img.shape, "{}, {}".format(mask.shape, img.shape)
    
    target_spacing = (1, 1, 1)
    spacing = (spacing[0], spacing[1], spacing[2])
    affine_pre = io_.make_affine2(spacing)
    resampled_img, affine = resample_volume_nib(img, affine_pre, spacing, target_spacing, mask=False)
    resampled_mask, affine = resample_volume_nib(mask, affine_pre, spacing, target_spacing, mask=True)


    min_clip, max_clip = -500, 275
    resampled_img = resampled_img.clip(min_clip, max_clip)
    resampled_img = normalize(resampled_img)
 
    
    bbox = get_bbox_3d(resampled_mask)
    offset = 25
    bbox = expand_bbox(resampled_img, bbox, expand_size=(offset, offset, offset), min_crop_size=(96, 96, 96))
    cropped_img = crop_img(resampled_img, bbox, min_crop_size=(96, 96, 96))
    cropped_mask = crop_img(resampled_mask, bbox, min_crop_size=(96, 96, 96))
    
    '''
    print(cropped_img.shape)

    nib.save(nib.Nifti1Image(cropped_img[:].astype(np.float32), np.eye(4)), save_reimg_root + 're_cl_' + img_path[i])
    nib.save(nib.Nifti1Image(cropped_mask[:].astype(np.float32), np.eye(4)), save_relab_root + 're_cl_' + lab_path[i])
    print(img_path[i] + ' finished!')
    '''

    save_to_h5(cropped_img, cropped_mask, save_h5_root + img_path[i][:8] + '.h5')
    print('saved : {}, resampled shape : {}, cropped shape : {}'.format(img_path[i][:8], resampled_img.shape, cropped_img.shape))









