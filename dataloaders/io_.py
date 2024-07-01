import os
import shutil
import time
from pathlib import Path

import numpy as np
from torch import multiprocessing
import nibabel as nib

from scipy.ndimage import gaussian_filter
import skimage.measure as skmeasure
import scipy.ndimage as ndi
import torch

"""
IO tools
"""


def read_mhd_affine(path):
    with Path(path).open('r') as f:
        lines = f.readlines()
        matrix_str = [float(s) for s in lines[5].strip().split('=')[-1].strip().split(' ')]
        offset = [float(s) for s in lines[6].strip().split('=')[-1].strip().split(' ')]
        matrix = np.array(matrix_str).reshape(3, 3)
        offset = np.array(offset).reshape(3, 1)
        res = np.concatenate([np.concatenate([matrix, offset], 1), np.array((0, 0, 0, 1)).reshape(1, 4)], 0)
    return res


def make_affine(sitkImg):
    # get affine transform in LPS
    rot = [sitkImg.TransformContinuousIndexToPhysicalPoint(p)
           for p in ((1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1),
                     (0, 0, 0))]
    rot = np.array(rot)
    affine = np.concatenate([
        np.concatenate([rot[0:3] - rot[3:], rot[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine


def make_affine2(spacing):
    affine = np.array(((0, 0, -1, 0),
                       (0, -1, 0, 0),
                       (-1, 0, 0, 0),
                       (0, 0, 0, 1)))
    spacing = np.diag(list(spacing) + [1])
    return np.matmul(affine, spacing)


def read_nii(path, method='nib'):
    """
    Read ".nii.gz" data
    :param path: path to image
    :param method: method to read data, only support ('nib', 'sitk')
    :returns:
      data    : numpy data [channel, x, y]
      spacing : (x_spacing, y_spacing, z_spacing)

    """
    import SimpleITK as sitk
    path = str(path)
    method = method.lower()
    if method == 'nib':
        from nibabel.filebasedimages import ImageFileError
        try:
            img = nib.load(path)
            data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
            affine = img.affine
            return data, spacing, affine
        except ImageFileError as e:
            method = 'sitk'

    if method == 'sitk':
        img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(img)
        # channel first
        spacing = img.GetSpacing()[::-1]
        affine = make_affine2(spacing)
        return data, spacing, affine
    else:
        raise Exception("method only supports nib(nibabel) or sitk(SimpleITK)")


def read_img(img_path):
    img_path = str(img_path)

    import skimage.io as skio
    if img_path.endswith('jpg') or img_path.endswith('png') or img_path.endswith('bmp'):
        return skio.imread(img_path)
    elif img_path.endswith('nii.gz') or img_path.endswith('.dcm'):
        return read_nii(img_path)[0]
    elif img_path.endswith('npy'):
        return np.load(img_path)
    elif img_path.endswith('.mhd'):
        return read_nii(img_path, method='sitk')[0]
    else:
        raise Exception("Error file format for {}, only support ['bmp', 'jpg', 'png', 'nii.gz', 'npy', 'mhd']".format(img_path))


def save_nii_with_sitk(np_data, path, origin=None, spacing=None):
    img = sitk.GetImageFromArray(np_data)
    if origin is not None:
        img.setOrigin(origin)
    if spacing is not None:
        img.setSpacing(spacing)
    sitk.WriteImage(img, path)


def save_nii(np_data, affine, path):
    path = str(path)
    img = nib.Nifti1Image(np_data, affine)
    nib.save(img, path)


def mkdir(path, level=2, create_self=True):
    """ Make directory for this path,
    level is how many parent folders should be created.
    create_self is whether create path(if it is a file, it should not be created)

    e.g. : mkdir('/home/parent1/parent2/folder', level=3, create_self=False),
    it will first create parent1, then parent2, then folder.

    :param path: string
    :param level: int
    :param create_self: True or False
    :return:
    """
    p = Path(path)
    if create_self:
        paths = [p]
    else:
        paths = []
    level -= 1
    while level != 0:
        p = p.parent
        paths.append(p)
        level -= 1

    for p in paths[::-1]:
        p.mkdir(exist_ok=True)


def move_files(path):
    path = Path(path)
    pathes = list(path.iterdir())
    for p in pathes:
        parent = p.parent
        name = p.name
        _, cid, number, suffix = name.split('_')
        case_dir = parent / cid
        case_dir.mkdir(exist_ok=True)
        target_p = case_dir / "{}_{}".format(number, suffix)
        shutil.move(str(p), str(target_p))
    print('finished')


def create_symlink(src, dst):
    try:
        # if exists, unlink first
        os.unlink(dst)
    except FileNotFoundError as e:
        pass

    try:
        # create link
        os.symlink(src, dst)
    except FileExistsError as e:
        pass


"""
Case file and folder tools
"""


def load_case(root, cid):
    img_p = get_nii_case_file(root, cid, 'imaging')
    mask_p = get_nii_case_file(root, cid, 'mask')
    img, spacing, affine = read_nii(img_p)
    mask, _, _ = read_nii(mask_p)
    return img, mask, spacing, affine


def get_case_folder(root_path, name, cid, create_self=True):
    """ Make case folder
    :param root_path:
    :param cid:
    :return:
    """
    img_folder = Path(root_path) / name / 'case_{:05d}'.format(cid)
    mkdir(img_folder, level=3, create_self=create_self)
    return img_folder


def get_nii_case_file(root, cid, filename):
    path = Path(root) / ("case_{:05d}".format(cid)) / ("{}.nii.gz".format(str(filename)))
    mkdir(path, level=2, create_self=False)
    return path

"""
To process quickly, use multi-cpu to process functions
"""


def multiprocess_task(func, dynamic_args, static_args=(), split_func=np.array_split, ret=False, cpu_num=None):
    """
    Process task with multi cpus.
    :param func: task to be processed, func must be the top level function
    :param dynamic_args: args to be split to assign to cpus,
              it is a list by default
    :param static_args:  args doesn't need to be split
    :param split_func:   function to split args, use the function
              to split a list args by default
    :return:
    """
    start = time.time()
    if cpu_num is None:
        cpu_num = multiprocessing.cpu_count() // 2

    if cpu_num <= 1:
        ret = func(dynamic_args, *static_args)
    else:
        # split dynamic args with cpu num
        dynamic_args_splits = split_func(dynamic_args, cpu_num)
        workers = multiprocessing.Pool(processes=cpu_num)
        processes = []
        for proc_id, dynamic_args in enumerate(dynamic_args_splits):
            # do processing, concat dynamic args and static args
            dynamic_args = list(dynamic_args)
            p = workers.apply_async(func, (dynamic_args, *static_args))
            processes.append(p)
        workers.close()
        workers.join()

    duration = time.time() - start
    print('total time : {} min'.format(duration / 60.))

    if ret:
        # collect results
        if cpu_num > 1:
            res = []
            for p in processes:
                p = p.get()
                res.extend(p)
        else:
            res = ret
        return res


def io_exception(func):
    def wrapper():
        try:
            func()
        except Exception as e:
            print(e)
    return wrapper
    


class BBoxException(Exception):
    pass


def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 0)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 2)

    return np.array(((min_x, max_x + 1),
                     (min_y, max_y + 1),
                     (min_z, max_z + 1)))


def pad_bbox(bbox, min_bbox, max_img):
    """
    :param bbox:  ndarray ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    :param min_bbox: list (d, h, w)
    :param max_img:  list (d, h,  w), image shape
    :return:
    """
    min_bbox = list(min_bbox)
    change_min_bbox = False
    for i, (min_x, max_img_x) in enumerate(zip(min_bbox, max_img)):
        if min_x > max_img_x:
            min_bbox[i] = max_img[i]
            change_min_bbox = True

    if change_min_bbox:
        print('min box {} is larger than max image size {}'.format(min_bbox, max_img))

    # z first
    bbox = np.array(bbox)[::-1, :]
    result_bbox = []
    for (min_x, max_x), min_size, max_size in zip(bbox, min_bbox, max_img):
        width = max_x - min_x
        if width < min_size:
            padding = min_size - width
            padding_left = padding // 2
            padding_right = padding - padding_left

            # find a best place to pad img
            while True:
                if (min_x - padding_left) < 0 and (max_x + padding_right) > max_size:
                    # pad to img size
                    padding_left = min_x
                    padding_right = max_size - max_x
                    break
                elif (min_x - padding_left) < 0:
                    # right shift pad
                    padding_left -= 1
                    padding_right += 1
                elif (max_x + padding_right) > max_size:
                    # left shift pad
                    padding_left += 1
                    padding_right -= 1
                else:
                    # no operation to pad
                    break
            min_x -= padding_left
            max_x += padding_right
        result_bbox.append((min_x, max_x))
    # x first
    return np.array(result_bbox)[::-1, :]


def expand_bbox(img, bbox, expand_size, min_crop_size):
    img_z, img_y, img_x = img.shape

    # expand [[154 371  15] [439 499  68]]
    bbox[:, 0] -= expand_size[::-1]  # min (x, y, z)
    bbox[:, 1] += expand_size[::-1]  # max (x, y, z)
    # prevent out of range
    bbox[0, :] = np.clip(bbox[0, :], 0, img_x)
    bbox[1, :] = np.clip(bbox[1, :], 0, img_y)
    bbox[2, :] = np.clip(bbox[2, :], 0, img_z)

    # expand, then pad
    bbox = pad_bbox(bbox, min_crop_size, img.shape)
    return bbox



def crop_img(img, bbox, min_crop_size):
    """ Crop image with expanded bbox.
    :param img:  ndarray (D, H, W)
    :param bbox: ndarray ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    :param min_crop_size: list (d, h ,w)
    :return:
    """

    # extract coords
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = bbox

    # crop
    cropped_img = img[min_z:max_z, min_y:max_y, min_x:max_x]

    padding = []
    for i, (cropped_width, min_width) in enumerate(zip(cropped_img.shape, min_crop_size)):
        if cropped_width < min_width:
           padding.append((0, min_width - cropped_width))
        else:
           padding.append((0, 0))
    padding = np.array(padding).astype(np.int)
    cropped_img = np.pad(cropped_img, padding, mode='constant', constant_values=0)
    return cropped_img


from dipy.align.reslice import reslice
def resample_volume_nib(np_data, affine, spacing_old, spacing_new=(1., 1., 1.), mask=False):
    """Resample 3D image(trilinear) and mask(nearest) to (1., 1., 1.) spacing.
       It seems works better than the method above, seen from generated image.

    :param np_data: ndarray, channel first
    :param affine: the affine returned from nibabel
    :param spacing_old:  current spacing
    :param spacing_new: target spacing, default is (1., 1., 1.)
    :param mask: if set True, use nearest instead of trilinear interpolation
    :return:
        resampled data : ndarray
        affine         : the modified affine.
    """
    if not mask:
        # trilinear
        resampled_data, affine = reslice(np_data, affine, spacing_old, spacing_new, order=1)
    else:
        # nearest
        resampled_data, affine = reslice(np_data, affine, spacing_old, spacing_new, order=0)
    return resampled_data, affine


if __name__ == '__main__':
    pass
