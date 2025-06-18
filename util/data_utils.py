"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, datamax=None):
    images = []
    images_rel = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    i = 0
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                path_rel = os.path.relpath(path, start=dir)
                images.append(path)
                images_rel.append(path_rel)
                i += 1
                if datamax is not None:
                    if i > datamax:
                        break
    return sorted(images), sorted(images_rel)

