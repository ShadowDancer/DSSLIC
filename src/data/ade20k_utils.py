""" This script transforms dataset from RGB segmentations to grayscale segmentations,
where each pixel is number of segmentation and can be converted to one_hot representation """

from glob import glob
from PIL import Image
import os
import numpy as np

dataset_path = '/media/data2/inpainting/databases/ADE20K_2016_07_26/images/training/**/'
image_paths = glob(dataset_path + 'ADE_train_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_seg.png', recursive=True)

print('Segmentations found:', len(image_paths))

orig_images = []
result_images = []

total = len(image_paths)
for idx, img_path in enumerate(image_paths):
    image = np.array(Image.open(img_path))

    encoded = image[:, :, 0] + image[:, :, 1] * 256 + image[:, :, 2] * 256 * 256
    u_vals, u_counts = np.unique(encoded, return_counts=True)
    u = list(filter(lambda x: x[0] != 0, zip(u_vals, u_counts)))
    u.sort(key=lambda x: x[1], reverse=True)

    for i, (v, _) in enumerate(u):
        np.place(encoded, encoded == v, i+1)

    gs_image = Image.fromarray(encoded.astype(np.uint8))

    dire = os.path.dirname(img_path)
    file, ext = os.path.splitext(os.path.basename(img_path))

    new_path = os.path.join(dire, file + "_one_hot" + ext)
    gs_image.save(new_path)
    print(idx, '/', total, new_path)

