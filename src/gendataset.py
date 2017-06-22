#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import imshow, show

import argparse
import os
import numpy as np
from phd_dapatinoco.phd_tools.datasets import MPEG7Dataset
from skimage.io import imsave
from skimage.morphology import medial_axis
from skimage.transform import rotate, ProjectiveTransform, warp
from glob import glob
from os.path import join
from phd_dapatinoco.phd_tools.data_augmentation import apply_transform



parser = argparse.ArgumentParser(description='Create an augmented dataset out of the MPEG7 shape database')
parser.add_argument('-s', '--source', help='Source folder', required=True)
parser.add_argument('-o', '--output', help='Output folder', required=True)
parser.add_argument('-d', '--generate-distance', help='Indicates whether the code will generate the medial axis image', required=False, default=False)
parser.add_argument('-p', '--training-rate', help='Indicates the rato of data that will be used as training samples', required=False, default=0.6)
args = vars(parser.parse_args())

source_dir = args['source']
output_dir = args['output']
generateDistance = args['generate_distance']
test_rate = 1 - float(args['training_rate'])

dataset = MPEG7Dataset(args['source'])

# Structure with all the transformation for Data Augmentation
transformations = {
    'type': 'rotation',
    'angles': ['identity', 'random', 'random', 'random', 'h_reflection'],
    'subtransformation': {
        'type': 'projective',
        'relations': ['identity', 'random', 'random', 'random']
    }
}

transformations = [
    ['identity'],
    ['rotation'],
    ['rotation'],
    ['rotation'],
    ['rotation'],
    ['rotation'],
    ['rotation'],
    ['rotation'],
    ['rotation'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
    ['rotation', 'projective'],
]

# Generate the
for i in range(dataset.num_images):
    image_name = dataset.image_names[i]
    idx = image_name.find('-')
    print 'Processing image:', image_name

    image_dir = os.path.join(output_dir, 'train', image_name[:idx])
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image = dataset.get_image_data(i)

    transformed_images = apply_transform(image, dataset.image_names[i], transformations)

    for t_image, t_image_name in transformed_images:
        if generateDistance:
            _, distance = medial_axis(t_image, return_distance=True)
            final_image = (255 * distance / distance.max()).astype('uint8')
        else:
            final_image = t_image

        #imsave(os.path.join(image_dir, dataset.image_names[i] + '.jpg'), final_image)
        imsave(os.path.join(image_dir, t_image_name + '.jpg'), final_image)

# Move some data to the valid folder according to training_rate
all_classes = glob(os.path.join(output_dir, 'train/*'))
for training_class_path in all_classes:

    if not os.path.exists(join(output_dir, 'valid')):
        os.makedirs(join(output_dir, 'valid'))

    all_images_in_class = glob(training_class_path + '/*')
    n = len(all_images_in_class)
    idx = np.random.permutation(n)[:int(np.floor(n*test_rate))]

    # Move images to the valid/test folder
    class_name = os.path.dirname(training_class_path)
    valid_class_path = training_class_path.replace('/train', '/valid')

    if not os.path.exists(valid_class_path):
        os.makedirs(valid_class_path)
    for i in idx:
        os.rename(all_images_in_class[i], all_images_in_class[i].replace('/train', '/valid'))

