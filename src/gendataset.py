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


def apply_transform(image, image_name, transformation):
    """
    Recursively apply a set of transformations to an image for data augmentation
    Args:
        image:
        image_name:
        transformation:

    Returns:

    """
    transformed_images = []

    # if the current transformation is a rotation
    if transformation['type'] == 'rotation':
        for angle in transformation['angles']:

            # Calculates the rotation angle
            if angle == 'identity':
                angle = 0
            elif angle == 'random':
                angle = int(180 * np.random.rand())
            elif angle == 'h_reflection':
                image = image[:, ::-1]
                angle = 0

            t_image = rotate(image, angle, resize=True)
            print '\tApplying transformation: rotation ({0}º) to {1}'.format(angle, image_name)

            if 'subtransformation' in transformation:
                transformed_images += apply_transform(t_image, image_name + '_rotation_' + str(angle), transformation['subtransformation'])
            else:
                transformed_images.append((t_image, image_name + '_rotation_' + str(angle)))

    elif transformation['type'] == 'projective':
        i = 1
        for relation in transformation['relations']:

            # Calculates the rotation angle
            matrix = np.eye(3, 3)
            output_shape = (image.shape[0], image.shape[1])
            if relation == 'random':
                matrix[2, 0] = np.random.randint(-80, 80) / 100000.0
                matrix[2, 1] = np.random.randint(-80, 80) / 100000.0
                output_shape = (int(1.5 * image.shape[0]), int(1.5 * image.shape[1]))

            print '\tApplying projective transformation to {0}'.format(image_name)

            tf = ProjectiveTransform(matrix=matrix)
            t_image = warp(image, tf, output_shape=output_shape)
            if 'subtransformation' in transformation:
                transformed_images += apply_transform(t_image, image_name + '_projective_' + str(matrix[2, 0]) + '-' + str(matrix[2, 1]), transformation['subtransformation'])
            else:
                transformed_images.append((t_image, image_name + '_projective_' + str(matrix[2, 0]) + '-' + str(matrix[2, 1])))

            i += 1

    return transformed_images

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

