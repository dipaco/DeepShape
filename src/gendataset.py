#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from phd_dapatinoco.phd_tools.datasets import MPEG7Dataset
from skimage.io import imsave

parser = argparse.ArgumentParser(description='Create an augmented dataset out of the MPEG7 shape database')
parser.add_argument('-s', '--source', help='Source folder', required=True)
parser.add_argument('-o', '--output', help='Output folder', required=True)
args = vars(parser.parse_args())

source_dir = args['source']
output_dir = args['output']

dataset = MPEG7Dataset(args['source'])

for i in range(dataset.num_images):
    image_name = dataset.image_names[i]
    idx = image_name.find('-')
    print 'Processing image:', image_name

    image_dir = os.path.join(output_dir, 'train', image_name[:idx])
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    imsave(os.path.join(image_dir, dataset.image_names[i] + '.jpg'), dataset.get_image_data(i))