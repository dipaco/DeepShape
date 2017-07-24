from __future__ import division, print_function
import matplotlib
matplotlib.use('TkAgg')
#%matplotlib inline
path = '../data/db/'
model_path = '../data/db/models/'
import importlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import os, json
from glob import glob
import numpy as np
import time
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_data, plot_confusion_matrix

''' Use a pretrained VGG model with our Vgg16 class '''

# As large as you can, but no larger than 64 is recommended.
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size = 16
# Import our class, and instantiate
import vgg16;
reload(vgg16)
from vgg16 import Vgg16

model_pklot = Vgg16()

batches = model_pklot.get_batches(path+'train', batch_size=batch_size)
val_batches = model_pklot.get_batches(path+'valid', batch_size=batch_size)
imgs, labels = next(batches)
#plots(imgs, titles=labels)
#model_pklot.predict(imgs, True)

'''Use our Vgg16 class to finetune the MPEG7 database model'''

start = time.time()
model_pklot.finetune(batches)
model_pklot.fit(batches, val_batches, nb_epoch=3)
end = time.time()
print((end - start)/60)

model_pklot.model.save_weights(model_path+'vgg_pklot.h5')

model_pklot.model.load_weights(model_path+'vgg_pklot.h5')

imgs,labels = next(val_batches)
plots(imgs, titles=labels)
#model_pklot.classes[:2] = ['Empty', 'Occupied']
model_pklot.classes[:70] = [str(i) for i in range(70)]
model_pklot.predict(imgs, True)


'''model_pklot.model.load_weights(model_path+'vgg_pklotT.h5')
path_val = path + 'validsunny'
val_data = get_data(path_val)
val_batches = model_pklot.get_batches(path_val)
start = time.time()
val = model_pklot.predict(val_data, True)
end = time.time()
print((end - start)/60)
val_classes = val_batches.classes
cm = confusion_matrix(val_classes, val[1])
cm_dict = {}
for i in range(70):
    cm_dict[str(i)] = i
plot_confusion_matrix(cm, {'Empty':0, 'Occupied':1})'''
