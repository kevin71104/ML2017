################################################################################
#                            Machine Learning 2017                             #
#                      Hw3 : Image Sentiment Classification                    #
#            Description : analyze output data at specific layer               #
#                          script : python3 model.h5                           #
################################################################################

#!/usr/bin/env python
# -- coding: utf-8 --

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import sys
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')
with open('train.csv','r') as csvFile:
    train = pd.read_csv(csvFile)
train_feature = train['feature']
train_label   = np.array(train['label'])
train = []

x = []
for i in range(train_feature.shape[0]):
    x.append(train_feature[i].split(' '))
train_feature = np.array(x, dtype=float)
x =[]

train_feature = train_feature.reshape(train_feature.shape[0],48,48,1)

def main():
    emotion_classifier = load_model(sys.argv[1])
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])

    input_img = emotion_classifier.input
    name_ls = ['activation_1','conv2d_1', 'batch_normalization_1', 'max_pooling2d_1', 'conv2d_2']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    choose_id = [9, 74, 1111, 5010]
    for idx in choose_id:
        photo = (train_feature[idx]/255).reshape(1,48,48,1)
        for cnt, fn in enumerate(collect_layers):
            im = fn([photo, 0]) #get the output of that layer
            fig = plt.figure(figsize=(14, 8))
            nb_filter = im[0].shape[3]
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/8, 8, i+1)
                ax.imshow(im[0][0, :, :, i], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            fig.suptitle('Output of layer({}) (Given image {})'.format(name_ls[cnt], idx))
            fig.savefig('./figure/filter/{}({}).png'.format(name_ls[cnt], idx))

if __name__ == "__main__":
    main()
