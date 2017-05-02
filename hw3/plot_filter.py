#!/usr/bin/env python
# -- coding: utf-8 --

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import sys


def normalize(x_inp):
    return x_inp / (K.sqrt(K.mean(K.square(x_inp))) + 1e-7)

def grad_ascent(num_step, input_image_data, iter_func):
    filter_images = []
    lr = 1
    losses = 0
    for _ in range(num_step):
        loss_value, grads_value = iter_func([input_image_data, 1])
        input_image_data += grads_value * lr
        #losses += loss_value
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    filter_images.append(input_image_data)
    filter_images.append(loss_value)
    return filter_images

def main():
    emotion_classifier = load_model(sys.argv[1])

    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[:])
    '''print('\n\n')
    emotion_classifier.summary()
    print('\n\n')'''
    input_img = emotion_classifier.input

    name_ls = ['activation_1','conv2d_1', 'batch_normalization_1', 'max_pooling2d_1', 'conv2d_2']
    collect_layers = [layer_dict[name].output for name in name_ls]
    num_filter = int(collect_layers[0].shape[3])
    num_step = 20

    for cnt, layer in enumerate(collect_layers):
        filter_imgs = [[] for i in range(num_filter)]
        for filter_idx in range(num_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(layer[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])

            filter_imgs[filter_idx] = grad_ascent(num_step, input_img_data, iterate)

        fig = plt.figure(figsize=(14, 10))
        for i in range(num_filter):
            axis = fig.add_subplot(num_filter/8, 8, i+1)
            axis.imshow(filter_imgs[i][0].reshape(48, 48), cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel("{:.3f}".format(filter_imgs[i][1]))
            plt.tight_layout()

        fig.suptitle("Filters of layer ({}) (# Ascent Epoch {} )".format(name_ls[cnt], num_step))
        fig.savefig('./figure/filter/{}.png'.format(name_ls[cnt]))

    K.clear_session()

if __name__ == "__main__":
    main()
