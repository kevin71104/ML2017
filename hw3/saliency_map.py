from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

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

model = load_model(sys.argv[1])
input_img = model.input
img_ids = [1111]

for idx in img_ids:
    origin = train_feature[idx].reshape(48, 48)
    plt.figure()
    plt.imshow(origin,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('./figure/saliency_map/'+str(idx)+".png", dpi=100)

    val_proba = model.predict((train_feature[idx]/255).reshape(1,48,48,1))
    pred = val_proba.argmax(axis=-1)
    target = K.mean(model.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    '''----------------heatmap processing----------------'''
    heatmap = fn([(train_feature[idx]/255).reshape(1,48,48,1),0]) # get the output gradient v.s. input
    heatmap = heatmap[0]
    #print(heatmap)
    heatmap = np.abs(heatmap)
    heatmap -= heatmap.mean()
    heatmap /= (heatmap.std()+1e-20)
    #heatmap /= np.max(heatmap)
    '''---------------end heatmap processing--------------'''

    thres = 0.2
    see = train_feature[idx].reshape(48, 48)
    loc = heatmap <= thres
    loc = loc.reshape(48,48)
    heatmap = heatmap.reshape(48,48)
    see[loc] = np.mean(see)

    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig("./figure/saliency_map/"+str(idx)+"_heatmap.png",dpi=100)

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('./figure/saliency_map/'+str(idx)+"_mask.png", dpi=100)
