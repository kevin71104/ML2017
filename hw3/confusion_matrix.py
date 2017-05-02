################################################################################
#                            Machine Learning 2017                             #
#                      Hw3 : Image Sentiment Classification                    #
#           Description : analyze model accuracy on training data              #
#                          script : python3 model.h5                           #
################################################################################
import sys
import numpy as np
import pandas as pd
import random as rand
import itertools
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, validsize,
                          title='Confusion matrix',
                          cmap=plt.cm.coolwarm
                          ):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + '({}) valid size({})'.format(sys.argv[1][:-3], validsize))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

################################## Read File ###################################
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

train_feature = train_feature/255
classNum = 7
############################### validation Data ################################
validNum = 10000
choose = rand.sample(range(0,train_feature.shape[0]-1),validNum)
valid_label = train_label[choose]
valid_feature = train_feature[choose]
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)

############################### confusion_matrix ###############################
model = load_model(sys.argv[1])

np.set_printoptions(precision=2)
predictions = model.predict_classes(valid_feature)
conf_mat = confusion_matrix(valid_label,predictions)

plt.figure()
plot_confusion_matrix(conf_mat,
                      classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"],
                      validsize = validNum )
#plt.show()
fig = plt.gcf()
plt.draw()
fig.savefig('./figure/confusion_matrix({}).png'.format(sys.argv[1][:-3]))
