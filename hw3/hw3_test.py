################################################################################
#                            Machine Learning 2017                             #
#                      Hw3 : Image Sentiment Classification                    #
#                         Convolutional Neural Network                         #
#                   description : use model to get prediction                  #
#           script : python3 hw3_test.py test.csv model.h5 ans.csv             #
################################################################################

import pandas as pd
import numpy as np
import sys
from keras.models import load_model
from keras.utils.vis_utils import plot_model

################################## Read File ###################################
with open(sys.argv[1],'r') as csvFile:
    test = pd.read_csv(csvFile)
test_feature = test['feature']
x = []
for i in range(test_feature.shape[0]):
    x.append(test_feature[i].split(' '))
test_feature = np.array(x, dtype=float)

test_feature = test_feature/255
test_feature = test_feature.reshape(test_feature.shape[0],48,48,1)

model = load_model('model.h5')
model.summary()
#plot_model(model,to_file='{}.png'.format(sys.argv[2][:-3])) # plot model

output = model.predict_classes(test_feature,batch_size=100,verbose=1)

x = 0
with open(sys.argv[2],'w') as csvFile:
    csvFile.write('id,label')
    for i in range(len(output)):
        csvFile.write('\n' + str(x) + ',' + str(output[i]))
        x = x+1
