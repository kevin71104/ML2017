#!/usr/bin/env python
# -- coding: utf-8 --
"""
Predict the operating condition of waterpoints
Training Part
"""

import os
import csv
import pickle
import pandas as pd
from argparse import ArgumentParser
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)

VALIDATION_SPLIT = 0.1
DROPOUT_RATE = 0.5

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def read_dataset(value_path, label_path):
    """ Read and process dataset """
    # Get label
    if label_path:
        label_dict = {'functional': 0, 'non functional': 1,
                      'functional needs repair': 2}
        label = []
        with open(label_path, 'r') as lab_file:
            for row in csv.DictReader(lab_file):
                label.append(label_dict[row['status_group']])

        label = np.array(label)

    # Get the feature
    # Feature listed in here is useless in my opinion or similar to other
    useless_features = ['id', 'installer', 'wpt_name', 'num_private', 'subvillage',
                        'region', 'region_code', 'district_code', 'ward', 'recorded_by',
                        'scheme_name', 'extraction_type', 'extraction_type_group',
                        'management_group', 'payment_type', 'water_quality', 'quantity_group',
                        'source_type', 'waterpoint_type_group']
    data = []
    id_value = []
    with open(value_path, 'r') as dat_file:
        for i, row in enumerate(csv.reader(dat_file)):
            if i == 0:
                feature_type = row
                feature_dict = {
                    feature_type[k]: k for k in range(len(feature_type))}
                for key in useless_features:
                    del feature_dict[key]
            else:
                # Data is a list with each element is a dictionary
                # Feature can be accessed with data[i][feature]
                data.append({key: row[value]
                             for key, value in feature_dict.items()})
                id_value.append(row[0])

    data_size = len(data)

    # Feature listed in here should be normalized
    continuous_features = ['amount_tsh', 'date_recorded', 'gps_height', 'longitude',
                           'latitude', 'population', 'construction_year']

    # date_recorded should be transformed to the days since it has been recorded
    # Fill some missing data with average value
    for i in range(data_size):
        year = int(data[i]['date_recorded'][:4])
        month = int(data[i]['date_recorded'][5:7])
        date = int(data[i]['date_recorded'][8:10])
        day = (30 - date) + 30 * (12 - month) + 365 * (2017 - year)
        data[i]['date_recorded'] = day
        if float(data[i]['gps_height']) == 0:
            data[i]['gps_height'] = 1016.97
        if float(data[i]['longitude']) == 0:
            data[i]['longitude'] = 35.15
        if float(data[i]['latitude']) > -0.1:
            data[i]['latitude'] = -5.88
        if float(data[i]['construction_year']) == 0:
            data[i]['construction_year'] = 1997
        if float(data[i]['construction_year']) >= 1960:
            data[i]['construction_year'] = float(data[i]['construction_year']) - 1960

    # Normalization
    tmp = [[data[i][feature] for feature in continuous_features]
           for i in range(data_size)]
    tmp = np.array(tmp, dtype=float)
    mean = np.mean(tmp, axis=0)
    std = np.std(tmp, axis=0)
    # This array can be then concatenate with one-hot encoded discrete data
    norm_data = (tmp - mean) / std
    value = norm_data
    # value = tmp

    # Other Feature is discrete and should be dealed with one-hot encoding
    discrete_features = ['funder', 'basin', 'lga', 'public_meeting', 'scheme_management',
                         'permit', 'extraction_type_class', 'management', 'payment',
                         'quality_group', 'quantity', 'source', 'source_class',
                         'waterpoint_type']

    for feature in discrete_features:
        # Temp is a list of dictionary. Each dictionary only contains 1 kind of feature
        tmp = [{feature: data[i][feature]} for i in range(data_size)]
        vec = DictVectorizer()
        # Can be concatenate to value, not yet concatenate
        data_array = vec.fit_transform(tmp).toarray()
        # Single integer to represent discrete features
        value = np.append(value, data_array.argmax(axis=1).reshape((data_array.shape[0], 1)), axis=1)
        # One hot vector to represent discrete features
        # value = np.append(value, data_array, axis=1)
        # Can be concatenate to the feature labels
        data_feature = vec.get_feature_names()
        # print('The size of {}: '.format(feature), end='')
        # print(data_array.shape)

    if label_path:
        return value, label
    else:
        return value, np.array(id_value)

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--random', action='store_true', help='Use Random Forest model')
    parser.add_argument('--xgb', action='store_true', help='Use XGBoost model')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--heatmap', action='store_true', help='Plot the corr heat map')
    args = parser.parse_args()

    data, train_label = read_dataset(os.path.join(BASE_DIR, 'data/values.csv'),
                                     os.path.join(BASE_DIR, 'data/train_labels.csv'))
    feature_list = ['amount_tsh', 'date_recorded', 'gps_height', 'longitude',
                               'latitude', 'population', 'construction_year',
                               'funder', 'basin', 'lga', 'public_meeting', 'scheme_management',
                             'permit', 'extraction_type_class', 'management', 'payment',
                             'quality_group', 'quantity', 'source', 'source_class','waterpoint_type']
        
    if args.heatmap:
        
        construct = [(feature_list[i],data[:,i]) for i in range(len(feature_list))]
        plot_data = pd.DataFrame.from_items(construct)
        label = pd.DataFrame({'status_group':train_label})        
    
        plot_data = pd.concat([plot_data,label],axis = 1)
        colormap = plt.cm.viridis
        
        fig,ax = plt.subplots(figsize=(12,12))
        plt.title('Correlation coefficient map of Features', y=1.05, size=30)
        sns.heatmap(plot_data.corr().round(3),linewidths=0.1,vmax=1.0, square=True,annot_kws={"size":8}
            , cmap=colormap, linecolor='white', annot=True,ax = ax)
        ax.set_xticklabels(plot_data.columns,rotation=90)
        ax.set_yticklabels(plot_data.columns[::-1],rotation=0)
        plt.tight_layout()
        #plt.show()
        fig.savefig('feature_heatmap.png',dpi = 300)

    
    train_data = data[:59400]
    print(train_data.shape)
    if not args.random and not args.xgb and not args.ensemble:
        train_label = to_categorical(train_label)
    indices = np.random.permutation(train_data.shape[0])
    train_data = train_data[indices]
    train_label = train_label[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

    # x_train = train_data[:-nb_validation_samples]
    # y_train = train_label[:-nb_validation_samples]
    # Use all training data
    x_train = train_data
    y_train = train_label
    x_val = train_data[-nb_validation_samples:]
    y_val = train_label[-nb_validation_samples:]

    if args.random:
        clf = RandomForestClassifier(min_samples_split=8,
                                     n_estimators=1000,
                                     oob_score=True,
                                     n_jobs=-1)
        clf.fit(x_train, y_train)
        importances = clf.feature_importances_
        fig,ax = plt.subplots(figsize = (30,15))
        plt.title("Feature importances")
        ax.barh(np.arange(x_train.shape[1]), importances, 0.75,color="b", align="center")
        plt.yticks(np.arange(x_train.shape[1]),feature_list[::-1])
        #plt.show()
        fig.savefig('fi.png',dpi = 200)

        train_ans = clf.predict(x_train)
        val_ans = clf.predict(x_val)
        with open('./model/clf.pkl', 'wb') as clf_file:
            pickle.dump(clf, clf_file)

        print("Training accuracy: {:f}".format(accuracy_score(y_train, train_ans)))
        print("Validation accuracy: {:f}".format(accuracy_score(y_val, val_ans)))
    elif args.xgb:
        
        def get_xgb_feat_importances(clf,features):
            with open('xgb.fmap','w') as f:
                for i, feat in enumerate(features):
                    f.write('{0}\t{1}\tq\n'.format(i, feat))

            if isinstance(clf, xgb.XGBModel):
                # clf has been created by calling
                # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
                fscore = clf.booster().get_fscore()
            else:
                # clf has been created by calling xgb.train.
                # Thus, clf is an instance of xgb.Booster.
                fscore = clf.get_fscore(fmap = 'xgb.fmap')

            feat_importances = []
            for ft, score in fscore.items():
                feat_importances.append({'Feature': ft, 'Importance': score})
            feat_importances = pd.DataFrame(feat_importances)
            # Divide the importances by the sum of all importances
            # to get relative importances. By using relative importances
            # the sum of all importances will equal to 1, i.e.,
            # np.sum(feat_importances['importance']) == 1
            feat_importances['Importance'] /= feat_importances['Importance'].sum()
            # Print the most important features and their importances
            return feat_importances

        xgb_params = {
            'objective': 'multi:softmax',
            'booster': 'gbtree',
            'eval_metric': 'merror',
            'num_class': 3,
            'learning_rate': .1,
            'max_depth': 14,
            'colsample_bytree': .4,
            'colsample_bylevel': .4
        }
        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dxtrain = xgb.DMatrix(x_train)
        dval = xgb.DMatrix(x_val)
        for i in range(11):
            model = xgb.train(dict(xgb_params, seed=i), dtrain, num_boost_round=1200)
            model.save_model("xgb.model_{:d}".format(i))
            importances = get_xgb_feat_importances(model,feature_list)
            print(importances)
            fig,ax = plt.subplots(figsize = (30,15))
            plt.title("Feature importances")
            ax.barh(np.arange(len(importances)), importances['Importance'], 0.75,color="b", align="center")
            plt.yticks(np.arange(len(importances)),importances['Feature'])
            fig.savefig('fscore_{}.png'.format(i),dpi = 100)

            if i == 0:
                train_ans = model.predict(dxtrain).reshape(x_train.shape[0], 1)
                val_ans = model.predict(dval).reshape(x_val.shape[0], 1)
            else:
                train_ans = np.append(train_ans, model.predict(dxtrain).reshape(x_train.shape[0], 1), axis=1)
                val_ans = np.append(val_ans, model.predict(dval).reshape(x_val.shape[0], 1), axis=1)

        for idx, arr in enumerate(train_ans):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            train_ans[idx] = tmp.argmax()

        for idx, arr in enumerate(val_ans):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            val_ans[idx] = tmp.argmax()

        train_ans = train_ans[:, 0]
        val_ans = val_ans[:, 0]
        print("Training accuracy: {:f}".format(accuracy_score(y_train, train_ans)))
        print("Validation accuracy: {:f}".format(accuracy_score(y_val, val_ans)))
    elif args.ensemble:
        n_estimators = [300, 500, 300, 300, 1500]
        rf_max_depth = [24, 25, 26, 27, None]
        for i in range(5):
            clf = RandomForestClassifier(min_samples_split=8,
                                         n_estimators=n_estimators[i],
                                         max_depth=rf_max_depth[i],
                                         oob_score=True,
                                         n_jobs=-1)
            clf.fit(x_train, y_train)
            if i == 0:
                train_ans = clf.predict(x_train).reshape(x_train.shape[0], 1)
                val_ans = clf.predict(x_val).reshape(x_val.shape[0], 1)
            else:
                tmp_train_ans = clf.predict(x_train)
                tmp_val_ans = clf.predict(x_val)
                train_ans = np.append(train_ans, tmp_train_ans.reshape(x_train.shape[0], 1), axis=1)
                val_ans = np.append(val_ans, tmp_val_ans.reshape(x_val.shape[0], 1), axis=1)
            with open("./model/rf_{:d}.pkl".format(i), 'wb') as clf_file:
                pickle.dump(clf, clf_file)

        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dxtrain = xgb.DMatrix(x_train)
        dval = xgb.DMatrix(x_val)
        xgb_max_depth = [4, 7, 32, 32, 48]
        for i in range(5):
            xgb_params = {
                'objective': 'multi:softmax',
                'booster': 'gbtree',
                'eval_metric': 'merror',
                'num_class': 3,
                'learning_rate': .1,
                'max_depth': xgb_max_depth[i],
                'colsample_bytree': .4,
                'colsample_bylevel': .4
            }
            model = xgb.train(dict(xgb_params, seed=i), dtrain, num_boost_round=1181)
            model.save_model("./model/xgb.model_{:d}".format(i))
            train_ans = np.append(train_ans, model.predict(dxtrain).reshape(x_train.shape[0], 1), axis=1)
            val_ans = np.append(val_ans, model.predict(dval).reshape(x_val.shape[0], 1), axis=1)

        for idx, arr in enumerate(train_ans):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            train_ans[idx] = tmp.argmax()

        for idx, arr in enumerate(val_ans):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            val_ans[idx] = tmp.argmax()

        train_ans = train_ans[:, 0]
        val_ans = val_ans[:, 0]
        print("Training accuracy: {:f}".format(accuracy_score(y_train, train_ans)))
        print("Validation accuracy: {:f}".format(accuracy_score(y_val, val_ans)))
    else:
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=x_train.shape[1]))
        model.add(Dropout(DROPOUT_RATE))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(BatchNormalization())
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        model_name = os.path.join(MODEL_DIR, "{epoch:02d}_model.hdf5")
        checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=0,
                                     save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit(x_train, y_train,
                  epochs=300,
                  batch_size=128,
                  validation_data=(x_val, y_val),
                  callbacks=callbacks_list)
    
if __name__ == '__main__':

    main()
