import sys
import os


import numpy as np
import utils.data_prepare as data
import utils.CNN as CNN
import IPython
from tensorflow.python import debug as tf_debug
import h5py as h5
from skimage import io, color
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
import IPython
sys.path.append('.\\')
import adv_method
import logging
import csv
DB_PATH = r"D:\Workspace\Projects\Adversarial Attack\Adversarial Attack\data\CASIA_depth.mat"


#def make_svm_format_data(X):
def grid_search(X,y, csv_filepath):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4,],
                     'C': np.logspace(-10,5,100).tolist()},
                     {'kernel': ['linear'], 'C': np.logspace(-10,5,100).tolist()}] 
    #scores = ['accuracy', 'roc_auc']
    scores = ['roc_auc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                        scoring=score, n_jobs=5)
        # 用训练集训练这个学习器 clf
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)
        print("Grid scores on development set:")
    
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))
    
    return clf
            
        
if __name__ == '__main__':
    mat = h5.File(DB_PATH, 'r')
    X, D, LBL = data.load_h5_data(mat, 'TRAIN', 1000) # X: 256X256X3 D:32X32X3
    #del X
    print(LBL.shape)
    

    num_sample = D.shape[0]
    new_shape_ = [num_sample, D.shape[1]*D.shape[2]*D.shape[3]]
    depth_feature = np.zeros(new_shape_)
    for i in range(num_sample):
        # flatten
        depth_feature[i,:] = np.reshape(D[i,:,:,:], [1,D.shape[1]*D.shape[2]*D.shape[3]]) * LBL[i]
        print("Preprocessing {}/{}".format(i+1, num_sample))

    print("Number of the genuine sample: ", sum(LBL))
    #IPython.embed()
    #svm_model = svm.SVC()
    #svm_model.fit(depth_feature, LBL)
    csv_file = "grid_search.csv"
    clf = grid_search(depth_feature, LBL, csv_file)

    # befor attack
    #y_true, y_pred = y_test, clf.predict(X_test)
        # 打印在测试集上的预测结果与真实值的分数
    #print(classification_report(y_true, y_pred))

    alpha = 0.01
    iteration = 35

    _, depth_adv_ = adv_method.fgm(X, D, True, alpha, iteration)
    num_sample = D.shape[0]
    new_shape_ = [num_sample, D.shape[1]*D.shape[2]*D.shape[3]]
    depth_adv = np.zeros(new_shape_)
    for i in range(num_sample):
        # flatten
        depth_adv[i,:] = np.reshape(depth_adv_[i,:,:,:], [1,depth_adv_.shape[1]*depth_adv_.shape[2]*depth_adv_.shape[3]]) 
        print("Preprocessing {}/{}".format(i+1, num_sample))
    del depth_adv_
    del _
    # after attack
    y_adv_pred = clf.predict(depth_adv)
    print(classification_report(LBL, y_adv_pred))





    
    
    