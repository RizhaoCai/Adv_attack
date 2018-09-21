import os

import tensorflow as tf
import numpy as np
import utils.data_prepare as data
import utils.CNN as CNN
import IPython
from tensorflow.python import debug as tf_debug
import h5py as h5
from skimage import io, color
import matplotlib.pyplot as plt
from matplotlib import animation

SEP = os.path.sep 
ckpt_filepath = 'data' + SEP + 'checkpoint' + SEP + "save-1000"

def preprocess(X):
    n = X.shape[0]
    X_hsv = np.zeros(X.shape)
    for i_ in range(n):
        im = X[i_]
        im = color.rgb2hsv(im)
        X_hsv[i_] = im
    return np.concatenate((X,X_hsv), axis=-1)/255.0

def fgm(x_input, y, is_targetd=True, alpha=1, iteration=1, save_path = None, sign=True, ):
    x_input = preprocess(x_input) # RGB -> RGB + HSV
    ckpt_filepath = 'data' + SEP + 'checkpoint' + SEP + "save-1000"
    [x_ph, d_ph, loss_op, d_predict_, train_op_] = CNN.create_fcn(False)  
    y_target = y*int(is_targetd)
    #loss_op = tf.losses.mean_squared_error(y_target, d_predict_)  
    dy_dx = tf.gradients(loss_op, x_ph)
    dy_dx = dy_dx[0]
    if is_targetd:
        x_adv = x_ph - alpha*tf.sign(dy_dx)
    else:
        x_adv = x_ph + alpha*tf.sign(dy_dx) 
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    sess = tf.InteractiveSession() # 如过使用 with tf.session as sess ， 会出现 checkpoint failed的错误
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filepath)
    xadv = x_input
    if save_path is not None and not os.path.exists(save_path):
        print("Make Dir: ", save_path)
        os.mkdir(save_path)

    #y_pred = sess.run([d_predict_], {x_ph:x_input} )
    for i in range(iteration):
        [xadv, dydx ]  = sess.run([x_adv,dy_dx] , feed_dict = {x_ph: xadv, d_ph:y_target})
        #xadv = xadv[0]
        print("iteration times:", i+1)
        # prediction results of y with adversarial examples
    depth_adv = sess.run([d_predict_], {x_ph:xadv})
        
    return xadv, depth_adv