"""Implementation of Adversarial Attack
        1. Fast Gradient Method
        2. Optimization Method
"""
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


#@fgm_with_plot
def fgm_demo(x_input, y, is_targetd=True, alpha=1, iteration=1, save_path = None, sign=True, ):
    ckpt_filepath = 'data' + SEP + 'checkpoint' + SEP + "save-1000"
    [x_ph, d_ph, loss_op, d_predict_, train_op_] = CNN.create_fcn(False)  
    y_target = y*int(is_targetd)
    loss_op = tf.losses.mean_squared_error(y_target, d_predict_)  
    dy_dx = tf.gradients(loss_op, x_ph)
    dy_dx = dy_dx[0]
    if is_targetd:
        x_adv = x_ph - alpha*tf.sign(dy_dx)
    else:
        x_adv = x_ph + alpha*tf.sign(dy_dx) 
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession() # 如过使用 with tf.session as sess ， 会出现 checkpoint failed的错误
    saver.restore(sess, ckpt_filepath)
    xadv = x_input
    if not os.path.exists(save_path):
        print("Make Dir: ", save_path)
        os.mkdir(save_path)

    y_pred = sess.run([d_predict_],{x_ph:x_input} )
    for i in range(iteration):
        [xadv, dydx ]  = sess.run([x_adv,dy_dx] , feed_dict = {x_ph: xadv, d_ph:y_target})
        #xadv = xadv[0]
        print("iteration times:", i+1)
        # prediction results of y with adversarial examples
        y_adv = sess.run([d_predict_], {x_ph:xadv})
        if save_path != None:
            if i< 5 or (i+1)%5 == 0: 
                i_ = 0          
                plt.figure()
                plt.subplot(231)
                plt.axis('off')
                plt.imshow(x_input[i_,:,:,0:3])
                plt.title("Input")

                
                plt.subplot(232)
                plt.axis('off')
                plt.imshow(xadv[i_,:,:,0:3])
                plt.title("Perturbated")


                plt.subplot(233)
                plt.axis('off')
                #dydx_adjustd = dydx + 1
                plt.imshow(dydx[i_,:,:,0])
                plt.title("Perturbation")
                
                plt.subplot(234)
                plt.axis('off')
                plt.imshow(y[i_,:,:,0],cmap="gray")
                plt.title("Pseudo Depth")
                
                plt.subplot(235)
                plt.axis('off')
                plt.imshow(y_pred[i_][0,:,:,0], cmap='gray')
                plt.title("Expected Depth")
                
                plt.subplot(236)
                plt.imshow(y_adv[0][i_,:,:,0], cmap='gray')
                plt.title("Adversarial Depth")
                plt.axis('off')

                #plt.pause(0.5)
                target = "Target"
                im_name = "{}_Alpha_{}_It_{}.jpg".format("Target" if is_targetd else "NonTarget", alpha, i  )
                plt_savedir = os.path.join(save_path,im_name)  
                plt.savefig(plt_savedir)
                plt.close()
  
    sess.close()
    return xadv, y_adv


def main(idx):
    # load session
    image_data_path = r"D:\Workspace\Projects\Adversarial Attack\Adversarial Attack\data\CASIA_depth.mat"
    print(image_data_path)
    mat = h5.File(image_data_path, 'r')
    X, D, LBL = data.load_h5_data(mat, 'TRAIN', 10000)
    mat.close()
    #X = np.random.rand(2,256,256,6)
    #D = np.random.rand(2,32,32,1)
   
    indexs_of_spoof =  np.argwhere( LBL == 0)[:,0:1]
    index = indexs_of_spoof[idx,0]
    X = X[index:index+1]
    D = D[index:index+1]
    X = preprocess(X)
    
    print("Data loaded: {}".format(X.shape[0]))
    graph1 = tf.Graph()
    root = "D:\\Workspace\\Projects\\Adversarial Attack\\Adversarial Attack\\data\\checkpoint"
    ckpt_filepath = os.path.join(root, 'save-1000') 

    with graph1.as_default():
        is_targetd = True   
        iteration = 200
        #alpha = 1
        for is_targetd in [True, False]:
            for alpha in [10,1,0.1,0.01,0.001,0.0001]:
                xadv, pred_adv= fgm(X, D, is_targetd, alpha, iteration, "FGM_"+ str(idx) ) # Non-targeted attack
    pred_adv = pred_adv[0]
    print(xadv.shape)
    print(X.shape)

    """
    i_ = 0
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.imshow(X[i_,:,:,0:3])
    plt.title("original")
    
    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.imshow(D[i_,:,:,0],cmap="gray")
    plt.title("Label Depth")
    
    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.imshow(depth_predict[i_,:,:,0], cmap='gray')
    plt.title("predict of original")
    
    plt.subplot(2, 2, 4)
    plt.imshow(pred_adv[i_,:,:,0], cmap='gray')
    plt.title("predict of the adversarial example")
    plt.axis('off')
    plt.show()
    #input("press any keys to show the next result")
    """

def fgm_demo(x_input, y, is_targetd=True, alpha=1, iteration=1, save_path = None, sign=True, ):
    ckpt_filepath = 'data' + SEP + 'checkpoint' + SEP + "save-1000"
    [x_ph, d_ph, loss_op, d_predict_, train_op_] = CNN.create_fcn(False)  
    y_target = y*int(is_targetd)
    loss_op = tf.losses.mean_squared_error(y_target, d_predict_)  
    dy_dx = tf.gradients(loss_op, x_ph)
    dy_dx = dy_dx[0]
    if is_targetd:
        x_adv = x_ph - alpha*tf.sign(dy_dx)
    else:
        x_adv = x_ph + alpha*tf.sign(dy_dx) 
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession() # 如过使用 with tf.session as sess ， 会出现 checkpoint failed的错误
    saver.restore(sess, ckpt_filepath)
    xadv = x_input
    if not os.path.exists(save_path):
        print("Make Dir: ", save_path)
        os.mkdir(save_path)

    y_pred = sess.run([d_predict_],{x_ph:x_input} )
    for i in range(iteration):
        [xadv, dydx ]  = sess.run([x_adv,dy_dx] , feed_dict = {x_ph: xadv, d_ph:y_target})
        #xadv = xadv[0]
        print("iteration times:", i+1)
        # prediction results of y with adversarial examples
        y_adv = sess.run([d_predict_], {x_ph:xadv})
        if save_path != None:
            if i< 5 or (i+1)%5 == 0: 
                i_ = 0          
                plt.figure()
                plt.subplot(231)
                plt.axis('off')
                plt.imshow(x_input[i_,:,:,0:3])
                plt.title("Input")

                
                plt.subplot(232)
                plt.axis('off')
                plt.imshow(xadv[i_,:,:,0:3])
                plt.title("Perturbated")


                plt.subplot(233)
                plt.axis('off')
                #dydx_adjustd = dydx + 1
                plt.imshow(dydx[i_,:,:,0])
                plt.title("Perturbation")
                
                plt.subplot(234)
                plt.axis('off')
                plt.imshow(y[i_,:,:,0],cmap="gray")
                plt.title("Pseudo Depth")
                
                plt.subplot(235)
                plt.axis('off')
                plt.imshow(y_pred[i_][0,:,:,0], cmap='gray')
                plt.title("Expected Depth")
                
                plt.subplot(236)
                plt.imshow(y_adv[0][i_,:,:,0], cmap='gray')
                plt.title("Adversarial Depth")
                plt.axis('off')

                #plt.pause(0.5)
                target = "Target"
                im_name = "{}_Alpha_{}_It_{}.jpg".format("Target" if is_targetd else "NonTarget", alpha, i  )
                plt_savedir = os.path.join(save_path,im_name)  
                plt.savefig(plt_savedir)
                plt.close()
  
    sess.close()
    return xadv, y_adv


def main(idx):
    # load session
    image_data_path = r"D:\Workspace\Projects\Adversarial Attack\Adversarial Attack\data\CASIA_depth.mat"
    print(image_data_path)
    mat = h5.File(image_data_path, 'r')
    X, D, LBL = data.load_h5_data(mat, 'TRAIN', 10000)
    mat.close()
    #X = np.random.rand(2,256,256,6)
    #D = np.random.rand(2,32,32,1)
   
    indexs_of_spoof =  np.argwhere( LBL == 0)[:,0:1]
    index = indexs_of_spoof[idx,0]
    X = X[index:index+1]
    D = D[index:index+1]
    X = preprocess(X)
    
    print("Data loaded: {}".format(X.shape[0]))
    graph1 = tf.Graph()
    root = "D:\\Workspace\\Projects\\Adversarial Attack\\Adversarial Attack\\data\\checkpoint"
    ckpt_filepath = os.path.join(root, 'save-1000') 

    with graph1.as_default():
        is_targetd = True   
        iteration = 200
        #alpha = 1
        for is_targetd in [True, False]:
            for alpha in [10,1,0.1,0.01,0.001,0.0001]:
                xadv, pred_adv= fgm(X, D, is_targetd, alpha, iteration, "FGM_"+ str(idx) ) # Non-targeted attack
    pred_adv = pred_adv[0]
    print(xadv.shape)
    print(X.shape)
if __name__ == '__main__':
    for idx in range(0,9):
        main(idx)
