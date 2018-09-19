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

#def FGM(sess, loss_op, x, y_pred, is_targeted = False, epochs=1, sign = False, clip_min = 0, clip_max = 1):
 #   x_adv = tf.identity(x)

def fgm(model,  clip_min = None, clip_max = None, targeted = False, y = None):
    """Fast Gradient Method
            
    """
    predictions = model.prob_predicts # y*
    # y is supposed to be not updated
    if y is None:
        y = tf.stop_gradient(tf.argmax(predictions,axis=-1)) #model.predicts # 不考虑输入
    else:
        y = tf.constant(y)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logits, labels=y)
    if targeted:
        loss = -loss
    # 
    gradient = tf.gradients(loss,model.x_input)[0] # 把inputs考虑进来

    if ord==np.inf:
        signed_grad = tf.sign(gradient)
    elif ord== 1:
        reduc_ind = list(range(1, len(model.x_input.get_shape())))
        signed_grad = gradient / tf.reduce_sum(tf.abs(gradient),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif ord == 2:
        reduc_ind = list(range(1,len(model.x_input.get_shape())))
        signed_grad = gradient / tf.sqrt(tf.reduce_sum(tf.square(gradient),
                                                       reduction_indices=reduc_ind,
                                                       keep_dims=True))
    else:
        raise NotImplementedError
    output = model.x_input + eps * signed_grad

    if (clip_min is not None) and (clip_max is not None):
        output = tf.clip_by_value(output, clip_min, clip_max)

    return output

def main():
    # load session
    image_data_path = "D:\\Workspace\\Projects\\Depth-basedCNN\\DepthCNN\\DepthCNN\\data\\C2Re.mat"
    #X, D = data.load_h5_data(image_data_path, 'TRAIN', 10000)
    X = np.random.rand(2,256,256,6)
    D = np.random.rand(2,32,32,1)
    print("Data loaded: {}".format(X.shape[0]))
    [x_ph, d_ph, loss_op, d_predict_, train_op_] = CNN.create_fcn(False)
    root = "E:\\checkpoints\\4-depth-net-C2RTVT\\output-d-0"
    ckpt_filepath = os.path.join(root, 'save-1000') 
    #saver = tf.train.Saver()
    sess = tf.InteractiveSession() # 如过使用 with tf.session as sess ， 会出现 checkpoint failed的错误
    
    #saver.restore(sess, ckpt_filepath)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.initialize_all_variables())
    [depth_predict] = sess.run([d_predict_], feed_dict={x_ph: X, d_ph:D} )

    
    #x_adv = tf.placeholder(tf.float32, [None, 256, 256, 6])
    D_attack = tf.placeholder(tf.float32, [None, 32, 32, 1])
    depth_predict_ph = tf.placeholder(tf.float32, [None, 32, 32, 1])

    # Fast Gradient Method - Non-targeted
    #grad_ys = tf.get_variable('y_grad',shape=depth_predict.validate_shape, initializer=tf.zeros_initializer())
    #loss_ = tf.nn.sigmoid_cross_entropy_with_logits(labels=D, logits=depth_predict)
  
    #grads = tf.gradients(loss_op, depth_predict, x_ph)#,grad_ys)
    #loss_attack = tf.nn.sigmoid_cross_entropy_with_logits(labels=D_attack, logits=)
    loss_attack = tf.losses.sigmoid_cross_entropy(D_attack, depth_predict_ph)
    dy_dx = tf.gradients(loss_op, x_ph)
    x_adv = tf.identity(x_ph)
    x_adv = tf.stop_gradient(x_adv + 2*tf.sign(dy_dx))
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    IPython.embed()


    x_shape = (2,256,256,6)
    y_shape = (2,32,32,1)
    
    #gradient = train_op_.compute_gradients(loss_op, x_adv)
    #print(type(gradient[0]))
    #init_local = tf.local_variables_initializer()
    #grad_ys.initializer.run()
    #x_adv = tf.add(x_adv, gradient[0])
    
    [x_adv] = sess.run([x_adv] , feed_dict = {x_ph: X,d_ph:D, D_attack:D, depth_predict_ph:depth_predict })
    # 
    print("IPython")
   
    
    sess.close()


        # test 

if __name__ == '__main__':
       main()