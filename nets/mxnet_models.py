from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import os, sys
sys.path.append('/data/tabsun/temp/Adversarial/InsightFace_TF/nets')

def get_mx50(input_tensor, model_path, sess):
    # import converted model
    model_converted = __import__('tf_resnet50').KitModel(input_tensor, model_path)
    input_tf, model_tf = model_converted
    output_tensor = tf.reduce_mean(model_tf, axis=0, keep_dims=True)

    # inference with tensorflow
    init = tf.global_variables_initializer()
    sess.run(init)
    return output_tensor

def get_mx100(input_tensor, model_path, sess):
    # import converted model
    model_converted = __import__('tf_resnet100').KitModel(input_tensor, model_path)
    input_tf, model_tf = model_converted
    output_tensor = tf.reduce_mean(model_tf, axis=0, keep_dims=True)

    # inference with tensorflow
    init = tf.global_variables_initializer()
    sess.run(init)
    return output_tensor

def get_mx34(input_tensor, model_path, sess):
    # import converted model
    model_converted = __import__('tf_resnet34').KitModel(input_tensor, model_path)
    input_tf, model_tf = model_converted
    output_tensor = tf.reduce_mean(model_tf, axis=0, keep_dims=True)

    # inference with tensorflow
    init = tf.global_variables_initializer()
    sess.run(init)
    return output_tensor

if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_place')

    sess = tf.Session()
    embedding = get_mx50(x, '../models/model-r50-am-lfw/resnet50.npy', sess)

    emd = embedding.eval({x:[np.zeros((112,112,3), dtype=np.float)]}, session=sess)
    print(emd.shape)
