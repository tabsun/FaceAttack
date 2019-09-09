import tensorflow as tf
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import math
from numpy import linalg as LA
import random
from tqdm import tqdm
#from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss, batch_arcface_loss
# for model c
from nets.L_Resnet_E_IR import get_resnet
# for model a
from nets.L_Resnet_E_IR_GBN import get_resnet_gbn
# for model d
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet_fi9
# for facenet
from nets.facenet import get_facenet, get_gacenet
# for sphereface
from nets.sphereface import get_spherefacenet
# for mxnet
from nets.mxnet_models import get_mx34, get_mx50, get_mx100
#import tensorlayer as tl
#from verification import ver_test


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file', default='./models/sphereface/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--ckpt_index',
                        default='model-20180626-205832.ckpt', help='ckpt file indexes')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # load lfw images
    root = './AISecurity'
    image_dir = os.path.join(root, 'securityAI_round1_images')
    df = pd.read_csv(os.path.join(root,'securityAI_round1_dev.csv'))

    original_images = []
    for image_id, image_name in zip(df.ImageId, df.ImageName):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        original_images.append(image)

    w, h = args.image_size
    input_deltas = tf.placeholder(name='delta_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    input_images = tf.placeholder(name='img_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    #dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    
    net_input = tf.add(input_images, tf.clip_by_value(input_deltas, -25.5, 25.5))
    net_input = tf.clip_by_value(net_input, 0, 255.0)

    left_shift =  tf.manip.roll(net_input, shift=1, axis=2)
    right_shift = tf.manip.roll(net_input, shift=-1, axis=2)
    up_shift =    tf.manip.roll(net_input, shift=1, axis=1)
    down_shift =  tf.manip.roll(net_input, shift=-1, axis=1)
    overlap = tf.concat([net_input, left_shift, right_shift, up_shift, down_shift], axis=0)
    flip_net_input = tf.reverse(overlap, [2])

    aug_input = tf.concat([overlap,flip_net_input],0)
    # resnet
    #w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    #net = get_resnet_fi9(aug_input, args.net_depth, scope='resnet_710000_50', type='ir', keep_rate=dropout_rate, w_init=w_init_method, trainable=False)
    #embedding_tensor = tf.reduce_mean(net.outputs, axis=0, keep_dims=True)
    # facenet
    #aug_input_160 = tf.image.resize_images(aug_input, (160,160), method=0)
    #embedding_tensor, _ = get_gacenet(aug_input, 1., phase_train=False, bottleneck_layer_size=512, weight_decay=0.0, reuse=None)
    # sphere face
    #embedding_tensor = get_spherefacenet(aug_input)
    #embedding_tensor = tf.reduce_mean(embedding_tensor, axis=0)

    sess = tf.Session()
    # mxnet 
    embedding_tensor = get_mx34(aug_input, './models/model-r34-amf/resnet34.npy', sess)
    #embedding_tensor = get_mx50(aug_input, './models/model-r50-am-lfw/resnet50.npy', sess)
    #embedding_tensor = get_mx100(aug_input, './models/model-r100-ii/resnet100.npy', sess)

    #saver = tf.train.Saver()
    #saver.restore(sess, args.ckpt_file + args.ckpt_index)

    with open('lfw_emd_mx34_multi.csv', 'w') as f:
        f.write('image_id,image_name,person_name,id_embedding\n')
    count = 0
    for image_id, image_name, person_name, image in zip(df.ImageId, df.ImageName, df.PersonName, original_images):
        print(count)
        count += 1
        probs = embedding_tensor.eval({input_images:[image], 
                input_deltas:[np.zeros((112,112,3),dtype=np.float)]}, session=sess).flatten()

        with open('lfw_emd_mx34_multi.csv', 'a') as f:
            f.write('%s,%s,%s,%s\n' % (image_id,image_name,person_name,'_'.join(['%.4f'%x for x in probs])))
