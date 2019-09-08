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
from losses.face_losses import arcface_loss, batch_arcface_loss, triplet_loss, single_loss
from nets.L_Resnet_E_IR_GBN import get_resnet_gbn
from nets.L_Resnet_E_IR import get_resnet
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet_fi9
from nets.facenet import get_facenet, get_gacenet
from nets.mxnet_models import get_mx34, get_mx50, get_mx100
#import tensorlayer as tl
#from verification import ver_test


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file_c', default='./models/ckpt_model_c/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_file_d', default='./models/ckpt_model_d/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_file_a', default='./models/ckpt_model_a/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_file_f', default='./models/facenet_114759/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_file_g', default='./models/facenet_102900/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--start_pos', type=int, default=0, help='start_point')
    parser.add_argument('--ckpt_index_c',
                        default='InsightFace_iter_best_1950000.ckpt', help='ckpt file indexes')
    parser.add_argument('--ckpt_index_d',
                        default='InsightFace_iter_best_710000.ckpt', help='ckpt file indexes')
    parser.add_argument('--ckpt_index_a',
                        default='InsightFace_iter_1060000.ckpt', help='ckpt file indexes')
    parser.add_argument('--ckpt_index_f',
                        default='model-20180402-114759.ckpt', help='ckpt file indexes')
    parser.add_argument('--ckpt_index_g',
                        default='model-20180408-102900.ckpt', help='ckpt file indexes')
    args = parser.parse_args()
    return args

def search_best_step_size(step_num, total_step):
    return max(0.08, min(0.5, 0.5 - step_num * 0.42 / 100))
    #return 0.25 + 0.25 * math.cos(3.14 * step_num / total_step)

    valid_step_sizes = []
    for step_size in range(5, 100, 5):
        step_size /= 100.
        if(LA.norm(delta - step_size * gradient) < max_norm):
            valid_step_sizes.append(step_size)
    valid_step_sizes = sorted(valid_step_sizes)
    return valid_step_sizes[len(valid_step_sizes)//2]


if __name__ == '__main__':
    args = get_args()
    # load lfw images
    root = '/data1/tabsun/AISecurity'
    image_dir = os.path.join(root, 'securityAI_round1_images')
    df = pd.read_csv(os.path.join(root,'securityAI_round1_dev.csv'))
    original_images = []
    start_deltas = []
    best_target_ids = dict()
    # get 712 lfw images' id in this pretrained model
    df_c = pd.read_csv('./lfw_emd_1950000_multi.csv')
    df_d = pd.read_csv('./lfw_emd_710000_multi.csv')
    df_a = pd.read_csv('./lfw_emd_1060000_multi.csv')
    df_f = pd.read_csv('./lfw_emd_facenet_multi.csv')
    df_g = pd.read_csv('./lfw_emd_gacenet_multi.csv')
    df_mx34 = pd.read_csv('./lfw_emd_mx34_multi.csv')
    df_mx50 = pd.read_csv('./lfw_emd_mx50_multi.csv')
    df_mx100 = pd.read_csv('./lfw_emd_mx100_multi.csv')
    emd_c_dict = dict()
    emd_a_dict = dict()
    emd_d_dict = dict()
    emd_f_dict = dict()
    emd_g_dict = dict()
    emd_34_dict = dict()
    emd_50_dict = dict()
    emd_100_dict = dict()
    count = 0
    for image_name, emd_c_, emd_d_, emd_a_, emd_f_, emd_g_, emd_34_, emd_50_, emd_100_ in zip(df_c.image_name, df_c.id_embedding, df_d.id_embedding, df_a.id_embedding, df_f.id_embedding, df_g.id_embedding, df_mx34.id_embedding, df_mx50.id_embedding, df_mx100.id_embedding):
        count += 1
        print(count)
        emd_c_dict[image_name] = [float(x) for x in emd_c_.split('_')]
        emd_d_dict[image_name] = [float(x) for x in emd_d_.split('_')]
        emd_a_dict[image_name] = [float(x) for x in emd_a_.split('_')]
        emd_f_dict[image_name] = [float(x) for x in emd_f_.split('_')]
        emd_g_dict[image_name] = [float(x) for x in emd_g_.split('_')]
        emd_34_dict[image_name] = [float(x) for x in emd_34_.split('_')]
        emd_50_dict[image_name] = [float(x) for x in emd_50_.split('_')]
        emd_100_dict[image_name] = [float(x) for x in emd_100_.split('_')]

    #
    print("Finding fastest image name")
    fastest_image_names = dict()
    with open('six_farest.csv', 'r') as f:
        for line in f.readlines():
            first, second = line.strip().split(',')
            fastest_image_names[first] = second
    #count = 0
    #for image_name in df_c.image_name:
    #    count += 1
    #    print(count)
    #    cur_emd_c = emd_c_dict[image_name]
    #    cur_emd_d = emd_d_dict[image_name]
    #    cur_emd_a = emd_a_dict[image_name]
    #    cur_emd_f = emd_f_dict[image_name]
    #    cur_emd_g = emd_g_dict[image_name]
    #    cur_emd_34 = emd_34_dict[image_name]
    #    cur_emd_50 = emd_50_dict[image_name]
    #    cur_emd_100 = emd_100_dict[image_name]


    #    fastest_image_name = None
    #    fastest_dist = 0
    #    for com_image_name in emd_c_dict.keys():
    #        arr1, arr2 = np.array(cur_emd_c), np.array(emd_c_dict[com_image_name])
    #        dist_c = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        arr1, arr2 = np.array(cur_emd_a), np.array(emd_a_dict[com_image_name])
    #        dist_a = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        arr1, arr2 = np.array(cur_emd_f), np.array(emd_f_dict[com_image_name])
    #        dist_f = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        #arr1, arr2 = np.array(cur_emd_d), np.array(emd_d_dict[com_image_name])
    #        #dist_d = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        #arr1, arr2 = np.array(cur_emd_g), np.array(emd_g_dict[com_image_name])
    #        #dist_g = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        arr1, arr2 = np.array(cur_emd_34), np.array(emd_34_dict[com_image_name])
    #        dist_34 = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        arr1, arr2 = np.array(cur_emd_50), np.array(emd_50_dict[com_image_name])
    #        dist_50 = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))

    #        arr1, arr2 = np.array(cur_emd_100), np.array(emd_100_dict[com_image_name])
    #        dist_100 = -np.dot(arr1, arr2) / (LA.norm(arr1) * LA.norm(arr2))
    #        dist = dist_c + dist_a + dist_f + dist_34 + dist_50 + dist_100
    #        if(dist > fastest_dist):
    #            fastest_dist = dist
    #            fastest_image_name = com_image_name
    #    
    #    fastest_image_names[image_name] = fastest_image_name
    #    with open('six_farest.csv', 'a') as f:
    #        f.write('%s,%s\n' % (image_name, fastest_image_name))
    #exit(0)

    
    for image_id, image_name in zip(df.ImageId, df.ImageName):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image = (image - 127.5) * 0.0078125
        #start_image = cv2.imread(os.path.join(start_image_dir, image_name))
        #start_image = (start_image - 127.5) * 0.0078125
        original_images.append(image)
        start_deltas.append(np.zeros((112,112,3), dtype=np.float))
        #start_deltas.append(np.load('mean_delta.npy')*0.0078125)
        #start_deltas.append((start_image - image))
        #start_deltas.append(np.random.uniform(low=-0.0078125, high=0.0078125, size=(112,112,3)))

    w, h = args.image_size
    input_deltas = tf.placeholder(name='delta_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    input_images = tf.placeholder(name='img_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    pos_embedding_c = tf.placeholder(name='pos_embedding_c', shape=[None, 512], dtype=tf.float32)
    neg_embedding_c = tf.placeholder(name='neg_embedding_c', shape=[None, 512], dtype=tf.float32)
    pos_embedding_d = tf.placeholder(name='pos_embedding_d', shape=[None, 512], dtype=tf.float32)
    neg_embedding_d = tf.placeholder(name='neg_embedding_d', shape=[None, 512], dtype=tf.float32)
    pos_embedding_a = tf.placeholder(name='pos_embedding_a', shape=[None, 512], dtype=tf.float32)
    neg_embedding_a = tf.placeholder(name='neg_embedding_a', shape=[None, 512], dtype=tf.float32)
    pos_embedding_f = tf.placeholder(name='pos_embedding_f', shape=[None, 512], dtype=tf.float32)
    neg_embedding_f = tf.placeholder(name='neg_embedding_f', shape=[None, 512], dtype=tf.float32)
    pos_embedding_g = tf.placeholder(name='pos_embedding_g', shape=[None, 512], dtype=tf.float32)
    neg_embedding_g = tf.placeholder(name='neg_embedding_g', shape=[None, 512], dtype=tf.float32)
    pos_embedding_34 = tf.placeholder(name='pos_embedding_34', shape=[None, 512], dtype=tf.float32)
    neg_embedding_34 = tf.placeholder(name='neg_embedding_34', shape=[None, 512], dtype=tf.float32)
    pos_embedding_50 = tf.placeholder(name='pos_embedding_50', shape=[None, 512], dtype=tf.float32)
    neg_embedding_50 = tf.placeholder(name='neg_embedding_50', shape=[None, 512], dtype=tf.float32)
    pos_embedding_100 = tf.placeholder(name='pos_embedding_100', shape=[None, 512], dtype=tf.float32)
    neg_embedding_100 = tf.placeholder(name='neg_embedding_100', shape=[None, 512], dtype=tf.float32)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    
    net_input = tf.add(input_images, tf.clip_by_value(input_deltas, -25.5*0.0078125, 25.5*0.0078125))
    net_input = tf.clip_by_value(net_input, -1.0, 1.0)

    # augmentation
    left_shift =  tf.manip.roll(net_input, shift=1, axis=2)
    right_shift = tf.manip.roll(net_input, shift=-1, axis=2)
    up_shift =    tf.manip.roll(net_input, shift=1, axis=1)
    down_shift =  tf.manip.roll(net_input, shift=-1, axis=1)
    overlap = tf.concat([net_input, left_shift, right_shift, up_shift, down_shift], axis=0)
    flip_net_input = tf.reverse(overlap, [2])
    aug_input = tf.concat([overlap, flip_net_input], 0)

    # network C
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net_c = get_resnet(aug_input, args.net_depth, scope='resnet_1950000_50', type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor_c = tf.reduce_mean(net_c.outputs, axis=0, keep_dims=True)

    # network D
    net_d = get_resnet_fi9(aug_input, args.net_depth, scope='resnet_710000_50', type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor_d = tf.reduce_mean(net_d.outputs, axis=0, keep_dims=True)

    # network A
    net_a = get_resnet_gbn(aug_input, args.net_depth, scope='resnet_1060000_50', type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor_a = tf.reduce_mean(net_a.outputs, axis=0, keep_dims=True)

    # facenet
    aug_input_160 = tf.image.resize_images(aug_input, (160,160), method=0)
    embedding_tensors_f, _ = get_facenet(aug_input_160, 1., phase_train=False, bottleneck_layer_size=512, weight_decay=0.0, reuse=None)
    embedding_tensor_f = tf.reduce_mean(embedding_tensors_f, axis=0, keep_dims=True)

    # gacenet
    embedding_tensors_g, _ = get_gacenet(aug_input_160, 1., phase_train=False, bottleneck_layer_size=512, weight_decay=0.0, reuse=None)
    embedding_tensor_g = tf.reduce_mean(embedding_tensors_g, axis=0, keep_dims=True)

    sess = tf.Session()
    # mxnet-resnet34 50 100
    aug_input_bgr = aug_input[..., ::-1]
    aug_input_abs = aug_input_bgr * 128. + 127.5

    embedding_tensor_34 = get_mx34(aug_input_abs, './models/model-r34-amf/resnet34.npy', sess)
    embedding_tensor_34 = tf.reduce_mean(embedding_tensor_34, axis=0, keep_dims=True)
    embedding_tensor_50 = get_mx50(aug_input_abs, './models/model-r50-am-lfw/resnet50.npy', sess)
    embedding_tensor_100 = get_mx100(aug_input_abs, './models/model-r100-ii/resnet100.npy', sess)

    triplet_alpha = 2.0
    trip_loss_c = triplet_loss(embedding_tensor_c, pos_embedding_c, neg_embedding_c, triplet_alpha)
    trip_loss_d = triplet_loss(embedding_tensor_d, pos_embedding_d, neg_embedding_d, triplet_alpha)
    trip_loss_a = triplet_loss(embedding_tensor_a, pos_embedding_a, neg_embedding_a, triplet_alpha)
    trip_loss_f = triplet_loss(embedding_tensor_f, pos_embedding_f, neg_embedding_f, triplet_alpha)
    trip_loss_g = triplet_loss(embedding_tensor_g, pos_embedding_g, neg_embedding_g, triplet_alpha)
    trip_loss_34 = triplet_loss(embedding_tensor_34, pos_embedding_34, neg_embedding_34, triplet_alpha)
    trip_loss_50 = triplet_loss(embedding_tensor_50, pos_embedding_50, neg_embedding_50, triplet_alpha)
    trip_loss_100 = triplet_loss(embedding_tensor_100, pos_embedding_100, neg_embedding_100, triplet_alpha)

    #trip_loss_c = single_loss(embedding_tensor_c, neg_embedding_c)
    #trip_loss_d = single_loss(embedding_tensor_d, neg_embedding_d)
    #trip_loss_a = single_loss(embedding_tensor_a, neg_embedding_a)
    #trip_loss_f = single_loss(embedding_tensor_f, neg_embedding_f)
    #trip_loss_g = single_loss(embedding_tensor_g, neg_embedding_g)
    #trip_loss_34 = single_loss(embedding_tensor_34, neg_embedding_34)
    #trip_loss_50 = single_loss(embedding_tensor_50, neg_embedding_50)
    #trip_loss_100 = single_loss(embedding_tensor_100, neg_embedding_100)
    total_loss = trip_loss_34 + trip_loss_50 + trip_loss_100  + trip_loss_c + trip_loss_a + trip_loss_f 
    delta_grad = tf.gradients(total_loss, input_deltas)[0]
    step_size = 0.35
    steps = 500

    
    #for v in tf.all_variables():
    #    if('1950000' not in v.name and '710000' not in v.name and '1060000' not in v.name and 'InceptionResnetV1' not in v.name and 'gacenet' not in v.name):
    #        print("You need to initialize these variables:")
    #        print(v.name)
    #        exit(0)
    saver_c = tf.train.Saver([v for v in tf.all_variables() if '1950000' in v.name])
    saver_c.restore(sess, args.ckpt_file_c + args.ckpt_index_c)
    saver_d = tf.train.Saver([v for v in tf.all_variables() if '710000' in v.name])
    saver_d.restore(sess, args.ckpt_file_d + args.ckpt_index_d)
    saver_a = tf.train.Saver([v for v in tf.all_variables() if '1060000' in v.name])
    saver_a.restore(sess, args.ckpt_file_a + args.ckpt_index_a)
    saver_f = tf.train.Saver([v for v in tf.all_variables() if 'InceptionResnetV1' in v.name])
    saver_f.restore(sess, args.ckpt_file_f + args.ckpt_index_f)
    saver_g = tf.train.Saver([v for v in tf.all_variables() if 'gacenet' in v.name])
    saver_g.restore(sess, args.ckpt_file_g + args.ckpt_index_g)

    limit_range = 25.5
    norm_cum = 0
    cum_num = 0
    for image_name, cur_attack_image, cur_start_delta in zip(df.ImageName[args.start_pos:args.start_pos+41], 
            original_images[args.start_pos:args.start_pos+41], 
            start_deltas[args.start_pos:args.start_pos+41]):
        #if(os.path.exists(os.path.join('./passed', image_name.replace('.jpg','.npy')))):
        #    continue
        # not update adv_images
        #processed = False
        #for fname in os.listdir('adv_images'):
        #    if(fname.endswith('.jpg') and (image_name+'_') in fname):
        #        processed = True
        #        break
        #if(processed):
        #    continue

        batch_adv_image = np.expand_dims(cur_attack_image.copy(), axis=0)

        target_image_names = [fastest_image_names[image_name]]
        
        min_delta = np.ones((112,112,3), dtype=np.float32)*limit_range*0.0078125
        for target_image_name in target_image_names:
            pos_emd_c = emd_c_dict[target_image_name]
            neg_emd_c = emd_c_dict[image_name]
            pos_emd_d = emd_d_dict[target_image_name]
            neg_emd_d = emd_d_dict[image_name]
            pos_emd_a = emd_a_dict[target_image_name]
            neg_emd_a = emd_a_dict[image_name]
            pos_emd_f = emd_f_dict[target_image_name]
            neg_emd_f = emd_f_dict[image_name]
            pos_emd_g = emd_g_dict[target_image_name]
            neg_emd_g = emd_g_dict[image_name]
            pos_emd_34 = emd_34_dict[target_image_name]
            neg_emd_34 = emd_34_dict[image_name]
            pos_emd_50 = emd_50_dict[target_image_name]
            neg_emd_50 = emd_50_dict[image_name]
            pos_emd_100 = emd_100_dict[target_image_name]
            neg_emd_100 = emd_100_dict[image_name]

            delta = np.expand_dims(cur_start_delta.copy(), axis=0)
            momentum = np.zeros((1,112,112,3), dtype=np.float32)
            vt = np.zeros((1,112,112,3), dtype=np.float32)
            mt = np.zeros((1,112,112,3), dtype=np.float32)
            alpha = 0.5
            beta = 0.5
            beta1 = 0.9
            beta2 = 0.99
            epsilon = 10e-8

            targeted = False
            lost = False
            zero_passed = False
            prev_total_val_loss = float('inf')
            for i in range(0, steps):
                total_val_loss, loss_c, loss_d, loss_a, loss_f, loss_g, loss_34, loss_50, loss_100 = sess.run([total_loss, trip_loss_c, trip_loss_d, trip_loss_a, trip_loss_f, trip_loss_g, trip_loss_34, trip_loss_50, trip_loss_100], feed_dict={input_deltas: delta, 
                    input_images: batch_adv_image, 
                    pos_embedding_c: np.array(pos_emd_c).reshape((-1, 512)),
                    neg_embedding_c: np.array(neg_emd_c).reshape((-1, 512)),
                    pos_embedding_d: np.array(pos_emd_d).reshape((-1, 512)),
                    neg_embedding_d: np.array(neg_emd_d).reshape((-1, 512)),
                    pos_embedding_a: np.array(pos_emd_a).reshape((-1, 512)),
                    neg_embedding_a: np.array(neg_emd_a).reshape((-1, 512)),
                    pos_embedding_f: np.array(pos_emd_f).reshape((-1, 512)),
                    neg_embedding_f: np.array(neg_emd_f).reshape((-1, 512)),
                    pos_embedding_g: np.array(pos_emd_g).reshape((-1, 512)),
                    neg_embedding_g: np.array(neg_emd_g).reshape((-1, 512)),
                    pos_embedding_34: np.array(pos_emd_34).reshape((-1, 512)),
                    neg_embedding_34: np.array(neg_emd_34).reshape((-1, 512)),
                    pos_embedding_50: np.array(pos_emd_50).reshape((-1, 512)),
                    neg_embedding_50: np.array(neg_emd_50).reshape((-1, 512)),
                    pos_embedding_100: np.array(pos_emd_100).reshape((-1, 512)),
                    neg_embedding_100: np.array(neg_emd_100).reshape((-1, 512)),
                    dropout_rate:1.0})
                
                print('Step @ %d : norm: %.4f target loss: %.4f %.4f %.4f %.4f %.4f %.4f' % (i, np.mean(LA.norm(delta[0]*128, axis=-1)), loss_c, loss_a, loss_f, loss_34, loss_50, loss_100))
                #if(prediction_loss_c[0] + prediction_loss_a[0] > prev_loss - 0.0001):
                #    lost = True
                #    break
                # range : 2 ~ 4
                stop_condition = ((loss_c < -0.3 and loss_a < -0.3 and loss_f < -0.3 and loss_34 < -0.3 and loss_50 < -0.3 and loss_100 < -0.3) \
                        or (total_val_loss > prev_total_val_loss-0.001) \
                        or total_val_loss < -3.)
                prev_total_val_loss = total_val_loss
                if(stop_condition): 
                    targeted = True
                    break

                if(False): 
                    print("Nothing")
                    #gradient = delta_grad.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    #grad_c = delta_grad_c.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    #grad_a = delta_grad_a.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)

                    
                    #if(prediction_val_c[0] == cur_target_label):
                    #    prediction_loss_c = 0
                    #if(prediction_val_d[0] == cur_target_label):
                    #    prediction_loss_d = 0

                    #c_loss = prediction_loss_c / (prediction_loss_c + prediction_loss_d)
                    #a_loss = prediction_loss_a / (prediction_loss_a + prediction_loss_a)
                    #kernel = np.ones((3,3), np.float32) / 9.
                    #grad_c[0, :, :, :] = cv2.filter2D(grad_c[0], -1, kernel)
                    #gradient = grad_c * c_loss + grad_a * a_loss
                    #if(prediction_val_c[0] != cur_target_label and prediction_val_d[0] != cur_target_label):
                    #    gradient = 0.3 * grad_c + 0.7 * grad_d
                    #elif(prediction_val_c[0] != cur_target_label):
                    #    gradient = grad_c + 0.1 * grad_d
                    #else:
                    #    gradient = grad_d + 0.1 * grad_c
                    #if(random.random() > 0.5):
                    #    gradient = delta_grad.eval({input_deltas: np.flip(delta,axis=2), input_images:np.flip(batch_adv_image,axis=2),input_labels:[cur_target_label], dropout_rate:1.0}, session=sess)
                else:
                    #if(not ori_targeted):
                    d_grad = delta_grad.eval({input_deltas: delta, 
                                              input_images: batch_adv_image, 
                                              pos_embedding_c: np.array(pos_emd_c).reshape((-1, 512)),
                                              neg_embedding_c: np.array(neg_emd_c).reshape((-1, 512)),
                                              pos_embedding_d: np.array(pos_emd_d).reshape((-1, 512)),
                                              neg_embedding_d: np.array(neg_emd_d).reshape((-1, 512)),
                                              pos_embedding_a: np.array(pos_emd_a).reshape((-1, 512)),
                                              neg_embedding_a: np.array(neg_emd_a).reshape((-1, 512)),
                                              pos_embedding_f: np.array(pos_emd_f).reshape((-1, 512)),
                                              neg_embedding_f: np.array(neg_emd_f).reshape((-1, 512)),
                                              pos_embedding_g: np.array(pos_emd_g).reshape((-1, 512)),
                                              neg_embedding_g: np.array(neg_emd_g).reshape((-1, 512)),
                                              pos_embedding_34: np.array(pos_emd_34).reshape((-1, 512)),
                                              neg_embedding_34: np.array(neg_emd_34).reshape((-1, 512)),
                                              pos_embedding_50: np.array(pos_emd_50).reshape((-1, 512)),
                                              neg_embedding_50: np.array(neg_emd_50).reshape((-1, 512)),
                                              pos_embedding_100: np.array(pos_emd_100).reshape((-1, 512)),
                                              neg_embedding_100: np.array(neg_emd_100).reshape((-1, 512)),
                                              dropout_rate:1.0}, session=sess)
                    # resize
                    #kernel = np.ones((3,3), np.float32)
                    #kernel /= np.sum(kernel)
                    #d_grad[0, :, :, :] = cv2.filter2D(d_grad[0], -1, kernel)
                    #elif(not flip_targeted):
                    #    d_grad = delta_grad.eval({input_deltas: np.flip(delta,axis=2), input_images: np.flip(batch_adv_image,axis=2), input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    #else:
                    #    assert("How could be?")
                    #    exit(0)

                    #n_grad = norm_grad.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    # gradient for 0 postion will be nan
                    #n_grad[np.isnan(n_grad)] = 0
                    #if(i % 3 == 0): #prediction_val[0] == cur_target_label):
                    #    gradient = 10 * n_grad
                    #else:
                    #    gradient = 0.1 * d_grad
                    gradient = d_grad# + 2*n_grad

                #best_step_size = None
                #min_try_loss = float('inf')
                #for step_size in range(3, 20, 2):
                #    step_size /= 100
                #    
                #    try_momentum = step_size * gradient * alpha + beta * momentum
                #    try_next_delta = delta - try_momentum
                #    try_next_delta = np.clip(try_next_delta, -limit_range*0.0078125, limit_range*0.0078125)
                #    try_delta = np.clip(batch_adv_image+try_next_delta,-0.99609375,0.99609375) - batch_adv_image
                #    try_loss = total_loss.eval({input_deltas: try_delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                #    if(try_loss < min_try_loss):
                #        min_try_loss = try_loss
                #        best_step_size = step_size

                #g_norm = np.mean(LA.norm(gradient,axis=-1))
                #if(np.array_equal(np.clip(delta - step_size*gradient, -limit_range * 0.0078125, limit_range * 0.0078125), delta)):
                #    break

                # ADAM
                #mt = beta1 * mt + (1.-beta1)*gradient
                #vt = beta2 * vt + (1.-beta2)*np.multiply(gradient,gradient)
                #mt_ = mt / (1-pow(beta1,i+1))
                #vt_ = vt / (1-pow(beta2,i+1))
                #next_delta = delta - 0.004 * (mt_ / (np.sqrt(vt_) + epsilon))

                # RADAM

                # ADAGRAD
                #next_delta = delta - 0.2*np.multiply(gradient, np.sqrt(mt+epsilon))
                #mt += np.multiply(gradient,gradient)
                
                # SGD
                #next_delta = delta - step_size * gradient

                # Momentum ver1.0
                momentum = step_size * gradient * alpha + beta * momentum
                next_delta = delta - momentum
                # Momentum ver1.1
                #momentum = -gradient * step_size + beta * momentum
                #next_delta = delta + momentum

                # FGV
                #momentum = beta * momentum + gradient / np.mean(LA.norm(gradient, axis=-1))
                #next_delta = step_size * np.sign(momentum)

                #next_delta = np.clip(next_delta, min_mask, max_mask)
                next_delta = np.clip(next_delta, -limit_range*0.0078125, limit_range*0.0078125)
                #delta = np.around(delta * 128) * 0.0078125
                delta = np.clip(batch_adv_image+next_delta, -1.0, 1.0) - batch_adv_image
                delta = np.around(delta * 128) * 0.0078125
                #if(np.array_equal(np.clip(batch_adv_image+delta,-1.0,1.0), batch_adv_image)):
                #    break

            # choose the best delta
            if(targeted and LA.norm(delta[0]) < LA.norm(min_delta)):
                min_delta = delta
        adv_image = np.clip((batch_adv_image + min_delta)*128+127.5, 0, 255)
        final_norm = np.mean(LA.norm(min_delta[0]*128., axis=-1))
        norm_cum += final_norm
        cum_num += 1
        print('Final norm:', final_norm)

        existed_score = float('inf')
        existed_filename = None
        for fname in os.listdir('adv_images'):
            if(fname.endswith('.jpg') and (image_name+'_') in fname):
                existed_score = float(fname.split('_')[-1].split('.jpg')[0])
                existed_filename = fname

        if(targeted and final_norm < existed_score):
            print('Update %s from %.4f to %.4f' % (image_name, existed_score, final_norm))
            if(existed_filename is not None):
                os.remove(os.path.join('adv_images', existed_filename))
            #np.save('npys/%s' % image_name.replace('jpg','npy'), min_delta[0])
            cv2.imwrite(os.path.join('adv_images', image_name.replace('jpg','png')), adv_image[0].astype(np.uint8))
            os.rename(os.path.join('adv_images', image_name.replace('jpg','png')), \
                    os.path.join('adv_images', '%s_%.4f.jpg' % (image_name, final_norm)))
        #if(cum_num > 50):
        #    break
    print('Norm average: %.2f' % (norm_cum/cum_num))
