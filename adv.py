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
from face_losses import triplet_loss
from nets.L_Resnet_E_IR_GBN import get_resnet_gbn
from nets.L_Resnet_E_IR import get_resnet
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet_fi9
from nets.facenet import get_facenet
from nets.mxnet_models import get_mx34, get_mx50, get_mx100


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file_c', default='./models/ckpt_model_c/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_file_a', default='./models/ckpt_model_a/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--ckpt_file_f', default='./models/facenet_114759/',
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
    parser.add_argument('--ckpt_index_a',
                        default='InsightFace_iter_1060000.ckpt', help='ckpt file indexes')
    parser.add_argument('--ckpt_index_f',
                        default='model-20180402-114759.ckpt', help='ckpt file indexes')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # load lfw images
    root = './AISecurity'
    image_dir = os.path.join(root, 'securityAI_round1_images')
    df = pd.read_csv(os.path.join(root,'securityAI_round1_dev.csv'))

    original_images = []
    start_deltas = []
    # get 712 lfw images' id in this pretrained model
    df_c = pd.read_csv('./csvs/lfw_emd_1950000_multi.csv')
    df_a = pd.read_csv('./csvs/lfw_emd_1060000_multi.csv')
    df_f = pd.read_csv('./csvs/lfw_emd_facenet_multi.csv')
    df_mx34 = pd.read_csv('./csvs/lfw_emd_mx34_multi.csv')
    df_mx50 = pd.read_csv('./csvs/lfw_emd_mx50_multi.csv')
    df_mx100 = pd.read_csv('./csvs/lfw_emd_mx100_multi.csv')
    emd_c_dict = dict()
    emd_a_dict = dict()
    emd_f_dict = dict()
    emd_34_dict = dict()
    emd_50_dict = dict()
    emd_100_dict = dict()
    count = 0
    for image_name, emd_c_, emd_a_, emd_f_, emd_34_, emd_50_, emd_100_ in zip(df_c.image_name, df_c.id_embedding, df_a.id_embedding, df_f.id_embedding, df_mx34.id_embedding, df_mx50.id_embedding, df_mx100.id_embedding):
        count += 1
        print(count)
        emd_c_dict[image_name] = [float(x) for x in emd_c_.split('_')]
        emd_a_dict[image_name] = [float(x) for x in emd_a_.split('_')]
        emd_f_dict[image_name] = [float(x) for x in emd_f_.split('_')]
        emd_34_dict[image_name] = [float(x) for x in emd_34_.split('_')]
        emd_50_dict[image_name] = [float(x) for x in emd_50_.split('_')]
        emd_100_dict[image_name] = [float(x) for x in emd_100_.split('_')]

    #
    print("Finding fastest image name")
    fastest_image_names = dict()
    with open('/csvs/six_farest.csv', 'r') as f:
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
        original_images.append(image)
        start_deltas.append(np.zeros((112,112,3), dtype=np.float))

    w, h = args.image_size
    input_deltas = tf.placeholder(name='delta_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    input_images = tf.placeholder(name='img_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    pos_embedding_c = tf.placeholder(name='pos_embedding_c', shape=[None, 512], dtype=tf.float32)
    neg_embedding_c = tf.placeholder(name='neg_embedding_c', shape=[None, 512], dtype=tf.float32)
    pos_embedding_a = tf.placeholder(name='pos_embedding_a', shape=[None, 512], dtype=tf.float32)
    neg_embedding_a = tf.placeholder(name='neg_embedding_a', shape=[None, 512], dtype=tf.float32)
    pos_embedding_f = tf.placeholder(name='pos_embedding_f', shape=[None, 512], dtype=tf.float32)
    neg_embedding_f = tf.placeholder(name='neg_embedding_f', shape=[None, 512], dtype=tf.float32)
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

    # network A
    net_a = get_resnet_gbn(aug_input, args.net_depth, scope='resnet_1060000_50', type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor_a = tf.reduce_mean(net_a.outputs, axis=0, keep_dims=True)

    # facenet
    aug_input_160 = tf.image.resize_images(aug_input, (160,160), method=0)
    embedding_tensors_f, _ = get_facenet(aug_input_160, 1., phase_train=False, bottleneck_layer_size=512, weight_decay=0.0, reuse=None)
    embedding_tensor_f = tf.reduce_mean(embedding_tensors_f, axis=0, keep_dims=True)

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
    trip_loss_a = triplet_loss(embedding_tensor_a, pos_embedding_a, neg_embedding_a, triplet_alpha)
    trip_loss_f = triplet_loss(embedding_tensor_f, pos_embedding_f, neg_embedding_f, triplet_alpha)
    trip_loss_34 = triplet_loss(embedding_tensor_34, pos_embedding_34, neg_embedding_34, triplet_alpha)
    trip_loss_50 = triplet_loss(embedding_tensor_50, pos_embedding_50, neg_embedding_50, triplet_alpha)
    trip_loss_100 = triplet_loss(embedding_tensor_100, pos_embedding_100, neg_embedding_100, triplet_alpha)

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
    saver_a = tf.train.Saver([v for v in tf.all_variables() if '1060000' in v.name])
    saver_a.restore(sess, args.ckpt_file_a + args.ckpt_index_a)
    saver_f = tf.train.Saver([v for v in tf.all_variables() if 'InceptionResnetV1' in v.name])
    saver_f.restore(sess, args.ckpt_file_f + args.ckpt_index_f)

    limit_range = 25.5
    norm_cum = 0
    cum_num = 0
    for image_name, cur_attack_image, cur_start_delta in zip(df.ImageName, 
            original_images, 
            start_deltas):

        batch_adv_image = np.expand_dims(cur_attack_image.copy(), axis=0)
        target_image_names = [fastest_image_names[image_name]]
        
        min_delta = np.ones((112,112,3), dtype=np.float32)*limit_range*0.0078125
        for target_image_name in target_image_names:
            pos_emd_c = emd_c_dict[target_image_name]
            neg_emd_c = emd_c_dict[image_name]
            pos_emd_a = emd_a_dict[target_image_name]
            neg_emd_a = emd_a_dict[image_name]
            pos_emd_f = emd_f_dict[target_image_name]
            neg_emd_f = emd_f_dict[image_name]
            pos_emd_34 = emd_34_dict[target_image_name]
            neg_emd_34 = emd_34_dict[image_name]
            pos_emd_50 = emd_50_dict[target_image_name]
            neg_emd_50 = emd_50_dict[image_name]
            pos_emd_100 = emd_100_dict[target_image_name]
            neg_emd_100 = emd_100_dict[image_name]

            delta = np.expand_dims(cur_start_delta.copy(), axis=0)
            momentum = np.zeros((1,112,112,3), dtype=np.float32)
            alpha = 0.5
            beta = 0.5
            epsilon = 10e-8

            targeted = False
            prev_total_val_loss = float('inf')
            for i in range(0, steps):
                total_val_loss, loss_c, loss_a, loss_f, loss_34, loss_50, loss_100 = sess.run([total_loss, trip_loss_c, trip_loss_a, trip_loss_f, trip_loss_34, trip_loss_50, trip_loss_100], feed_dict={input_deltas: delta, 
                    input_images: batch_adv_image, 
                    pos_embedding_c: np.array(pos_emd_c).reshape((-1, 512)),
                    neg_embedding_c: np.array(neg_emd_c).reshape((-1, 512)),
                    pos_embedding_a: np.array(pos_emd_a).reshape((-1, 512)),
                    neg_embedding_a: np.array(neg_emd_a).reshape((-1, 512)),
                    pos_embedding_f: np.array(pos_emd_f).reshape((-1, 512)),
                    neg_embedding_f: np.array(neg_emd_f).reshape((-1, 512)),
                    pos_embedding_34: np.array(pos_emd_34).reshape((-1, 512)),
                    neg_embedding_34: np.array(neg_emd_34).reshape((-1, 512)),
                    pos_embedding_50: np.array(pos_emd_50).reshape((-1, 512)),
                    neg_embedding_50: np.array(neg_emd_50).reshape((-1, 512)),
                    pos_embedding_100: np.array(pos_emd_100).reshape((-1, 512)),
                    neg_embedding_100: np.array(neg_emd_100).reshape((-1, 512)),
                    dropout_rate:1.0})
                
                print('Step @ %d : norm: %.4f target loss: %.4f %.4f %.4f %.4f %.4f %.4f' % (i, np.mean(LA.norm(delta[0]*128, axis=-1)), loss_c, loss_a, loss_f, loss_34, loss_50, loss_100))
                stop_condition = ((loss_c < -0.3 and loss_a < -0.3 and loss_f < -0.3 and loss_34 < -0.3 and loss_50 < -0.3 and loss_100 < -0.3) \
                        or (total_val_loss > prev_total_val_loss-0.001) \
                        or total_val_loss < -3.)
                prev_total_val_loss = total_val_loss
                if(stop_condition): 
                    targeted = True
                    break

                d_grad = delta_grad.eval({input_deltas: delta, 
                                          input_images: batch_adv_image, 
                                          pos_embedding_c: np.array(pos_emd_c).reshape((-1, 512)),
                                          neg_embedding_c: np.array(neg_emd_c).reshape((-1, 512)),
                                          pos_embedding_a: np.array(pos_emd_a).reshape((-1, 512)),
                                          neg_embedding_a: np.array(neg_emd_a).reshape((-1, 512)),
                                          pos_embedding_f: np.array(pos_emd_f).reshape((-1, 512)),
                                          neg_embedding_f: np.array(neg_emd_f).reshape((-1, 512)),
                                          pos_embedding_34: np.array(pos_emd_34).reshape((-1, 512)),
                                          neg_embedding_34: np.array(neg_emd_34).reshape((-1, 512)),
                                          pos_embedding_50: np.array(pos_emd_50).reshape((-1, 512)),
                                          neg_embedding_50: np.array(neg_emd_50).reshape((-1, 512)),
                                          pos_embedding_100: np.array(pos_emd_100).reshape((-1, 512)),
                                          neg_embedding_100: np.array(neg_emd_100).reshape((-1, 512)),
                                          dropout_rate:1.0}, session=sess)
                gradient = d_grad

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

                next_delta = np.clip(next_delta, -limit_range*0.0078125, limit_range*0.0078125)
                delta = np.clip(batch_adv_image+next_delta, -1.0, 1.0) - batch_adv_image
                delta = np.around(delta * 128) * 0.0078125

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
            cv2.imwrite(os.path.join('adv_images', image_name.replace('jpg','png')), adv_image[0].astype(np.uint8))
            os.rename(os.path.join('adv_images', image_name.replace('jpg','png')), \
                    os.path.join('adv_images', '%s_%.4f.jpg' % (image_name, final_norm)))
    print('Norm average: %.2f' % (norm_cum/cum_num))
