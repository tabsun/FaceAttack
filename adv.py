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
from nets.L_Resnet_E_IR import get_resnet
#import tensorlayer as tl
#from verification import ver_test


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file', default='./models/ckpt_model_c/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--ckpt_index',
                        default='InsightFace_iter_best_1950000.ckpt', help='ckpt file indexes')
    args = parser.parse_args()
    return args

def search_best_step_size(step_num, total_step):
    return max(0.1, min(0.6, 0.1 + step_num * 0.5 / 10))
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
    original_labels = []
    target_labels = []
    start_deltas = []
    # get 712 lfw images' id in this pretrained model
    lfw_df = pd.read_csv('lfw_ids_1950000_flip.csv')
    lfw_ids = dict()
    lfw_losses = dict()
    for image_name, id_prob in zip(lfw_df.image_name, lfw_df.id_prob):
        prob = [float(x) for x in id_prob.split('_')]
        lfw_ids[image_name] = prob.index(min(prob))
        lfw_losses[image_name] = min(prob)

    best_target_ids = dict()
    for image_name, id_prob in zip(lfw_df.image_name, lfw_df.id_prob):
        prob = [float(x) for x in id_prob.split('_')]
        valid_prob = [prob[i] for i in lfw_ids.values()]
        #sorted_prob = sorted(valid_prob)
        sorted_prob = sorted(prob)
        jump_step = len(sorted_prob) // 10
        best_target_ids[image_name] = [prob.index(sorted_prob[-1])] #for x in range(1, len(sorted_prob), jump_step)]

    #for i in range(1,713):
    #    key = '%05d.jpg' % i
    #    print('%s: %d' % (key, best_target_ids[key]))
    #exit(0)

    start_image_dir = './start_images'
    for image_id, image_name in zip(df.ImageId, df.ImageName):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image = (image - 127.5) * 0.0078125
        start_image = cv2.imread(os.path.join(start_image_dir, image_name))
        start_image = (start_image - 127.5) * 0.0078125
        original_images.append(image)
        original_labels.append(int(image_id)-1)
        target_image_name = None
        while(True):
            target_image_name = '%05d.jpg' % (int(random.random()*713*3) % 713)
            if(target_image_name != image_name and target_image_name in lfw_ids.keys()):
                break
        target_label = lfw_ids[target_image_name]
        target_labels.append(target_label)
        start_deltas.append(np.zeros((112,112,3), dtype=np.float))
        #start_deltas.append(start_image - image)
        #start_deltas.append(np.random.uniform(low=-0.0078125, high=0.0078125, size=(112,112,3)))

    w, h = args.image_size
    input_deltas = tf.placeholder(name='delta_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    input_images = tf.placeholder(name='img_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    input_labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
    
    net_input = tf.add(input_images, tf.clip_by_value(input_deltas, -25 * 0.0078125, 25 * 0.0078125))
    net_input = tf.clip_by_value(net_input, -1.0, 1.0)

    # augmentation
    flip_net_input = tf.reverse(net_input, [2])
    #bgr_net_input = tf.reverse(net_input, [3])
    
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(tf.concat([net_input,flip_net_input], 0), args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor = tf.reduce_mean(net.outputs, axis=0, keep_dims=True)

    # 3.2 get arcface loss
    #logit = arcface_loss(embedding=embedding_tensor, labels=input_labels, w_init=w_init_method, out_num=args.num_output)
    logit = arcface_loss(embedding=embedding_tensor, labels=input_labels, w_init=w_init_method, out_num=args.num_output)
    #logit = tf.squeeze(logit, axis=0)
    prediction = tf.argmax(logit, axis=1)
    
    inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=input_labels)
    norm_loss = tf.reduce_mean(tf.norm(input_deltas, axis=-1))
    #theta = 0.9
    #total_loss = (1-theta)*norm_loss + theta*inference_loss
    #total_loss = tf.cond(tf.equal(prediction[0],input_labels[0]), lambda: norm_loss, lambda: inference_loss)
    #prediction = tf.argmin(inference_loss, 0) #tf.argmax(logit[best_target], axis=1)
    #prediction_loss = tf.reduce_min(inference_loss)

    delta_grad = tf.gradients(inference_loss, input_deltas)[0]
    norm_grad = tf.gradients(norm_loss, input_deltas)[0]
    #total_grad = tf.gradients(total_loss, input_deltas)[0]

    #step_size = 0.5
    steps = 100

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, args.ckpt_file + args.ckpt_index)

    #with open('lfw_ids_1950000.csv', 'w') as f:
    #    f.write('image_id,image_name,person_name,id_prob\n')
    #count = 0
    #for image_id, image_name, person_name, image in zip(df.ImageId, df.ImageName, df.PersonName, original_images):
    #    print(count)
    #    count += 1
    #    start_index = 0
    #    id_losses = [0] * 85164
    #    while(start_index < args.num_output):
    #        probs = inference_loss.eval({input_images:[image], input_deltas:[start_deltas[0]], input_labels:list(range(start_index,start_index+1812)), dropout_rate:1.0}, session=sess)
    #        id_losses[start_index:start_index+1812] = probs
    #        start_index += 1812
    #    with open('lfw_ids_1950000.csv', 'a') as f:
    #        f.write('%s,%s,%s,%s\n' % (image_id,image_name,person_name,'_'.join(['%.4f'%x for x in id_losses])))
    #exit(0)

    limit_range = 25
    norm_cum = 0
    cum_num = 0
    for image_name, cur_attack_image, cur_attack_label, cur_start_delta in zip(df.ImageName, original_images, target_labels, start_deltas):
        if(os.path.exists(os.path.join('adv_images', image_name))):
            continue
        batch_adv_image = np.expand_dims(cur_attack_image.copy(), axis=0)
        
        #batch_target_labels = list([cur_attack_label])
        #loop_target_labels = list(lfw_ids.values())
        #loop_target_labels.remove(lfw_ids[image_name])
        loop_target_labels = best_target_ids[image_name]

        min_delta = np.ones((112,112,3), dtype=np.float32)*limit_range*0.0078125
        # max_norm = LA.norm(min_delta)
        for cur_target_label in loop_target_labels:
            delta = np.expand_dims(cur_start_delta.copy(), axis=0)
            #delta = np.zeros((1,112,112,3), dtype=np.float32)
            momentum = np.zeros((1,112,112,3), dtype=np.float32)
            vt = np.zeros((1,112,112,3), dtype=np.float32)
            mt = np.zeros((1,112,112,3), dtype=np.float32)
            alpha = 0.2
            beta = 0.8
            beta1 = 0.9
            beta2 = 0.99
            epsilon = 10e-8

            #first_target_loss = float('inf')
            #begin_to_focus_on_norm = False
            targeted = False
            for i in range(0, steps):
                #cur_adv_image = np.clip(batch_adv_image + delta, -1.0, 1.0)
                prediction_val = prediction.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                #prediction_val_flip = prediction.eval({input_deltas: np.flip(delta,axis=2), input_images: np.flip(batch_adv_image,axis=2),input_labels:[cur_target_label],dropout_rate:1.0}, session=sess)
                prediction_loss = inference_loss.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                
                print('Step @ %d : target: %d now: %d norm: %.4f target loss: %.4f' % (i, cur_target_label, prediction_val[0], np.mean(LA.norm(delta[0], axis=-1)), prediction_loss[0]))
                ori_targeted = (prediction_val[0] == cur_target_label)
                #flip_targeted = (prediction_val_flip[0] == cur_target_label)
                if(ori_targeted): #prediction_val_loss < target_loss/2.0 ):
                    targeted = True
                    break
                    #first_target_loss = prediction_loss[0]
                    #begin_to_focus_on_norm = True

                if(True): #not begin_to_focus_on_norm or prediction_loss[0] > first_target_loss):
                    gradient = delta_grad.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    #if(random.random() > 0.5):
                    #    gradient = delta_grad.eval({input_deltas: np.flip(delta,axis=2), input_images:np.flip(batch_adv_image,axis=2),input_labels:[cur_target_label], dropout_rate:1.0}, session=sess)
                else:
                    #if(not ori_targeted):
                    d_grad = delta_grad.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    #elif(not flip_targeted):
                    #    d_grad = delta_grad.eval({input_deltas: np.flip(delta,axis=2), input_images: np.flip(batch_adv_image,axis=2), input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    #else:
                    #    assert("How could be?")
                    #    exit(0)

                    n_grad = norm_grad.eval({input_deltas: delta, input_images: batch_adv_image, input_labels: [cur_target_label], dropout_rate:1.0}, session=sess)
                    # gradient for 0 postion will be nan
                    n_grad[np.isnan(n_grad)] = 0
                    #if(i % 3 == 0): #prediction_val[0] == cur_target_label):
                    #    gradient = 10 * n_grad
                    #else:
                    #    gradient = 0.1 * d_grad
                    gradient = 0.1*d_grad + 10*n_grad

                step_size = 0.15 #search_best_step_size(i, steps)
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

                next_delta = np.clip(next_delta, -limit_range*0.0078125, limit_range*0.0078125)

                #next_delta = np.clip(delta - step_size * gradient, -limit_range * 0.0078125, limit_range * 0.0078125)
                #delta = np.around(delta * 128) * 0.0078125
                delta = np.clip(batch_adv_image+next_delta,-0.99609375,0.99609375) - batch_adv_image
                #delta = np.around(delta * 128) * 0.0078125
                #if(np.array_equal(np.clip(batch_adv_image+delta,-1.0,1.0), batch_adv_image)):
                #    break

            # choose the best delta
            if(targeted and LA.norm(delta[0]) < LA.norm(min_delta)):
                min_delta = delta
        adv_image = np.clip((batch_adv_image + min_delta)*128 + 127.5, 0, 255)
        final_norm = np.mean(LA.norm(min_delta[0]*128, axis=-1))
        norm_cum += final_norm
        cum_num += 1
        print('Final norm:', final_norm)

        if(targeted):
            cv2.imwrite(os.path.join('adv_images', image_name.replace('jpg','png')), adv_image[0].astype(np.uint8))
            os.rename(os.path.join('adv_images', image_name.replace('jpg','png')), \
                    os.path.join('adv_images', image_name))
        #if(cum_num > 50):
        #    break
    print('Norm average: %.2f' % (norm_cum/cum_num))
