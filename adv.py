import tensorflow as tf
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
#from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
#import tensorlayer as tl
#from verification import ver_test


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file', default='./models/ckpt_model_d/',
                       type=str, help='the ckpt file path')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    parser.add_argument('--ckpt_index',
                        default='InsightFace_iter_best_710000.ckpt', help='ckpt file indexes')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # load lfw images
    root = '/data1/tabsun/AISecurity'
    image_dir = os.path.join(root, 'securityAI_round1_images')
    df = pd.read_csv(os.path.join(root,'securityAI_round1_dev.csv'))
    original_images = []
    original_labels = []
    target_labels = []
    # get 712 lfw images' id in this pretrained model
    lfw_df = pd.read_csv('lfw_ids.csv')
    lfw_ids = dict()
    for image_name, id_prob in zip(lfw_df.image_name, lfw_df.id_prob):
        prob = [float(x) for x in id_prob.split('_')]
        lfw_ids[image_name] = prob.index(min(prob))

    best_target_ids = dict()
    for image_name, id_prob in zip(lfw_df.image_name, lfw_df.id_prob):
        prob = [float(x) for x in id_prob.split('_')]
        valid_prob = [prob[i] for i in lfw_ids.values()]
        sorted_prob = sorted(valid_prob)
        best_target_ids[image_name] = prob.index(sorted_prob[1])

    #for i in range(1,713):
    #    key = '%05d.jpg' % i
    #    print('%s: %d' % (key, best_target_ids[key]))
    #exit(0)

    for image_id, image_name in zip(df.ImageId, df.ImageName):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image = (image - 127.5) * 0.0078125
        original_images.append(image)
        original_labels.append(int(image_id)-1)
        target_image_name = None
        while(True):
            target_image_name = '%05d.jpg' % (int(random.random()*713*3) % 713)
            if(target_image_name != image_name and target_image_name in lfw_ids.keys()):
                break
        target_label = lfw_ids[target_image_name]
        target_labels.append(target_label)

    w, h = args.image_size
    images = tf.placeholder(name='img_inputs', shape=[None, h, w, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
    embedding_tensor = net.outputs
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)
    #logit = tf.squeeze(logit, axis=0)
    prediction = tf.argmax(logit, axis=1)
    inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
    #gt_index = tf.argmin(inference_loss, 1)

    img_grad = tf.gradients(inference_loss, images)[0]

    step_size = 0.5
    steps = 500

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, args.ckpt_file + args.ckpt_index)

    #with open('lfw_ids.csv', 'w') as f:
    #    f.write('image_id,image_name,person_name,id_prob\n')
    #count = 0
    #for image_id, image_name, person_name, image in zip(df.ImageId, df.ImageName, df.PersonName, original_images):
    #    print(count)
    #    count += 1
    #    start_index = 0
    #    min_value = float('inf')
    #    gt = None
    #    id_losses = [0] * 85164
    #    while(start_index < args.num_output):
    #        probs = inference_loss.eval({images:[image], labels:list(range(start_index,start_index+1812)), dropout_rate:1.0}, session=sess)
    #        id_losses[start_index:start_index+1812] = probs
    #        #cur = np.min(probs)
    #        #if(cur < min_value):
    #        #    min_value = cur
    #        #    gt = start_index + np.where(probs==cur)[0]
    #        start_index += 1812
    #    with open('lfw_ids.csv', 'a') as f:
    #        f.write('%s,%s,%s,%s\n' % (image_id,image_name,person_name,'_'.join(['%.4f'%x for x in id_losses])))
    #exit(0)


    for image_name, cur_attack_image, cur_attack_label in zip(df.ImageName, original_images, target_labels):
        batch_adv_image = np.expand_dims(cur_attack_image, axis=0).copy()
        batch_target_labels = list([cur_attack_label])

        delta = np.zeros((1,112,112,3), dtype=np.float32)
        for i in range(0, steps):
            cur_adv_image = np.clip(batch_adv_image + delta, -1.0, 1.0)
            prediction_val = prediction.eval({images: cur_adv_image, labels: batch_target_labels, dropout_rate:1.0}, session=sess)
            print('Step @ %d : target: %d now: %d' % (i, batch_target_labels[0], prediction_val[0]))
            if(prediction_val[0] == batch_target_labels[0]):
                break
            gradient = img_grad.eval({images: cur_adv_image, labels: batch_target_labels, dropout_rate:1.0}, session=sess)
            if(np.array_equal(np.clip(delta - step_size*gradient, -25.5 * 0.0078125, 25.5 * 0.0078125), delta)):
                break
            delta = np.clip(delta - step_size * gradient, -25.5 * 0.0078125, 25.5 * 0.0078125)
            if(np.array_equal(np.clip(batch_adv_image+delta,-1.0,1.0), batch_adv_image)):
                break
        adv_image = np.clip((batch_adv_image + delta)*128 + 127.5, 0, 255)
        cv2.imwrite(os.path.join('adv_images', image_name), adv_image[0].astype(np.uint8))
