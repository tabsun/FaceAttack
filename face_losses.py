import tensorflow as tf
import math

def single_multi_loss(anchor, negative):
    with tf.variable_scope('multi_single_loss'):
        anchor = tf.nn.l2_normalize(anchor)
        negative = tf.nn.l2_normalize(negative, axis=-1)
        negative = tf.reduce_mean(negative, axis=0, keep_dims=True)
        dist = tf.reduce_sum(tf.multiply(anchor, negative))
    return dist

def single_loss(anchor, negative):
    with tf.variable_scope('single_loss'):
        anchor = tf.nn.l2_normalize(anchor)
        negative = tf.nn.l2_normalize(negative, axis=-1)
        neg_dist = tf.reduce_sum(tf.multiply(anchor, negative))
        
    return neg_dist

def my_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        anchor = tf.nn.l2_normalize(anchor)
        positive = tf.nn.l2_normalize(positive)
        negative = tf.nn.l2_normalize(negative)
        cosp = tf.minimum(1., tf.reduce_sum(tf.multiply(anchor, positive)))
        cosn = tf.minimum(1., tf.reduce_sum(tf.multiply(anchor, negative)))
        sinp = tf.sqrt(1-cosp*cosp)
        sinn = tf.sqrt(1-cosn*cosn)

        cond_v = cosn - cosp
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        loss = tf.where(cond, 0.877-cosp*cosn-sinn*sinp, cosp*cosn+sinn*sinp-0.877)

    return loss #(0.877 - cosp*cosn - sinn*sinp)*(cosn - cosp)

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        anchor = tf.nn.l2_normalize(anchor)
        positive = tf.nn.l2_normalize(positive)
        negative = tf.nn.l2_normalize(negative)
        pos_dist = 1. - tf.reduce_sum(tf.multiply(anchor, positive))
        neg_dist = 1. - tf.reduce_sum(tf.multiply(anchor, negative))
       
        basic_loss = tf.subtract(pos_dist,neg_dist)

        #loss = tf.maximum(basic_loss, 0.0)
      
    return basic_loss

def arcface_loss(embedding, labels, out_num, scope=None, w_init=None, s=64., m=0.5, reuse=False):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    if(scope is None):
        scope = 'arcface_loss'
    embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
    embedding = tf.div(embedding, embedding_norm)
    with tf.variable_scope(scope, reuse=reuse):
        # inputs and weights norm
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')

    # cos(theta+m)
    cos_t = tf.matmul(embedding, weights) # None x 85162
    cos_t2 = tf.square(cos_t)
    sin_t2 = tf.subtract(1., cos_t2)
    sin_t = tf.sqrt(sin_t2)
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m))

    # this condition controls the theta+m should in range [0, pi]
    #      0<=theta+m<=pi
    #     -m<=theta<=pi-m
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)

    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)

    mask = tf.one_hot(labels, depth=out_num) # 85164 x 85164
    inv_mask = tf.subtract(1., mask)

    s_cos_t = tf.multiply(s, cos_t) # 1 x 85164
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask))
    return output


def batch_arcface_loss(embedding, labels, out_num, labels_num, scope=None, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    if(scope is None):
        scope = 'arcface_loss'
    with tf.variable_scope(scope):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t') # None x 85162
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask') # 712 x 85164
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t') # 1 x 85164
        repeat = tf.constant([labels_num, 1])
        s_cos_t_full = tf.reshape(tf.tile(s_cos_t, repeat), [tf.shape(s_cos_t)[0], labels_num, tf.shape(s_cos_t)[1]])
        cos_mt_temp_full = tf.reshape(tf.tile(cos_mt_temp, repeat), [tf.shape(cos_mt_temp)[0], labels_num, tf.shape(cos_mt_temp)[1]])

        output = tf.add(tf.multiply(s_cos_t_full, inv_mask), tf.multiply(cos_mt_temp_full, mask), name='arcface_loss_output')
    return output


#def arcface_loss_c(embedding, labels, out_num, w_init=None, s=64., m=0.5):
#    '''
#    :param embedding: the input embedding vectors
#    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
#    :param s: scalar value default is 64
#    :param out_num: output class num
#    :param m: the margin value, default is 0.5
#    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
#    '''
#    cos_m = math.cos(m)
#    sin_m = math.sin(m)
#    mm = sin_m * m  # issue 1
#    threshold = math.cos(math.pi - m)
#    with tf.variable_scope('arcface_loss_1950000'):
#        # inputs and weights norm
#        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
#        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
#        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
#                                  initializer=w_init, dtype=tf.float32)
#        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
#        weights = tf.div(weights, weights_norm, name='norm_weights')
#        # cos(theta+m)
#        cos_t = tf.matmul(embedding, weights, name='cos_t') # None x 85162
#        cos_t2 = tf.square(cos_t, name='cos_2')
#        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
#        sin_t = tf.sqrt(sin_t2, name='sin_t')
#        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
#
#        # this condition controls the theta+m should in range [0, pi]
#        #      0<=theta+m<=pi
#        #     -m<=theta<=pi-m
#        cond_v = cos_t - threshold
#        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
#
#        keep_val = s*(cos_t - mm)
#        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
#
#        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask') # 85164 x 85164
#        inv_mask = tf.subtract(1., mask, name='inverse_mask')
#
#        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t') # 1 x 85164
#        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
#    return output
#
#
#def batch_arcface_loss_c(embedding, labels, out_num, labels_num, w_init=None, s=64., m=0.5):
#    '''
#    :param embedding: the input embedding vectors
#    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
#    :param s: scalar value default is 64
#    :param out_num: output class num
#    :param m: the margin value, default is 0.5
#    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
#    '''
#    cos_m = math.cos(m)
#    sin_m = math.sin(m)
#    mm = sin_m * m  # issue 1
#    threshold = math.cos(math.pi - m)
#    with tf.variable_scope('arcface_loss_1950000'):
#        # inputs and weights norm
#        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
#        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
#        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
#                                  initializer=w_init, dtype=tf.float32)
#        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
#        weights = tf.div(weights, weights_norm, name='norm_weights')
#        # cos(theta+m)
#        cos_t = tf.matmul(embedding, weights, name='cos_t') # None x 85162
#        cos_t2 = tf.square(cos_t, name='cos_2')
#        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
#        sin_t = tf.sqrt(sin_t2, name='sin_t')
#        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
#
#        # this condition controls the theta+m should in range [0, pi]
#        #      0<=theta+m<=pi
#        #     -m<=theta<=pi-m
#        cond_v = cos_t - threshold
#        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
#
#        keep_val = s*(cos_t - mm)
#        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
#
#        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask') # 712 x 85164
#        inv_mask = tf.subtract(1., mask, name='inverse_mask')
#
#        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t') # 1 x 85164
#        repeat = tf.constant([labels_num, 1])
#        s_cos_t_full = tf.reshape(tf.tile(s_cos_t, repeat), [tf.shape(s_cos_t)[0], labels_num, tf.shape(s_cos_t)[1]])
#        cos_mt_temp_full = tf.reshape(tf.tile(cos_mt_temp, repeat), [tf.shape(cos_mt_temp)[0], labels_num, tf.shape(cos_mt_temp)[1]])
#
#        output = tf.add(tf.multiply(s_cos_t_full, inv_mask), tf.multiply(cos_mt_temp_full, mask), name='arcface_loss_output')
#    return output


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output


def combine_loss_val(embedding, labels, w_init, out_num, margin_a, margin_m, margin_b, s):
    '''
    This code is contributed by RogerLo. Thanks for you contribution.

    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                              initializer=w_init, dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
    cos_t = tf.matmul(embedding_unit, weights_unit)
    ordinal = tf.constant(list(range(0, embedding.get_shape().as_list()[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, labels], axis=1)
    zy = cos_t * s
    sel_cos_t = tf.gather_nd(zy, ordinal_y)
    if margin_a != 1.0 or margin_m != 0.0 or margin_b != 0.0:
        if margin_a == 1.0 and margin_m == 0.0:
            s_m = s * margin_b
            new_zy = sel_cos_t - s_m
        else:
            cos_value = sel_cos_t / s
            t = tf.acos(cos_value)
            if margin_a != 1.0:
                t = t * margin_a
            if margin_m > 0.0:
                t = t + margin_m
            body = tf.cos(t)
            if margin_b > 0.0:
                body = body - margin_b
            new_zy = body * s
    updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), zy.get_shape()))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
    predict_cls = tf.argmax(updated_logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    predict_cls_s = tf.argmax(zy, 1)
    accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls_s, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    return zy, loss, accuracy, accuracy_s, predict_cls_s
