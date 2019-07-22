import tensorflow as tf
from tensorflow import identity
from tensorflow.nn import conv2d, dropout, relu, sigmoid, softmax, bidirectional_dynamic_rnn
from tensorflow.nn.rnn_cell import LSTMCell, DropoutWrapper

####### Character Embedding Layer #######
def CharEmbLayer(name,inputs,filter_shape,output_channel_num,stride,padding,init,reuse=False,is_training=True):
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        cnn_filter = tf.get_variable(name='cnn_filter',shape=filter_shape, dtype='float',initializer=init)
        cnn_bias = tf.get_variable(name='cnn_bias',shape=[output_channel_num], dtype='float',initializer=init)
        if is_training==True:
            _x=dropout(inputs,keep_prob=0.8)
        else:
            _x=identity(inputs)
        cnn_x = conv2d(_x, cnn_filter, stride, padding)
        cnn_mask = tf.cast(tf.not_equal(cnn_x,0.),'float32')
        cnn_x = tf.reduce_max(relu(tf.multiply(cnn_x+cnn_bias,cnn_mask)),2)
    return cnn_x

####### Word Embedding Layer #######
def WordEmbLayer(name,wordVec, charVec):
    return tf.concat(name=name,values=[wordVec,charVec],axis=-1)


def DenseLayer(name,inputs,output_node_num,activation,weight_init,bias_init,reuse=False,is_training=True):
    input_node_num=inputs.get_shape()[-1]
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        W = tf.get_variable(name="W", shape=[input_node_num,output_node_num],
                            dtype='float',initializer=weight_init)
        b = tf.get_variable(name="b", shape=[output_node_num],
                            dtype='float',initializer=bias_init)
    return activation(tf.add(tf.matmul(inputs,W),b),name=name)


####### Highway Network Layer #######
def HighwayLayer(name,inputs,weight_init,bias_init,reuse=False,is_training=True):
    original_shape=inputs.get_shape()
    output_node_num=original_shape[-1]
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        _inputs=tf.reshape(inputs,[-1,output_node_num])
        carry=DenseLayer('carry_gate',_inputs,output_node_num,sigmoid,weight_init,bias_init,reuse=reuse,is_training=is_training)
        transform=DenseLayer('transform_gate',_inputs,output_node_num,relu,weight_init,bias_init,reuse=reuse,is_training=is_training)
        highway=tf.add(tf.multiply(tf.subtract(1.,carry),transform),tf.multiply(carry,_inputs))
        _highway_mask = tf.cast(tf.not_equal(_inputs,0.),'float32')
        highway=tf.reshape(tf.multiply(highway,_highway_mask),original_shape)
    return highway

####### Contextual Embedding Layer #######
def ContextualEmbLayer(name,inputs,d,sequence_length,reuse=False,is_training=True):
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        cell = LSTMCell(d)
        cell = DropoutWrapper(cell, output_keep_prob=0.8)
        (_fw,_bw),_ = bidirectional_dynamic_rnn(cell,cell,inputs,sequence_length,dtype ='float32')
    return tf.concat([_fw,_bw],-1)

####### Attnetion Layer #######
def AttentionLayer(name,inputs,d,weight_init,reuse=False,is_training=True):
    h,u=inputs
    bc,mlq,_=u.get_shape() # bc: batch_size, mlq: max_word_num_q, d: hidden_size
    mlx=h.get_shape()[1] # mlx: max_word_num_x
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        hh=tf.tile(h,[1,1,mlq])
        hh=tf.reshape(hh,[bc,-1])
        hh=tf.reshape(hh,[bc,mlq*mlx,2*d])
        uu=tf.tile(u,[1,mlx,1])
        uu=tf.reshape(uu,[bc,mlq*mlx,2*d])
        element_mul=tf.multiply(hh,uu)
        vec_6d=tf.reshape(tf.concat([hh,uu,element_mul],-1),[-1,6*d])
        with tf.variable_scope(name_or_scope='sim_mat',reuse=reuse):
            W_s = tf.get_variable(name="W_sim_mat", shape=[6*d,1], dtype='float',initializer=weight_init)
            S=tf.matmul(vec_6d,W_s)
            S=tf.reshape(S,[bc,mlx,mlq])
        ### context2query ###
        a=softmax(S,axis=2)
        U=tf.matmul(a,u)
        ### query2context ###
        b=softmax(tf.reduce_max(S,-1))
        b=tf.expand_dims(b,-1)
        _H=tf.reduce_sum(tf.multiply(b,h),axis=1)
        H_t=tf.tile(_H,[1,mlx])
        H=tf.reshape(H_t,[bc,mlx,2*d])
        ### Linked attention ###
        mul_h_U=tf.multiply(h,U)
        mul_h_H=tf.multiply(h,H)
        G=tf.concat([h,U,mul_h_U,mul_h_H],2)
    return G


####### Modeling Layer #######
def TwoLSTMs_ModelingLayer(name,inputs,d,sequence_length,reuse=False,is_training=True):
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        cell_1 = LSTMCell(d)
        cell_1 = DropoutWrapper(cell_1, output_keep_prob=0.8)
        cell_2 = LSTMCell(d)
        cell_2 = DropoutWrapper(cell_2, output_keep_prob=0.8)
        (_fw_1,_bw_1),_ = bidirectional_dynamic_rnn(cell_1,cell_1,inputs,sequence_length,dtype ='float32',scope='lstm1')
        _M=tf.concat([_fw_1,_bw_1],-1)
        (_fw_2,_bw_2),_ = bidirectional_dynamic_rnn(cell_2,cell_2,_M,sequence_length,dtype ='float32',scope='lstm2')
    return tf.concat([_fw_2,_bw_2],-1)


####### Output Layer #######
def OutputLayer(name,inputs,d,sequence_length,weight_init,reuse=False,is_training=True):
    G,M=inputs
    bc,mlx,_=G.get_shape() # bc: batch_size, mlx: max_word_num_x, d: hidden_size
    with tf.variable_scope(name_or_scope=name+'_Out1',reuse=reuse):
        G_M = tf.concat([G, M], -1)
        with tf.variable_scope(name_or_scope='output', reuse=reuse):
            W_p1 = tf.get_variable(name="W_p1", shape=[10 * d, 1], dtype='float', initializer=weight_init)
        G_M = tf.reshape(G_M, [-1, 10 * d])
        _mask = tf.cast(tf.sequence_mask(sequence_length, mlx), 'float')
        logit1 = tf.reshape(tf.matmul(G_M, W_p1), [bc, mlx])
        logit1 = tf.multiply(logit1, _mask)
        p1 = softmax(logit1)

    with tf.variable_scope(name_or_scope=name+'_Out2',reuse=reuse):
        cell_m2 = LSTMCell(d)
        cell_m2 = DropoutWrapper(cell_m2, output_keep_prob=0.8)
        with tf.variable_scope(name_or_scope='M2', reuse=reuse):
            (_fw_m2, _bw_m2), _ = bidirectional_dynamic_rnn(cell_m2, cell_m2, M, sequence_length, dtype='float32', scope='lstm')
        M2 = tf.concat([_fw_m2, _bw_m2], -1)
        G_M2 = tf.concat([G, M2], -1)
        with tf.variable_scope(name_or_scope='output', reuse=reuse):
            W_p2 = tf.get_variable(name="W_p2", shape=[10 * d, 1], dtype='float', initializer=weight_init)
        G_M2 = tf.reshape(G_M2, [-1, 10 * d])
        logit2 = tf.reshape(tf.matmul(G_M2, W_p2), [bc, mlx])
        logit2 = tf.multiply(logit2, _mask)
        p2 = softmax(logit2)
    return logit1,logit2,p1,p2