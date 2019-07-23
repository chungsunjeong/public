import tensorflow as tf

from tensorflow.nn import conv2d, dropout, relu, sigmoid, softmax


def DenseLayer(name,inputs,output_node_num,activation,weight_init,bias_init,reuse=False,is_training=True):
    input_node_num=inputs.get_shape()[-1]
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        W = tf.get_variable(name="W", shape=[input_node_num,output_node_num],
                            dtype='float',initializer=weight_init)
        b = tf.get_variable(name="b", shape=[output_node_num],
                            dtype='float',initializer=bias_init)
    return activation(tf.add(tf.matmul(inputs,W),b),name=name)

def LogisticRegression(name,inputs,num_classes,weight_init,bias_init,output_type='logits',reuse=False,is_training=True):
    if output_type not in ['prediction','logits']:
        raise ValueError('Specify the output node type: prediction (or) logits')
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        if output_type=='prediction':
            out = DenseLayer("FC_softmax",inputs,num_classes,softmax,weight_init,bias_init,reuse=reuse,is_training=is_training)
        elif output_type=='logits':
             out = DenseLayer("FC_identity",inputs,num_classes,tf.identity,weight_init,bias_init,reuse=reuse,is_training=is_training)
    return out
                     