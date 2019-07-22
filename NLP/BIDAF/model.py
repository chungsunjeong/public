import tensorflow as tf
from tensorflow.nn import softmax_cross_entropy_with_logits
from layer import CharEmbLayer,WordEmbLayer,HighwayLayer,\
    ContextualEmbLayer,AttentionLayer,TwoLSTMs_ModelingLayer,OutputLayer


class my_QAmodel(object):
    def __init__(self, config):
        self.config = config
        self._max_word_num_x = config['max_word_num_x']
        self._max_word_num_q = config['max_word_num_q']
        self._max_word_len = config['max_word_len']
        self._word_vec_dim = config['word_vec_dim']
        self._d = config['hidden_size']
        self._batch_size = config['batch_size']
        self._char_vec_dim = config['char_vec_dim']
        self._is_training = config['is_training ']
        self._cnn_height = config['cnn_height']
        self._cnn_stride = config['cnn_stride']
        self._cnn_padding = config['cnn_padding']
        self._d = config['hidden_size']
        self._cnn_output_channel_dim = config['cnn_output_channel_num']
        self.xw = tf.placeholder(dtype='float32', shape=[self._batch_size, self._max_word_num_x, self._word_vec_dim],
                                 name='x_word')
        self.xc = tf.placeholder(dtype='float32',
                                 shape=[self._batch_size, self._max_word_num_x, self._max_word_len, self._char_vec_dim],
                                 name='x_char')
        self.qw = tf.placeholder(dtype='float32', shape=[self._batch_size, self._max_word_num_q, self._word_vec_dim],
                                 name='q_word')
        self.qc = tf.placeholder(dtype='float32',
                                 shape=[self._batch_size, self._max_word_num_q, self._max_word_len, self._char_vec_dim],
                                 name='q_char')
        self.len_x = tf.placeholder(dtype='int32', shape=[self._batch_size], name='word_len_x')
        self.len_q = tf.placeholder(dtype='int32', shape=[self._batch_size], name='word_len_q')
        self.y1 = tf.placeholder(dtype='bool', shape=[self._batch_size, self._max_word_num_x], name='y_start')
        self.y2 = tf.placeholder(dtype='bool', shape=[self._batch_size, self._max_word_num_x], name='y_end')
        self._build_graph()
        self._build_loss()

    def _initializer(self):
        cnn_init = self.config['initializer']['cnn']
        weight_init, bias_init = self.config['initializer']['dense']
        return cnn_init, [weight_init, bias_init]

    def _build_graph(self):
        cnn_init, [weight_init, bias_init] = self._initializer()
        d = self._d
        input_channel_num = self._char_vec_dim
        output_channel_num = self._cnn_output_channel_dim
        stride = self._cnn_stride
        padding = self._cnn_padding
        is_training = self._is_training
        filter_shape = [1, self._cnn_height, input_channel_num, output_channel_num]
        cnn_xc = CharEmbLayer('CNN', self.xc, filter_shape, output_channel_num, stride, padding, cnn_init,
                              is_training=is_training)
        cnn_qc = CharEmbLayer('CNN', self.qc, filter_shape, output_channel_num, stride, padding, cnn_init, reuse=True,
                              is_training=is_training)

        emb_x = WordEmbLayer('WordEmb_x', self.xw, cnn_xc)
        emb_q = WordEmbLayer('WordEmb_q', self.qw, cnn_qc)

        highway_x = HighwayLayer('Highway', emb_x, weight_init, bias_init, is_training=is_training)
        highway_q = HighwayLayer('Highway', emb_q, weight_init, bias_init, reuse=True, is_training=is_training)

        h = ContextualEmbLayer('ContEmb', highway_x, d, self.len_x, is_training=is_training)
        u = ContextualEmbLayer('ContEmb', highway_q, d, self.len_q, reuse=True, is_training=is_training)

        G = AttentionLayer('Attention', [h, u], d, weight_init, is_training=is_training)

        M = TwoLSTMs_ModelingLayer('Modeling', G, d, self.len_x, is_training=is_training)
        logit1, logit2, p1, p2 = OutputLayer('Output', [G, M], d, self.len_x, weight_init, is_training=is_training)

        self.cnn_xc = cnn_xc
        self.cnn_qc = cnn_qc
        self.emb_x = emb_x
        self.emb_q = emb_q
        self.highway_x = highway_x
        self.highway_q = highway_q
        self.h = h
        self.u = u
        self.G = G
        self.M = M
        self.logit1 = logit1
        self.logit2 = logit2
        self.p1 = p1
        self.p2 = p2

    def _build_loss(self):
        loss1 = tf.reduce_mean(softmax_cross_entropy_with_logits(labels=self.y1, logits=self.logit1))
        loss2 = tf.reduce_mean(softmax_cross_entropy_with_logits(labels=self.y2, logits=self.logit2))
        self.loss = loss1 + loss2

    def reset_graph(self):
        tf.reset_default_graph()
        self.__init__(self.config)

    def update_graph(self, new_config):
        tf.reset_default_graph()
        self.__init__(new_config)

    def get_feed_dict(self, batch_dict):
        xw, xc, qw, qc, len_x, len_q, y1, y2 = self.xw, self.xc, self.qw, self.qc, self.len_x, self.len_q, self.y1, self.y2
        feed = {xw: batch_dict['xw'], xc: batch_dict['xc'], qw: batch_dict['qw'],
                qc: batch_dict['qc'], len_x: batch_dict['x_len'], len_q: batch_dict['q_len'],
                y1: batch_dict['y_start'], y2: batch_dict['y_end']}
        return feed