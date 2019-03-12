import tensorflow as tf

class MyTensorflow(object):
    def __init__(self, session, config, is_training=False):
        self.config = config
        self._session = session
        self._batch_size = config.Train['batch_size']
        self._training_iteration = config.Train['training_iteration']
        self._learning_rate = config.Train['learning_rate']
        self._input_size = [None, config.Model['feature_size']]
        self._output_size = [None, config.Model['n_class']]
        self._feature_size=config.Model['feature_size']
        self._n_class=config.Model['n_class']

    def train(self, DTI):
        avg_cost = 0.
        batch_xs, batch_ys = DTI.train.next_batch(self._batch_size, DTI.pos_neg_label)
        self._session.run(self._optimize, feed_dict={self._x: batch_xs, self._y: batch_ys})
        avg_cost += self._session.run(self._loss, feed_dict={self._x: batch_xs, self._y: batch_ys}) / self._batch_size
        return avg_cost

    def get_accuracy(self, DTI):
        test_x, test_y = DTI.test.next_batch(DTI.test.data.shape[0], DTI.pos_neg_label, one_hot_encoding=True)
        ac = self._session.run(self._accuracy, feed_dict={self._x: test_x, self._y: test_y})
        return ac


class mymodel(MyTensorflow):
    def __init__(self, session, config, is_training=False):
        super(mymodel, self).__init__(session, config, is_training)
        self.build_graph()

    def loss(self, true_logits, predicted_logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predicted_logits, labels=true_logits))

    def optimize(self, loss, optimizer=tf.train.AdamOptimizer()):
        optimizer.learning_rate = self._learning_rate
        return optimizer.minimize(loss)

    def build_graph(self):
        x = tf.placeholder(tf.float32, self._input_size)
        y = tf.placeholder(tf.float32, self._output_size)
        W = tf.Variable(tf.zeros([self._feature_size, self._n_class]), name='W')
        b = tf.Variable(tf.zeros([self._n_class]), name='b')
        y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

        _loss = self.loss(y, y_pred)
        _optimize = self.optimize(_loss, optimizer=self.config.Train['optimizer'])

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype='float32'))

        self._x = x
        self._y = y
        self._W = W
        self._b = b
        self._y_pred = y_pred
        self._loss = _loss
        self._optimize = _optimize
        self._correct_prediction = correct_prediction
        self._accuracy = accuracy
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
