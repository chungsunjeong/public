import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


class MyKeras(tf.keras.Model):
    def __init__(self, inputs, outputs):
        super(MyKeras, self).__init__(inputs, outputs)

    def fit(self, **kw):
        self.hist = super(MyKeras, self).fit(**kw)

    def get_info(self):
        print('------------------------ GET Model Info. ------------------------')
        print('Current class and base class(es):\t')
        class_list = []
        for base in self.__class__.mro():
            class_list.append(base.__name__)
        print(' < '.join(class_list))
        print('\nSummary my model:')
        self.summary()

    def plots_epoch_acc_loss(self):
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(self.hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(self.hist.history['val_loss'], 'r', label='val loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='lower left', fancybox=True)

        acc_ax.plot(self.hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(self.hist.history['val_acc'], 'g', label='val acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend(loc='upper left', fancybox=True)
        plt.show()

    def plot_roc_curve(self,test_x,test_y):
        y_test=[]
        for v in test_y:
            y_test.append(list(v).index(1))
        y_pred = self.predict(test_x)[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=3, label='ROC curve (AUROC = %0.2f)' % roc_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

class LinearClassifier(MyKeras):
    def __init__(self, inputs, optimizer=tf.keras.optimizers.Adam()):
        outputs = tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)(inputs)
        super(LinearClassifier, self).__init__(inputs, outputs)
        self.optimizer = optimizer
        super(LinearClassifier, self).compile(loss=tf.keras.losses.categorical_crossentropy,
                                                optimizer=self.optimizer, metrics=['accuracy'])

class LogisticRegression(MyKeras):
    def __init__(self, inputs, optimizer=tf.keras.optimizers.Adam()):
        outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(inputs)
        super(LogisticRegression, self).__init__(inputs, outputs)
        self.optimizer = optimizer
        super(LogisticRegression, self).compile(loss=tf.keras.losses.categorical_crossentropy,
                                                optimizer=self.optimizer, metrics=['accuracy'])

class MultiLayerPerceptron(MyKeras):
    def __init__(self, inputs, optimizer=tf.keras.optimizers.Adam()):
        x = tf.keras.layers.Dense(500, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)
        super(MultiLayerPerceptron, self).__init__(inputs, outputs)
        self.optimizer = optimizer
        super(MultiLayerPerceptron, self).compile(loss=tf.keras.losses.categorical_crossentropy,
                                                optimizer=self.optimizer, metrics=['accuracy'])
