import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score
from keras.utils import to_categorical

def get_metrics_values(classifier, test_x, test_y,SVM=False,verbose=1):
    if SVM==True:
        y_pred = to_categorical(np.argmax(classifier.predict_proba(test_x), axis=-1), 2)
    else:
        y_pred = to_categorical(np.argmax(classifier.predict(test_x), axis=-1), 2)
    
    auroc= str(round(get_auroc(classifier,test_x,test_y,SVM=SVM),4))
    accuracy = str(round(accuracy_score(test_y, y_pred)*100,4))
    tn, fp, fn, tp=confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y_pred, axis=1)).ravel()
    TPR=tp/(tp+fn)
    if math.isnan(TPR):TPR=0
    TNR=tn/(tn+fp)
    if math.isnan(TNR):TNR=0
    PRE=tp/(tp+fp)
    if math.isnan(PRE):PRE=0
    F1=2*PRE*TPR/(PRE+TPR)
    if math.isnan(F1):F1=0
    if verbose==1:
        print('Accuracy: ' + accuracy)
        print('AUROC: ' + auroc)
        print('TPR: ' + str(TPR))
        print('TNR: ' + str(TNR))
        print('PRE: ' + str(PRE))
        print('F1: ' + str(F1))
    return y_pred,accuracy,auroc,str(round(TPR,4)),str(round(TNR,4)),str(round(PRE,4)),str(round(F1,4))


def plot_mse_mae_history(hist):
    history=hist.history
    x= range(1,len(history['val_loss'])+1)

    plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(x, history['mean_squared_error'],
           label='Train Error')
    plt.plot(x, history['val_mean_squared_error'],
           label = 'Val Error')
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(x, history['mean_absolute_error'],
           label='Train Error')
    plt.plot(x, history['val_mean_absolute_error'],
           label = 'Val Error')
    plt.legend()
    plt.show()
def get_auroc(model,test_x,test_y,SVM=False):
    y_test=[]
    for v in test_y:
        y_test.append(list(v).index(1))
    if SVM==True:
        y_pred = model.predict_proba(test_x)[:, 1]
    else:
        y_pred = model.predict(test_x)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def plot_roc_curve(model,test_x,test_y,SVM=False,**kw):
    y_test=[]
    for v in test_y:
        y_test.append(list(v).index(1))
    if SVM==True:
        y_pred = model.predict_proba(test_x)[:, 1]
    else:
        y_pred = model.predict(test_x)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=3, label='ROC curve (AUROC = %0.4f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_epoch(hist,loc=0,label='loss',**kw):
    history=np.transpose(hist)[loc]
    history=pd.DataFrame(history, columns=[label])
    history.plot()
    plt.show()

def plot_epoch_loss_w_test(hist_train,hist_test,loss_loc=0,**kw):
    history_train=np.transpose(hist_train)[loss_loc]
    history_test=np.transpose(hist_test)[loss_loc]
    df=pd.DataFrame(np.transpose(np.array([history_train,history_test])), columns=['train','test'])
    df.plot()
    plt.show()


def plot_epoch_acc_loss(hist,validation='off',**kw):
    if validation=='on':
        history = pd.DataFrame(hist, columns=['loss', 'accuracy','val_loss','val_accuracy'])
    else:
        history = pd.DataFrame(hist, columns=['loss', 'accuracy'])
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(history['loss'], 'y', label='train loss')
    if validation=='on':
        loss_ax.plot(history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='lower left', fancybox=True)

    acc_ax.plot(history['accuracy'], 'b', label='train acc')
    if validation=='on':
        acc_ax.plot(history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left', fancybox=True)
    plt.show()

# def plot_epoch_acc_loss(hist):
#     history=hist.history
#     fig, loss_ax = plt.subplots()
#     acc_ax = loss_ax.twinx()
#     loss_ax.plot(history['loss'], 'y', label='train loss')
#     loss_ax.plot(history['val_loss'], 'r', label='val loss')
#     loss_ax.set_xlabel('epoch')
#     loss_ax.set_ylabel('loss')
#     loss_ax.legend(loc='lower left', fancybox=True)
#
#     acc_ax.plot(history['acc'], 'b', label='train acc')
#     acc_ax.plot(history['val_acc'], 'g', label='val acc')
#     acc_ax.set_ylabel('accuracy')
#     acc_ax.legend(loc='upper left', fancybox=True)
#     plt.show()

# class AutoEncoder():
# self.inputs=inputs
        # self.encoding_dim=50
        # self.encoded = tf.keras.layers.Dense(300, activation=tf.nn.relu)(self.inputs)
        # # encoded = tf.keras.layers.Dense(100, activation=tf.nn.relu)(encoded)
        # self.z = tf.keras.layers.Dense(self.encoding_dim, activation=tf.nn.relu)(self.encoded)
        # # decoded = tf.keras.layers.Dense(100, activation=tf.nn.relu)(z)
        # self.decoded = tf.keras.layers.Dense(300, activation=tf.nn.relu)(self.z)
        # self.outputs = tf.keras.layers.Dense(int(self.inputs.shape[1]), activation=tf.nn.sigmoid)(self.decoded)

        # super(Encoder, self).__init__(self.inputs, self.outputs)
        # super(AutoEncoder, self).compile(loss=tf.keras.losses.mean_squared_error,
        #                                         optimizer=self.optimizer,
        #                                  metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])

        # self.encoder=tf.keras.models.Model(self.inputs,self.z)
        # encoded_input = tf.keras.layers.Input(shape=(self.encoding_dim,))
        # self.decoder = tf.keras.models.Model(encoded_input,self.outputs)