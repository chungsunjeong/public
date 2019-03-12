import time
import collections
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from model import keras_models
from module_DTI import my_dataset
start_time=time.time()

config=collections.namedtuple('config',['Dataset','Model','Train','Save'])

def main():
    '''Data Loading'''
    dict_directories = {'dir_ROOT': 'C:\\Users\\csjeong\\Desktop\\research\\dataset\\conv_DTI\\2012, Tabei', }
    dict_directories.update({
        'DTI_adjmat': dict_directories['dir_ROOT'] + '\\inter_admat.txt',
        'drug': dict_directories['dir_ROOT'] + '\\drug_repmat.txt',
        'target': dict_directories['dir_ROOT'] + '\\target_repmat.txt'
    })
    config_Dataset = {
        'dict_directories': dict_directories,
        'neg_to_pos_ratio': 1,
        'split_ratio': 0.95
    }

    DTI = my_dataset.DTI_Dataset(config_Dataset['dict_directories'])
    DTI.load_data()
    DTI.load_pos_neg_samples(config_Dataset['neg_to_pos_ratio'])
    DTI.split_train_test_set(config_Dataset['split_ratio'])

    '''Model Building'''
    config_Model={
        'feature_size':DTI.train.data.shape[1],
        'n_class':2
    }

    '''Model Training'''
    config_Train = {
        'batch_size': 500,
        'learning_rate': 0.001,
        'epoch': 40,
        'validation_split': 0.1
    }
    config_Train['training_iteration']=int(config_Train['epoch'] * DTI.train.data.shape[0]
                                           / config_Train['batch_size']+0.9999)
    config_Train['optimizer']=tf.keras.optimizers.Adam(config_Train['learning_rate'])

    '''Model Saver'''
    config_Save = {
        'checkpoint_path': './training_checkpoints/keras.ckpt',
        'save_model_path': './saved_model/keras.hdf5'
    }
    config_Save['checkpoint'] = ModelCheckpoint(filepath=config_Save['checkpoint_path'], monitor='val_acc', verbose=0,
                                                save_weights_only=True,
                                                period=5)
    config.Dataset=config_Dataset
    config.Model=config_Model
    config.Train=config_Train
    config.Save=config_Save

    n_train = DTI.train.data.shape[0]
    n_test = DTI.test.data.shape[0]
    train_x, train_y = DTI.train.next_batch(batch_size=n_train, pos_neg_label=DTI.pos_neg_label)
    test_x, test_y = DTI.test.next_batch(batch_size=n_test, pos_neg_label=DTI.pos_neg_label, one_hot_encoding=True)

    inputs = tf.keras.Input(shape=(config.Model['feature_size'],))
    m = keras_models.LogisticRegression(inputs=inputs,optimizer=config.Train['optimizer'])
    m.get_info()

    m.fit(x=train_x, y=train_y, batch_size=config.Train['batch_size'], epochs=config.Train['epoch'], verbose=0,callbacks=[config.Save['checkpoint']],
          validation_split=config.Train['validation_split'], shuffle=True)

    loss, acc = m.evaluate(test_x, test_y)
    print('loss of test set: %.3f' % (loss))
    print('Acc. of test set: %.3f %%' % (acc * 100))
    m.plots_epoch_acc_loss()
    m.plot_roc_curve(test_x,test_y)
    print('epoch:\t' + str(config.Train['epoch']))


if __name__=='__main__':
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
