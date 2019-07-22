import numpy as np
from load_dataset import get_preprocessed_dataset, load_json_file
import tensorflow as tf
from model import my_QAmodel
import datetime, time


def get_batch_dict(indexes,config,dataset,word2vec,char2vec):
    xw_batch=np.zeros([config['batch_size'],config['max_word_num_x'],config['word_vec_dim']],dtype='float32')
    xc_batch=np.zeros([config['batch_size'],config['max_word_num_x'],config['max_word_len'],config['char_vec_dim']],dtype='float32')
    qw_batch=np.zeros([config['batch_size'],config['max_word_num_q'],config['word_vec_dim']],dtype='float32')
    qc_batch=np.zeros([config['batch_size'],config['max_word_num_q'],config['max_word_len'],config['char_vec_dim']],dtype='float32')
    y1_batch = np.zeros([config['batch_size'], config['max_word_num_x']], dtype='float32')
    y2_batch = np.zeros([config['batch_size'], config['max_word_num_x']], dtype='float32')
    feed_dict=dict()
    feed_dict['xw']=xw_batch
    feed_dict['xc']=xc_batch
    feed_dict['qw']=qw_batch
    feed_dict['qc']=qc_batch
    feed_dict['x_len']=[]
    feed_dict['q_len']=[]
    feed_dict['y_start']=y1_batch
    feed_dict['y_end'] = y2_batch
    for sample,qInd in enumerate(indexes):
        context=dataset['x_dataset'][dataset['qInd2xInd'][qInd]]
        context_c=dataset['x_char_dataset'][dataset['qInd2xInd'][qInd]]
        query=dataset['q_dataset'][qInd]
        query_c=dataset['q_char_dataset'][qInd]
        feed_dict['q_len'].append(dataset['word_len_query'][qInd])
        feed_dict['x_len'].append(dataset['word_len_context'][dataset['qInd2xInd'][qInd]])
        y1_batch[sample][dataset['y_start_dataset'][qInd]] = 1
        y2_batch[sample][dataset['y_end_dataset'][qInd]] = 1
        for i,wordInd in enumerate(context):
            xw_batch[sample][i]=word2vec['wordInd2wordVec'][wordInd]
            for j,charInd in enumerate(context_c[i]):
                xc_batch[sample][i][j]=char2vec['charInd2charVec'][charInd]
        for i,wordInd in enumerate(query):
            qw_batch[sample][i]=word2vec['wordInd2wordVec'][wordInd]
            for j,charInd in enumerate(query_c[i]):
                qc_batch[sample][i][j]=char2vec['charInd2charVec'][charInd]
    return feed_dict


def get_init_charEmbVec(index_data,char_list,vec_dim,initializer=np.random.normal):
    char2vec=dict()
    tot_char_num=len(char_list)
    char_vec_init = list(map(list, initializer(loc=0., scale=0.1, size=[tot_char_num, vec_dim])))
    char2vec['char_vec_dim']=vec_dim
    char2vec['char2vec']=char_vec_init

    charInd2charVec={index_data['char2charInd'][char]:char_vec_init[i] for i,char in enumerate(char_list)}
    char2vec['charInd2charVec']=charInd2charVec
    return char2vec

def set_hyperparameter(config):
    hidden_size = 50
    batch_size = 30
    char_vec_dim = 10
    is_training = True
    config['cnn_height'] = 5
    config['cnn_stride'] = [1, 1, 1, 1]
    config['cnn_padding'] = 'VALID'
    config['cnn_output_channel_num'] = 50
    config['initializer'] = dict()
    config['initializer']['cnn'] = tf.truncated_normal_initializer(stddev=0.5)
    config['initializer']['dense'] = [tf.truncated_normal_initializer(stddev=0.5),
                                      tf.truncated_normal_initializer(stddev=0.5)]
    config['optimizer']=tf.train.AdamOptimizer
    config['learning_rate'] = 0.001

    config['hidden_size'] = hidden_size
    config['batch_size'] = batch_size
    config['char_vec_dim'] = char_vec_dim
    config['is_training '] = is_training

    return hidden_size,batch_size,char_vec_dim,is_training,config


if __name__=='__main__':
    ##################################################################################
    # Setting: your path and file directory configuration
    mode = 'train'
    data_path = 'D:/Wisdom/workspace/_dataset/BiDAF/squad/'
    word2vec_path = 'D:/Wisdom/workspace/_dataset/BiDAF/glove/glove.840B.300d.txt'
    version = 'v1.1'
    save_path="./saved_model/01/model.ckpt"
    ##################################################################################
    # Dataset loading part
    json_file = data_path + mode + '-' + version + '.json'
    print('Start dataset loading')
    print('File:' +json_file)
    data = load_json_file(json_file)
    config, full_dataset, sub_info_dataset, index_dataset, word2vec_dataset = \
        get_preprocessed_dataset(data, mode=mode, word2vec_file=word2vec_path)
    print('Finish dataset loading')
    ##################################################################################
    # Setting: Hyperparameters and Tensorflow session
    # 1. Hyperparameter: Please go to set_hyperparameter method. and check it
    d, batch_size, char_vec_dim, is_training, config= set_hyperparameter(config)

    # 2. Tensorflow session
    config_tf=tf.ConfigProto()
    config_tf.log_device_placement=True
    config_tf.gpu_options.allow_growth=True
    # Most of user setting are ended. Try to run.
    ##################################################################################
    n_sample = len(full_dataset['q_dataset'])
    max_word_num_x = config['max_word_num_x']
    max_word_num_q = config['max_word_num_q']
    max_word_len = config['max_word_len']
    word_vec_dim = config['word_vec_dim']

    # Initialization of character embedding vector
    tot_char_list = sub_info_dataset['tot_char_list']
    char2vec_dataset = get_init_charEmbVec(index_dataset, tot_char_list, char_vec_dim, initializer=np.random.normal)

    QAmodel = my_QAmodel(config=config)
    optimizer = config['optimizer'](learning_rate=config['learning_rate']).minimize(QAmodel.loss)

    sess = tf.Session(config=config_tf)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)

    batch_time = n_sample // batch_size
    print('Batch size: ' + str(batch_size))
    print('Batch time: ' + str(batch_time))
    print('-------- Start --------')
    sample_index = np.arange(n_sample)
    np.random.shuffle(sample_index)
    _t_start = time.time()
    _t_start_ = time.time()


    for i in range(batch_time):
        t_start = time.time()
        index_range = sample_index[i * batch_size:(i + 1) * batch_size]
        batch_dict = get_batch_dict(index_range, config, full_dataset, word2vec_dataset, char2vec_dataset)
        feed = QAmodel.get_feed_dict(batch_dict)
        l, _ = sess.run([QAmodel.loss, optimizer], feed_dict=feed)
        if i % 10 == 0:
            saver.save(sess, save_path, global_step=i)
            print('%d/%d complete.\tLoss: %.4f\tTime: %.4f[s]' % (i + 1, batch_time, l, time.time() - _t_start))
            _t_start = time.time()

    now = datetime.datetime.now()
    DayTime = now.strftime('%Y-%m-%d-%H-%M')
    saver.save(sess, save_path.replace('.ckpt', '-{' + DayTime + '}.ckpt'))
    sess.close()
    print('-------- Finish --------')
    print('Running Time: %4.f' % (_t_start - _t_start_))

