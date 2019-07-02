from module_DTI import my_dataset
from keras.utils import to_categorical
import numpy as np
def load_label_train_test(train_x,train_y,N_label='tot',stacked_model=None):
    if N_label=='tot':
        if stacked_model==None:
            label_train_x=train_x
        else:
            label_train_x = stacked_model.predict(train_x)
        label_train_y=train_y
    else:
        j,k=0,0
        _ind=[]
        for ind,ele in enumerate(train_y):
            if list(ele)==[1,0] and j!=int(N_label)//2:
                j+=1
                _ind.append(ind)
            elif list(ele)==[0,1] and k!=int(N_label)//2:
                k+=1
                _ind.append(ind)
        if stacked_model==None:
            label_train_x=train_x[_ind,:]
        else:
            label_train_x = stacked_model.predict(train_x[_ind,:])
        label_train_y=train_y[_ind,:]
    return label_train_x,label_train_y

def load_DTI(config,**kw):
    DTI = my_dataset.DTI_Dataset(dict_directories=config.Dataset['dict_directories'])
    print('--------------------------------------------------------')
    print('Load DTI data.')
    DTI.load_data(**kw)
    print('--------------------------------------------------------')
    return DTI

def load_unlabel(config,filename):
    print('Load existing unlabel sample file.')
    unlabeled_data = np.load(config.Dataset['dict_directories']['dir_ROOT']+'/unlabel/' + filename)
    print('Complete loading all default dataset & variables.')
    print('\n--------------------------------------------------------')
    return unlabeled_data
def load_pos_neg_samples(config,DTI,**kw):
    DTI.load_pos_neg_samples(config.Dataset['neg_to_pos_ratio'],pos_filename=config.Dataset['pos_filename'],
                           neg_filename=config.Dataset['neg_filename'],**kw)
    

def load_train_test_5fold_CV(config,DTI,rand_ind,order_CV):
    step=1-config.Dataset['split_ratio']
    n=len(DTI.pos.label)
    if order_CV>5 or order_CV<1:
        raise ValueError("Fixed cross-validation as five fold.")
    elif order_CV==1:
        test_ind=rand_ind[(order_CV-1)*int(n*step):order_CV*int(n*step)]
        train_ind=rand_ind[order_CV*int(n*step):]
    elif order_CV==5:
        test_ind=rand_ind[(order_CV-1)*int(n*step):]
        train_ind=rand_ind[:(order_CV-1)*int(n*step)]
    else:
        test_ind=rand_ind[(order_CV-1)*int(n*step):order_CV*int(n*step)]
        train_ind1=rand_ind[:(order_CV-1)*int(n*step)]
        train_ind2=rand_ind[order_CV*int(n*step):]
        train_ind=np.concatenate((train_ind1,train_ind2))

    tmp_tr=np.arange(len(train_ind)*2)
    tmp_te=np.arange(len(test_ind)*2)
    np.random.shuffle(tmp_tr)
    np.random.shuffle(tmp_te)

    train_x=np.concatenate((DTI.pos.data.values[train_ind],DTI.neg.data.values[train_ind]))[tmp_tr]
    test_x=np.concatenate((DTI.pos.data.values[test_ind],DTI.neg.data.values[test_ind]))[tmp_te]
    
    tmp_y0=np.zeros(len(train_x)//2)
    tmp_y1=np.ones(len(train_x)//2)
    train_y=to_categorical(np.concatenate((tmp_y1,tmp_y0))[tmp_tr])
    tmp_y0=np.zeros(len(test_x)//2)
    tmp_y1=np.ones(len(test_x)//2)
    test_y=to_categorical(np.concatenate((tmp_y1,tmp_y0))[tmp_te])
    return train_x,train_y,test_x,test_y

def load_train_test(config,DTI,**kw):
    DTI.split_train_test_set(config.Dataset['split_ratio'],**kw)
    n_train = DTI.train.data.shape[0]
    n_test = DTI.test.data.shape[0]
    train_x, train_y = DTI.train.next_batch(batch_size=n_train, pos_neg_label=DTI.pos_neg_label, one_hot_encoding=True)
    test_x, test_y = DTI.test.next_batch(batch_size=n_test, pos_neg_label=DTI.pos_neg_label, one_hot_encoding=True)
    return train_x,train_y,test_x,test_y

# def default_config():
#     '''Data Loading'''
#     config = collections.namedtuple('config', ['Dataset', 'Model', 'Train', 'Save'])
#
#     dict_directories = {'dir_ROOT': 'D:/Wisdom/workspace_python/research/dataset/final', }
#     dict_directories.update({
#         'DTI_adjmat': dict_directories['dir_ROOT'] + '/drug-target_mat.tsv',
#         'drug': dict_directories['dir_ROOT'] + '/drug_descriptor.tsv',
#         'target': dict_directories['dir_ROOT'] + '/protein_descriptor.tsv'
#     })
#     config_Dataset = {
#         'dict_directories': dict_directories,
#         'neg_to_pos_ratio': 1,
#         'split_ratio': 0.9
#     }
#     # '''Model Training'''
#     # config_Train = {
#     #     'n_sample': 9592 * 2,  # 4170,#2103
#     #     'batch_size': 100,
#     #     'learning_rate': 0.001,
#     #     'epoch': 30,
#     #     'optimizer': tf.train.AdamOptimizer(0.001)
#     # }
#     #
#     # '''Model Building'''
#     # config_Model = {
#     #     'feature_size': 190 + 1437,  # 190 # 1437
#     #     'n_class': 2
#     # }
#     #
#     # '''Model Saver'''
#     # config_Save = {
#     #     'save_model_path': 'D:/Wisdom/workspace_python/research/model/model_saved/'
#     # }
#     config.Dataset = config_Dataset
#     # config.Train = config_Train
#     # config.Model = config_Model
#     # config.Save = config_Save
#     return config