import os
import collections
import numpy as np
import pandas as pd
from module_DTI import load_data
from module_DTI import handle_sample


class Dataset(object):
    def __init__(self,data):
        self.data=data
        self.label=data.index
        self._index_in_epoch=0
        self._epochs_completed=0
        self._data = None

    def next_batch(self, batch_size, pos_neg_label,one_hot_encoding=True):
        _num_examples = self.data.shape[0]
        start = self._index_in_epoch
        # self._data, _batch_data =pd.DataFrame , pd.DataFrame
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, _num_examples)
            np.random.shuffle(idx)
            self._data = self.data.loc[self.data.index[idx]]
        if start + batch_size > _num_examples:
            self._epochs_completed += 1
            rest_num_examples = _num_examples - start
            data_rest_part = self._data.loc[self._data.index[start:_num_examples]]

            idx0 = np.arange(0, _num_examples)
            np.random.shuffle(idx0)
            self._data = self.data.loc[self.data.index[idx0]]
            start = 0

            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            _batch_data = pd.concat([data_rest_part, data_new_part])
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            _batch_data = self._data[start:end]

        idx_pair = _batch_data.index
        label = list()
        if one_hot_encoding==True:
            for pair in idx_pair:
                if pair in pos_neg_label.pos:
                    label.append([0, 1])
                elif pair in pos_neg_label.neg:
                    label.append([1, 0])
                else:
                    raise ValueError('Index pair (' + pair + ') belongs to neither pos/neg label.')
        else:
            for pair in idx_pair:
                if pair in pos_neg_label.pos:
                    label.append(1)
                elif pair in pos_neg_label.neg:
                    label.append(0)

        return _batch_data.values, np.array(label)


class DTI_Dataset(object):
    def __init__(self,dict_directories):
        self.dict_directories=dict_directories
        self.dir_ROOT = self.dict_directories['dir_ROOT']
        self.pos, self.neg, self.train, self.validation, self.test= [Dataset for _ in range(5)]
        self.matrix_DTI, self.matrix_drug, self.matrix_target = [None for _ in range(3)]
        self.dict_DTI = dict()
        self.pos_neg_label =  collections.namedtuple('pos_neg_label',['pos','neg'])
        self._n_drug, self._n_target, self._n_feature_drug, self._n_feature_target = [None for _ in range(4)]
        self._neg_to_pos_ratio, self._split_ratio= [None for _ in range(2)]
        self._n_pos, self._n_neg = [0 for _ in range(2)]
        self.feature_size = 0
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def load_data(self):
        self.matrix_DTI = load_data.load_drug_target_interaction_adjacency_matrix(self.dict_directories['DTI_adjmat'])
        self.matrix_drug = load_data.load_drug_descriptor_matrix(self.dict_directories['drug'])
        self.matrix_target = load_data.load_target_descriptor_matrix(self.dict_directories['target'])
        self.dict_DTI=handle_sample.get_DTI_dict_from_adjmat(adjmat=self.matrix_DTI)
        self._n_drug=self.matrix_drug.shape[0]
        self._n_target=self.matrix_target.shape[0]
        self._n_feature_drug=self.matrix_drug.shape[1]
        self._n_feature_target = self.matrix_target.shape[1]

    def load_pos_neg_samples(self,neg_to_pos_ratio=1):
        self._neg_to_pos_ratio=neg_to_pos_ratio
        if 'sample' in os.listdir(self.dir_ROOT) \
            and 'pos_sample.txt' in os.listdir(self.dir_ROOT + '\\sample') \
            and 'neg_sample.txt' in os.listdir(self.dir_ROOT + '\\sample'):
                print('Load existing positive & negative sample files.')
                pos = load_data.load_pos_samples(self.dir_ROOT + '\\sample\\pos_sample.txt')
                neg = load_data.load_neg_samples(self.dir_ROOT + '\\sample\\neg_sample.txt')
        else:
            pos = handle_sample.construct_pos_samples(self.dict_DTI, self.matrix_drug, self.matrix_target,
                                                file_name=self.dir_ROOT+'\\sample\\pos_sample.txt')
            neg = handle_sample.construct_neg_samples(self.dict_DTI, self.matrix_drug, self.matrix_target, self._neg_to_pos_ratio,
                                                file_name=self.dir_ROOT+'\\sample\\neg_sample.txt')
        self.pos = Dataset(data=pos)
        self.neg = Dataset(data=neg)
        self._n_pos = self.pos.data.shape[0]
        self._n_neg = self.neg.data.shape[0]
        if self._n_neg != int(self._n_pos * self._neg_to_pos_ratio):
            raise ValueError(
                'Please check neg_to_pos_ratio value or existing *sample.txt files. \n'
                '(# of negative samples) is not equal to (# of positive samples) * (neg_to_pos_ratio) \n'
                'Remove existing (pos_sample.txt / neg_sample.txt) files, or change neg_to_pos_ratio\n'
                'Currently, \n# of pos: %d\n# of neg: %d \nneg_to_pos_ratio: %f' % (
                self._n_pos, self._n_neg, self._neg_to_pos_ratio))
        if self.pos.data.shape[1] != self.neg.data.shape[1]:
            raise ValueError('# of features of a positive sample is not equal to # of features of a positive sample')
        self.feature_size = self.pos.data.shape[1]
        self.pos_neg_label.pos=self.pos.data.index
        self.pos_neg_label.neg=self.neg.data.index


    def split_train_test_set(self, split_ratio=0.9):
        self._split_ratio=split_ratio
        pos_rand = self.pos.data.sample(frac=1)
        pos_training = pos_rand.loc[pos_rand.index[:int(split_ratio * self._n_pos)]]
        pos_test = pos_rand.loc[pos_rand.index[int(split_ratio * self._n_pos):]]
        neg_rand = self.neg.data.sample(frac=1)
        neg_training = neg_rand.loc[neg_rand.index[:int(split_ratio * self._n_neg)]]
        neg_test = neg_rand.loc[neg_rand.index[int(split_ratio * self._n_neg):]]

        training_samples = pos_training.T.join(neg_training.T)
        training_samples = training_samples.T
        training_samples = training_samples.sample(frac=1)
        self.train = Dataset(data=training_samples)
        test_samples = pos_test.T.join(neg_test.T)
        test_samples = test_samples.T
        test_samples = test_samples.sample(frac=1)
        self.test = Dataset(data=test_samples)

