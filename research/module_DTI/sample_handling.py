import pandas as pd
import numpy as np


def get_DTI_dict_from_adjmat(adjmat):
    dict_DTI=dict()
    adjmat=adjmat.T
    i=0
    for drug in adjmat.keys():
        target_list=adjmat.index[adjmat[drug]==1].tolist()
        dict_DTI[drug]=target_list
        i+=len(target_list)
    print('# of drug-target interactions: '+str(i)+'\n')
    return dict_DTI


def get_pos_labels(dict_DTI):
    pos_label = set()

    for drug, targets in dict_DTI.items():
        for target in targets:
            pair_name = [str(drug) + '_' + target]
            pos_label.update(pair_name)
    return pos_label


def construct_pos_samples(dict_DTI, matrix_drug, matrix_target, file_name=None):
    pos_label = get_pos_labels(dict_DTI)

    i = 0
    pos_sample=pd.DataFrame()
    for drug, targets in dict_DTI.items():
        for target in targets:
            pair_name=[str(drug) + '_' + target]
            if i==0:
                pos_sample=pd.DataFrame(pd.concat([matrix_drug.loc[drug],matrix_target.loc[target]]),columns=pair_name)
            else:
                df_tmp=pd.DataFrame(pd.concat([matrix_drug.loc[drug],matrix_target.loc[target]]),columns=pair_name)
                pos_sample=pos_sample.join(df_tmp)
            i+=1

    pos_sample = pos_sample.T
    print('# of constructed positive samples: ' + str(pos_sample.shape[0]))
    print('# of features in constructed positive sample: ' + str(pos_sample.shape[1]))
    if pos_sample.shape[0] != len(pos_label):
        raise ValueError(
            'The size of positive sample in the positive sample matrix is not matched to the lenghth of positive label list.')

    if file_name is not None:
        pos_sample.to_csv(file_name,sep='\t')

    return pos_sample, pos_label


def construct_neg_samples(dict_DTI, matrix_drug, matrix_target, neg_to_pos_ratio=1, type='whole', file_name=None):
    pos_label = get_pos_labels(dict_DTI)

    n_positive = len(pos_label)
    n_negative = n_positive * neg_to_pos_ratio
    neg_sample = pd.DataFrame()

    if type == 'whole':
        neg_label_total = set()
        for ind1 in matrix_target.index:
            for ind2 in matrix_drug.index:
                neg_label_total.update([str(ind2) + '_' + str(ind1)])
        n_tmp1=len(neg_label_total)
        neg_label_total = neg_label_total.difference(pos_label)
        neg_label = np.random.choice(list(neg_label_total), size=n_negative, replace=False)
        n_tmp2=len(neg_label_total)
        if n_tmp1 != n_tmp2 + n_positive:
            raise ValueError(
                'wrong difference between the positive samples and total negative samples')

        j = 0
        for neg in neg_label:
            drug, target = neg.split('_', 1)
            if j == 0:
                neg_sample = pd.DataFrame(pd.concat([matrix_drug.loc[drug], matrix_target.loc[target]]),
                                          columns=[neg])
            else:
                df_tmp = pd.DataFrame(pd.concat([matrix_drug.loc[drug], matrix_target.loc[target]]), columns=[neg])
                neg_sample = neg_sample.join(df_tmp)
            j += 1
    neg_sample = neg_sample.T
    print('# of constructed negative samples: ' + str(neg_sample.shape[0]))
    print('# of features in constructed negative sample: ' + str(neg_sample.shape[1]))
    if neg_sample.shape[0] != len(neg_label):
        raise ValueError(
            'The size of negative sample in the negative sample matrix is not matched to the lenghth of negative label list.')

    if file_name is not None:
        neg_sample.to_csv(file_name, sep='\t')

    return neg_sample, neg_label


# if __name__=='__main__':
    # dir_dataset='C:\\Users\\csjeong\\Desktop\\research\\dataset\\conv_DTI\\2012, Tabei'
    #
    # dir_DTI_adjmat=dir_dataset+'\\inter_admat.txt'
    # dir_drug=dir_dataset+'\\drug_repmat.txt'
    # dir_target = dir_dataset + '\\target_repmat.txt'
    #
    # matrix_DTI = load_data.load_drug_target_interaction_adjacency_matrix(dir_DTI_adjmat)
    # matrix_drug=load_data.load_drug_descriptor_matrix(dir_drug)
    # matrix_target=load_data.load_target_descriptor_matrix(dir_target)
    #
    # dict_DTI = get_DTI_dict_from_adjmat(matrix_DTI)
    # # construct_pos_samples(dict_DTI, matrix_drug, matrix_target, file_name='./sample/pos_sample.txt')
    # construct_neg_samples(dict_DTI,matrix_drug,matrix_target,neg_to_pos_ratio=1,
    #                       type='whole',file_name='./sample/neg_sample.txt')
