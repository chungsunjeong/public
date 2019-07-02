import pandas as pd
import numpy as np

def get_concat_features_mat(drug_descriptor_mat,target_descriptor_mat,type='linear'):
    print('There are two types in concatenation process: linear(default) / tensor-product')
    if type=='linear':
        n_feature_drug = len(drug_descriptor_mat.keys())
        n_feature_target = len(target_descriptor_mat.keys())
        n_sample_drug = len(drug_descriptor_mat.index)
        n_sample_target = len(target_descriptor_mat.index)
        tmp_drug = pd.DataFrame(pd.np.tile(drug_descriptor_mat, (n_sample_target, 1)), columns=drug_descriptor_mat.keys())
        print(len(tmp_drug.index))
        ind_pair = []
        for ind1 in target_descriptor_mat.index:
            for ind2 in drug_descriptor_mat.index:
                ind_pair.append(str(ind2) + '_' + str(ind1))

        tmp_target = pd.DataFrame(
            np.reshape(pd.np.tile(target_descriptor_mat, (1, n_sample_drug)), (n_sample_drug * n_sample_target, n_feature_target)),
            columns=target_descriptor_mat.keys())
        tmp_concat = pd.concat([tmp_drug, tmp_target], axis=1)
        tmp_concat.index = ind_pair
    if type=='tensor-product':
        tmp_concat=pd.DataFrame(np.kron(drug_descriptor_mat, target_descriptor_mat),
                     columns=pd.MultiIndex.from_product([drug_descriptor_mat, target_descriptor_mat]),
                     index=pd.MultiIndex.from_product([drug_descriptor_mat.index, target_descriptor_mat.index]))
    print('# of drug-target pair: ' + str(tmp_concat.shape[0]))
    print('# of features of a drug-target pair: : ' + str(tmp_concat.shape[1]) + '\n')
    return tmp_concat

# if __name__=='__main__':
#     dir_dataset='D:\\Wisdom\\research\\data\\2012, Tabei'
#
#     dir_DTI_adjmat=dir_dataset+'\\inter_admat.txt'
#     dir_drug=dir_dataset+'\\drug_repmat.txt'
#     dir_target = dir_dataset + '\\target_repmat.txt'
#
#     matrix_DTI = load_data.load_drug_target_interaction_adjacency_matrix(dir_DTI_adjmat)
#     matrix_drug=load_data.load_drug_descriptor_matrix(dir_drug)
#     matrix_target=load_data.load_target_descriptor_matrix(dir_target)
