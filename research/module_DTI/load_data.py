import pandas as pd

def load_general_matrix(dir,type):
    mat = pd.read_csv(dir, delim_whitespace=True)
    print('# of '+type+'s: ' + str(mat.shape[0]))
    print('# of features of a '+type+': ' + str(mat.shape[1]) + '\n')
    mat.index = mat.index.map(str)
    return mat

def load_drug_target_interaction_adjacency_matrix(dir):
    mat=pd.read_csv(dir, delim_whitespace=True)
    print('# of drug: '+str(mat.shape[0]))
    print('# of target: '+str(mat.shape[1])+'\n')
    mat.index=mat.index.map(str)
    return mat


def load_drug_target_interaction_dict(dir):
    mat = pd.read_csv(dir, delim_whitespace=True, header=None,names=['target','drug'])
    dict_DTI=dict()
    i=0
    for value in mat.values:
        if value[1] not in dict_DTI.keys():
            dict_DTI[value[1]]=[value[0]]
            i+=1
        else:
            dict_DTI[value[1]].append(value[0])
            i+=1

    print('# of drug-target interactions: ' + str(i) + '\n')
    return dict_DTI


def load_drug_descriptor_matrix(dir):
    return load_general_matrix(dir,type='drug')


def load_target_descriptor_matrix(dir):
    return load_general_matrix(dir, type='target')


def load_drug_target_pair_matrix(dir):
    return load_general_matrix(dir, type='drug-target pair')


def load_pos_samples(dir):
    return load_general_matrix(dir, type='positive sample')


def load_neg_samples(dir):
    return load_general_matrix(dir, type='negative sample')


def load_etc(dir):
    pass
