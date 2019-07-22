import pandas as pd

def load_general_matrix(dir,type,header='infer'):
    mat = pd.read_csv(dir, delim_whitespace=True,header=header)
    print('# of '+type+'s: ' + str(mat.shape[0]))
    print('# of features of a '+type+': ' + str(mat.shape[1]) + '\n')
    mat.index = mat.index.map(str)
    return mat

def load_drug_target_interaction_adjacency_matrix(dir,header='infer'):
    mat=pd.read_csv(dir, delim_whitespace=True,header=header)
    print('# of drug: '+str(mat.shape[0]))
    print('# of target: '+str(mat.shape[1])+'\n')
    mat.index=mat.index.map(str)
    return mat


def load_drug_target_interaction_dict(dir,header='infer'):
    mat = pd.read_csv(dir, delim_whitespace=True, header=header,names=['target','drug'])
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


def load_drug_descriptor_matrix(dir,header='infer'):
    return load_general_matrix(dir,type='drug',header=header)


def load_target_descriptor_matrix(dir,header='infer'):
    return load_general_matrix(dir, type='target',header=header)


def load_drug_target_pair_matrix(dir,header='infer'):
    return load_general_matrix(dir, type='drug-target pair',header=header)


def load_pos_samples(dir,header='infer'):
    return load_general_matrix(dir, type='positive sample',header=header)


def load_neg_samples(dir,header='infer'):
    return load_general_matrix(dir, type='negative sample',header=header)


def load_etc(dir,header='infer'):
    pass