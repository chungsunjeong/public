import numpy as np
from collections import defaultdict
import json
from nltk.tokenize import word_tokenize

def str_lower_cleaner(s):
    return s.replace("''",'" ').replace('``','" ').lower()

def my_tokenize(s):
    tokens = word_tokenize(s)
    tokens = ['"' if t == "''" or t=='``' else t for t in tokens]
    return tokens

def load_json_file(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_corresponding_glove_word2vec(filename,word_list):
    word_vec_dim = int(filename.split('.')[-2].replace('d', ''))
    word2vec = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word = line[:line.index(' ')]
            if word in word_list:
                l = line.rstrip().split(" ")
                word2vec[word] = list(map(float, l[1:]))
    return word2vec,word_vec_dim

def get_word_char_counter(data):
    context_word_counter = defaultdict(int)
    context_char_counter = defaultdict(int)
    query_word_counter = defaultdict(int)
    query_char_counter = defaultdict(int)
    n_article = len(data['data'])
    for a in range(n_article):
        for c in range(len(data['data'][a]['paragraphs'])):
            context = str_lower_cleaner(data['data'][a]['paragraphs'][c]['context'])
            for char in set(context):
                context_char_counter[char] += context.count(char)
            word_list = my_tokenize(context)

            for word in word_list:
                context_word_counter[word] += 1

            for q in range(len(data['data'][a]['paragraphs'][c]['qas'])):
                query = str_lower_cleaner(data['data'][a]['paragraphs'][c]['qas'][q]['question'])
                for char in set(query):
                    query_char_counter[char] += query.count(char)
                word_list = my_tokenize(query)
                for w in word_list:
                    query_word_counter[w] += 1
    return context_word_counter,context_char_counter,query_word_counter,query_char_counter

def get_ind_dictionaries(word_list,char_list):
    word2ind, ind2word = dict(), dict()
    char2ind, ind2char = dict(), dict()
    for i, key in enumerate(word_list):
        word2ind[key] = i + 1
        ind2word[i + 1] = key

    for i, key in enumerate(char_list):
        char2ind[key] = i + 1
        ind2char[i + 1] = key

    word2ind['-EMPTY-'] = 0
    ind2word[0] = '-EMPTY-'
    char2ind['-EMPTY-'] = 0
    ind2char[0] = '-EMPTY-'
    return word2ind, ind2word, char2ind, ind2char


def get_preprocessed_dataset(data,mode='train'):
    config=defaultdict(dict)
    full_dataset=defaultdict(dict)
    sub_info_dataset=defaultdict(dict)
    index_dataset=defaultdict(dict)
    word2vec_dataset=defaultdict(dict)

    max_word_num_x = 0
    max_word_num_q = 0
    max_word_len = 0

    # x_dataset {key=sample_ind, value=[a context's words' indexes list (shape: [max_word_num_x])]}
    x_dataset = dict()
    # x_char_dataset {key=sample_ind, value=[a context's wodrds' chars' indexes list (shape: [max_word_num_x by max_word_len])]}
    x_char_dataset = dict()
    # q_dataset {key=sample_ind, value=[a query's words' indexes list (shape: [max_word_num_q])]}
    q_dataset = dict()
    # q_char_dataset {key=sample_ind, value=[a query's words' chars' indexes list (shape: [max_word_num_q by max_word_len])]}
    q_char_dataset = dict()
    # y_start_dataset {key=sample_ind, value=[answers' start_word's index (shape: [answer_num by 1])]}
    y_start_dataset = dict()
    # y_end_dataset {key=sample_ind, value=[answers' end_word's index (shape: [answer_num by 1])]}
    y_end_dataset = dict()
    # qind2xind {key=q_sample_ind, value=x_sample_ind }
    qind2xind = dict()

    context_word_counter, context_char_counter, query_word_counter, query_char_counter=get_word_char_counter(data)
    word_len_context, word_len_query = dict(), dict()
    tot_word_list = set(query_word_counter.keys()).union(set(context_word_counter.keys()))
    tot_char_list = set(query_char_counter.keys()).union(set(context_char_counter.keys()))

    word2ind, ind2word, char2ind, ind2char = get_ind_dictionaries(tot_word_list, tot_char_list)

    glove_file = './data/glove/glove.840B.300d.txt'
    word2vec, word_vec_dim=get_corresponding_glove_word2vec(glove_file, tot_word_list)
    old_word2vec_word_list = set(word2vec.keys())
    new_word2vec_word_list = tot_word_list.difference(old_word2vec_word_list)
    # Initialization of new word's embedding vector
    new_word_vec_init = list(map(list, np.random.normal(0, 0.5, size=[len(new_word2vec_word_list), word_vec_dim])))
    for i, word in enumerate(new_word2vec_word_list):
        word2vec[word] = new_word_vec_init[i]

    word2char = dict()
    wordind2vec = dict()

    for word in tot_word_list:
        wordind2vec[word2ind[word]] = word2vec[word]
        word2char[word] = list(word)
        max_word_len = max(max_word_len, len(word))

    x_sample_ind = 0
    q_sample_ind = 0
    ans_sample_ind = 0
    n_article = len(data['data'])

    for a in range(n_article):
        for c in range(len(data['data'][a]['paragraphs'])):
            word_ind_list = []
            char_ind_list = []
            context = str_lower_cleaner(data['data'][a]['paragraphs'][c]['context'])
            word_list = my_tokenize(context)
            max_word_num_x = max(max_word_num_x, len(word_list))
            for word in word_list:
                word_ind_list.append(word2ind[word])
                char_ind_list.append([char2ind[char] for char in word2char[word]])
            x_dataset[x_sample_ind] = word_ind_list
            x_char_dataset[x_sample_ind] = char_ind_list
            word_len_context[x_sample_ind] = len(word_list)
            for q in range(len(data['data'][a]['paragraphs'][c]['qas'])):
                word_ind_list = []
                char_ind_list = []
                query = str_lower_cleaner(data['data'][a]['paragraphs'][c]['qas'][q]['question'])
                word_list = my_tokenize(query)
                max_word_num_q = max(max_word_num_q, len(word_list))
                for word in word_list:
                    word_ind_list.append(word2ind[word])
                    char_ind_list.append([char2ind[char] for char in word2char[word]])
                q_dataset[q_sample_ind] = word_ind_list
                q_char_dataset[q_sample_ind] = char_ind_list
                qind2xind[q_sample_ind] = x_sample_ind
                word_len_query[q_sample_ind] = len(word_list)
                if mode == 'train':
                    answer = data['data'][a]['paragraphs'][c]['qas'][q]['answers']
                    start = answer[0]['answer_start']
                    end = start + len(answer[0]['text'])
                    if start == 0:
                        start += 1
                        end = start + len(answer[0]['text'])
                    y_start_dataset[q_sample_ind] = len(my_tokenize(context[:start]))
                    y_end_dataset[q_sample_ind] = len(my_tokenize(context[:end])) - 1
                else:
                    answers = data['data'][a]['paragraphs'][c]['qas'][q]['answers']
                    start_list = []
                    end_list = []
                    for ans in range(len(answers)):
                        start = answers[ans]['answer_start']
                        end = start + len(answers[ans]['text'])
                        if start == 0:
                            start += 1
                            end = start + len(answers[ans]['text'])
                        start_list.append(len(my_tokenize(context[:start])))
                        end_list.append(len(my_tokenize(context[:end])) - 1)
                    y_start_dataset[q_sample_ind] = start_list
                    y_end_dataset[q_sample_ind] = end_list
                q_sample_ind += 1
            x_sample_ind += 1

    config['word_vec_dim']=word_vec_dim
    config['max_word_len']=max_word_len
    config['max_word_num_x'] = max_word_num_x
    config['max_word_num_q'] = max_word_num_q
    config['max_word_len'] =max_word_len
    sub_info_dataset['tot_word_list'] = tot_word_list
    sub_info_dataset['tot_char_list'] = tot_char_list
    sub_info_dataset['n_article'] = n_article
    sub_info_dataset['n_paragraph'] = x_sample_ind
    sub_info_dataset['n_query'] = q_sample_ind
    sub_info_dataset['n_answer'] = ans_sample_ind
    sub_info_dataset['context_word_counter']=context_word_counter
    sub_info_dataset['context_char_counter'] = context_char_counter
    sub_info_dataset['query_word_counter'] = query_word_counter
    sub_info_dataset['query_char_counter'] = query_char_counter
    index_dataset['word2wordInd'] = word2ind
    index_dataset['wordInd2word'] = ind2word
    index_dataset['char2charInd'] = char2ind
    index_dataset['charInd2char'] = ind2char
    word2vec_dataset['word2vec'] = word2vec
    word2vec_dataset['wordInd2wordVec'] = wordind2vec
    word2vec_dataset['word_vec_dim']=word_vec_dim
    full_dataset['x_dataset']=x_dataset
    full_dataset['x_char_dataset']=x_char_dataset
    full_dataset['q_dataset']=q_dataset
    full_dataset['q_char_dataset']=q_char_dataset
    full_dataset['y_start_dataset']=y_start_dataset
    full_dataset['y_end_dataset']=y_end_dataset
    full_dataset['qInd2xInd'] = qind2xind
    full_dataset['word_len_context'] = word_len_context
    full_dataset['word_len_query'] = word_len_query

    return config,full_dataset,sub_info_dataset,index_dataset,word2vec_dataset

if __name__=='__main__':
    mode = 'train'
    path='D:/Wisdom/git/tmp/BIDAF/data/squad/'
    version='small'

    # json_file=path+mode+'-'+version+'.json'
    # data=load_json_file(json_file)
    # config, full_dataset, sub_info_dataset, counter_dataset, index_dataset, word2vec_dataset= \
    #     get_preprocessed_dataset(data,mode='train')
