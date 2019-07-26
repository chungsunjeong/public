from collections import Counter
import string
import re
import tensorflow as tf
import sys
from load_dataset import get_preprocessed_dataset, load_json_file
from model import my_QAmodel
from main import set_hyperparameter
def copy_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def copy_f1_score(prediction, ground_truth):
    prediction_tokens = copy_normalize_answer(prediction).split()
    ground_truth_tokens = copy_normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def copy_exact_match_score(prediction, ground_truth):
    return (copy_normalize_answer(prediction) == copy_normalize_answer(ground_truth))

def copy_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def copy_evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += copy_metric_max_over_ground_truths(
                    copy_exact_match_score, prediction, ground_truths)
                f1 += copy_metric_max_over_ground_truths(
                    copy_f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__=='__main__':
    ##################################################################################
    # Setting: your path and file directory configuration
    mode = 'dev'
    data_path = 'D:/Wisdom/git/tmp/BIDAF/data/squad/'
    version = 'small'  # 'v1.1'
    save_path = "./saved_model/"
    meta_model_file = save_path + 'my_test_model-{2019-07-18-14-23}.ckpt.meta'
    whole_model_file = save_path + 'my_test_model-{2019-07-18-14-23}.ckpt'
    ##################################################################################
    # Dataset loading part
    json_file = data_path + mode + '-' + version + '.json'
    print('Start dataset loading')
    print('File:' + json_file)
    data = load_json_file(json_file)
    config, full_dataset, sub_info_dataset, index_dataset, word2vec_dataset = \
        get_preprocessed_dataset(data, mode=mode)
    print('Finish dataset loading')
    ##################################################################################
    # Setting: Tensorflow session
    config_tf = tf.ConfigProto()
    config_tf.log_device_placement = True
    config_tf.gpu_options.allow_growth = True
    ##################################################################################
    d, batch_size, char_vec_dim, is_training, config = set_hyperparameter(config)
    config['is_training'] = False

    QAmodel = my_QAmodel(config=config)
    config_tf = tf.ConfigProto()
    config_tf.log_device_placement = True
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
    loader = tf.train.import_meta_graph(meta_model_file)
    loader.restore(sess, whole_model_file)


    # graph = tf.get_default_graph()
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    # w1 = graph.get_tensor_by_name("w1:0")
    # p1=graph.get_tensor_by_name("p1:0")
    # p2 = graph.get_tensor_by_name("p2:0")