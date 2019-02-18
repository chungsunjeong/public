from module_DTI import load_data
from module_DTI import sample_handling
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def split_train_test_set(pos_samples, neg_samples, split_ratio=0.9):
    n_pos = pos_samples.shape[0]
    n_neg = neg_samples.shape[0]
    if pos_samples.shape[1] != neg_samples.shape[1]:
        raise ValueError('# of features of a positive sample is not equal to # of features of a positive sample')
    pos_rand = pos_samples.sample(frac=1)
    pos_training = pos_rand.loc[pos_rand.index[:int(split_ratio * n_pos)]]
    pos_test = pos_rand.loc[pos_rand.index[int(split_ratio * n_pos):]]
    neg_rand = neg_samples.sample(frac=1)
    neg_training = neg_rand.loc[neg_rand.index[:int(split_ratio * n_neg)]]
    neg_test = neg_rand.loc[neg_rand.index[int(split_ratio * n_neg):]]

    training_samples = pos_training.T.join(neg_training.T)
    training_samples = training_samples.T
    training_samples = training_samples.sample(frac=1)
    test_samples = pos_test.T.join(neg_test.T)
    test_samples = test_samples.T
    test_samples = test_samples.sample(frac=1)
    return training_samples, test_samples


def next_batch(num, data):
    idx = np.arange(0, data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]

    idx_pair = data.index[idx]
    label = list()
    for pair in idx_pair:
        if pair in pos_label:
            label.append([0, 1])
        elif pair in neg_label:
            label.append([1, 0])
        else:
            raise ValueError('Index pair (' + pair + ') belongs to neither pos/neg label.')
    return data.loc[idx_pair].values, np.array(label)


if __name__=='__main__':
    dir_dataset='D:\\Wisdom\\research\\data\\2012, Tabei'

    dir_DTI_adjmat=dir_dataset+'\\inter_admat.txt'
    dir_drug=dir_dataset+'\\drug_repmat.txt'
    dir_target = dir_dataset + '\\target_repmat.txt'

    matrix_DTI = load_data.load_drug_target_interaction_adjacency_matrix(dir_DTI_adjmat)
    matrix_drug=load_data.load_drug_descriptor_matrix(dir_drug)
    matrix_target=load_data.load_target_descriptor_matrix(dir_target)
    dict_DTI = sample_handling.get_DTI_dict_from_adjmat(matrix_DTI)

    dir_pos='./sample/pos_sample.txt'
    dir_neg = './sample/neg_sample.txt'


    pos_samples = load_data.load_pos_samples(dir_pos)
    neg_samples = load_data.load_neg_samples(dir_neg)

    pos_label = pos_samples.index
    neg_label = neg_samples.index

    split_ratio=0.6
    training_samples, test_samples = split_train_test_set(pos_samples, neg_samples, split_ratio=split_ratio)
    lr = 0.01
    feature_size = 1757
    n_class = 2
    batch_size = 100
    training_epochs = 10

    x = tf.placeholder(tf.float32, [None, feature_size])
    W = tf.Variable(tf.zeros([feature_size, n_class]), name='W')
    b = tf.Variable(tf.zeros([n_class]), name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, n_class])

    cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    # accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))

    sess = tf.Session()

    batch_xs, batch_ys = next_batch(batch_size, training_samples)

    init = tf.global_variables_initializer()
    sess.run(init)
    avg_cost_list = []
    total_batch = int(training_samples.shape[0] / batch_size)
    for epoch in range(training_epochs):
        # accuracy_list=[]
        avg_cost = 0.
        for step in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, training_samples)
            sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}) / total_batch
        avg_cost_list.append(avg_cost)
    plt.plot(range(training_epochs), avg_cost_list)
    # show training accuracy as iteraction in one epoch
    #         accuracy_list.append(sess.run(accuracy_train,feed_dict={x:batch_xs,y_:batch_ys}))
    # itr=range(total_batch)
    # acc=accuracy_list
    # plt.plot(itr,acc)
    # plt.xlabel('iteraction')
    # plt.ylabel('training accuracy')
    plt.show()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype='float32'))
    test_x, test_y = next_batch(test_samples.shape[0], test_samples)
    print(test_samples.shape[0])
    ac = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print(str(ac * 100) + '%')

    sess.close()


