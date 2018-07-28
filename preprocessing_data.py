import pickle as pkl
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import Counter


def load_data(filename):
    assert '.pkl' in filename
    with open(filename, 'rb') as fr:
        train_data, test_data = pkl.load(fr)
    return train_data, test_data


def get_series(tag_list, path_name):
    train_data, test_data = load_data(path_name)
    return [train_data[tag] for tag in tag_list], \
           [test_data[tag] for tag in tag_list if tag != 'class']


def padding(li, max_len=2000):
    length = len(li)
    if length < max_len:
        li += [0] * (max_len - length)
    else:
        li = li[:max_len]
    assert len(li) == max_len
    return li


if __name__ == '__main__':

    path = 'data_raw'  # 数据集所在路径
    name = 'data_drop.pkl'  # 文件名
    path_name = os.path.join(path, name)

    (train_text, train_label), (test_text,) = get_series(['word_seg', 'class'], path_name)

    # computing the word number of new dictionary
    word_seg_counter = Counter()
    train_text.apply(lambda x: word_seg_counter.update(x))
    test_text.apply(lambda x: word_seg_counter.update(x))

    new_dictionary = dict([(index, i) for i, (index, num) in enumerate(word_seg_counter.most_common())])
    # print(new_dictionary)
    print('the number of dictionary:', len(new_dictionary))

    # embedding index
    train_text.apply(lambda x: [new_dictionary[i] for i in x])
    test_text.apply(lambda x: [new_dictionary[i] for i in x])

    # 每篇文章填充到2000长度，因为小于1963长度的文章已占98%
    train_text = train_text.apply(padding)
    test_text = test_text.apply(padding)

    train_text = np.array(train_text.tolist())
    test_text = np.array(test_text.tolist())
    print(train_text.shape, test_text.shape)

    # 处理label
    num_label = train_label.max()
    print(num_label)
    train_label = train_label.apply(lambda x: int(x)-1)

    onehot_encoder = OneHotEncoder(sparse=False)
    label = onehot_encoder.fit_transform(train_label.values.reshape(-1, 1))

    with open('data_preprocessed/data_train_keras.pkl', 'wb') as fw_train, \
            open('data_preprocessed/data_test_keras.pkl', 'wb') as fw_test:
        pkl.dump([train_text, label], fw_train)
        pkl.dump(test_text, fw_test)

