import pickle as pkl
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_recall_fscore_support,precision_score,recall_score,f1_score
from sklearn.model_selection import StratifiedKFold
import os


path = 'data_raw'
train_name = 'data_train.pkl'
test_name = 'data_test.pkl'
LinearSVC().fit()
vector = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
train_tfidf = vector.fit_transform(train_data['word_seg'])

with open(os.path.join(path, train_name), 'rb') as fr_train, \
     open(os.path.join(path, test_name), 'rb') as fr_test:
    train_data = pkl.load(fr_train)
    test_data = pkl.load(fr_test)
k_fold = StratifiedKFold(n_splits=5)
for train_index, dev_index in k_fold.split(train_data['tf_idf'], train_data['class']):
    train_dataset = tra




