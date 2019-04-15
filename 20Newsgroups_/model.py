import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from scipy.sparse import hstack
import numpy as np
import xgboost as xgb
from scipy.sparse import coo_matrix

from feature import Feature, WordFeature
import config

def load_data(tag):
    if tag == "train":
        path = config.clean_doc_train_path
    elif tag == "test":
        path = config.clean_doc_test_path
    with open(path, "rb") as pkl:
        return pickle.load(pkl)

def load_feature():
    print("loading feature....")
    feature = Feature()
    word_feature = WordFeature()

    _, train_labels = load_data("train")
    _, test_labels = load_data("test")
    labels_set = set(test_labels)
    assert len(labels_set) == 20
    labels_dic = {}
    for idx, label in enumerate(labels_set):
        labels_dic[label] = idx
    assert len(labels_dic) == 20
    train_labels = np.asarray([labels_dic[label] for label in train_labels])
    test_labels = np.asarray([labels_dic[label] for label in test_labels])

    tfidf_word_train, tfidf_word_test = feature.get_tfidf("word")
    # tfidf_char_train, tfidf_char_test = feature.get_tfidf("char")
    # tfidf_train = hstack([tfidf_word_train, tfidf_char_train])
    # tfidf_test = hstack([tfidf_word_test, tfidf_char_test])
    tfidf_train = tfidf_word_train
    tfidf_test = tfidf_word_test
    pwords_train, pwords_test = word_feature.get_prob_feature()
    pwords_train = coo_matrix(pwords_train)
    pwords_test = coo_matrix(pwords_test)
    # hit_train, hit_test = word_feature.get_powerful_word_feature()
    # hit_train = coo_matrix(hit_train)
    # hit_test = coo_matrix(hit_test)
    #
    # pwords_train = hstack([pwords_train, hit_train])
    # pwords_test = hstack([pwords_test, hit_test])

    # train_feature, test_feature = tfidf_train, tfidf_test
    train_feature = hstack([tfidf_train, pwords_train])
    test_feature = hstack([tfidf_test, pwords_test])
    return train_feature, train_labels, test_feature, test_labels, labels_dic


def train_valid(X_train, Y_train, X_test, Y_test, labels_dic):
    print("train model......")
    # model = GradientBoostingClassifier(subsample=0.9,
    #                                      min_samples_leaf=5,
    #                                      verbose=1)
    model = MultinomialNB()
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    model.score(X_test, Y_test)
    print(classification_report(Y_test, Y_predict, target_names=list(labels_dic.keys())))

class Xgboost():
    def __init__(self):
        # self.n_folds = 10
        self.params = {  'booster':'gbtree',
                         'max_depth':6,
                         'eta':0.1,
                         'objective':'multi:softmax',
                         'subsample':0.7,
                         'colsample_bytree':0.7,
                         'lambda':10,
                         'alpha':1,
                         'nthread':24,
                         'silent':True,
                         'gamma': 0.1,
                         'eval_metric':'mlogloss',
                         'num_class': 20,
                    }
        self.num_rounds = 1000
        self.early_stop_rounds = 10

    def train(self):
        train_feature, train_labels, test_feature, test_labels, labels_dic = load_feature()
        xgb_train = xgb.DMatrix(train_feature, label=train_labels)
        xgb_test = xgb.DMatrix(test_feature, label=test_labels)
        watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
        model = xgb.train(self.params, xgb_train, self.num_rounds, watchlist,
                          early_stopping_rounds=self.early_stop_rounds)
        predict = model.predict(xgb_test)
        print(classification_report(test_labels, predict, target_names=list(labels_dic.keys())))


if  __name__ == "__main__":
    # X_train, Y_train, X_test, Y_test, labels_dic = load_feature()
    # train_valid(X_train, Y_train, X_test, Y_test, labels_dic)
    xgboost = Xgboost()
    xgboost.train()


