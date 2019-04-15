import os, pickle
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
from tqdm import tqdm

import config

class Feature():
    def train_tfidf_model(self, tag="word"):
        print('tfidf model on ' + tag)

        if tag == "word":
            path = "word_tfidf_model.m"
        elif tag == "char":
            path = "char_tfidf_model.m"
        if os.path.exists(path):
            return joblib.load(path)

        with open(config.clean_doc_train_path, "rb") as pkl:
            train_doc, _ = pickle.load(pkl)
        with open(config.clean_doc_test_path, "rb") as pkl:
            test_doc, _ = pickle.load(pkl)
        corpus = []
        corpus.extend(train_doc)
        corpus.extend(test_doc)
        corpus = [" ".join(sent) for sent in corpus]

        if tag == 'word':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='word',
                ngram_range=(1, 1),
                max_features=10000
            )
        elif tag == 'char':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='char',
                ngram_range=(1, 4),
                max_features=30000
            )

        vectorizer.fit(corpus)
        joblib.dump(vectorizer, path)
        return vectorizer

    def get_tfidf(self, tag):
        """
        save (train_tfidf, test_tfidf)
        """
        if tag == "word":
            path = "tfidf_word.pkl"
        elif tag == "char":
            path = "tfidf_char.pkl"

        if os.path.exists(path):
            with open(path, "rb") as pkl:
                return pickle.load(pkl)

        tfidf_model = self.train_tfidf_model(tag)

        with open(config.clean_doc_train_path, "rb") as pkl:
            train_doc, _ = pickle.load(pkl)
        with open(config.clean_doc_test_path, "rb") as pkl:
            test_doc, _ = pickle.load(pkl)

        train_doc = [" ".join(sent) for sent in train_doc]
        test_doc = [" ".join(sent) for sent in test_doc]

        train_tfidf = tfidf_model.transform(train_doc)
        test_tfidf = tfidf_model.transform(test_doc)

        with open(path, "wb") as pkl:
            pickle.dump((train_tfidf, test_tfidf), pkl)
        return train_tfidf, test_tfidf

class WordFeature():
    def load_data(self, tag):
        if tag == "train":
            path = config.clean_doc_train_path
        elif tag == "test":
            path = config.clean_doc_test_path
        with open(path, "rb") as pkl:
            return pickle.load(pkl)


    def generate_powerful_word(self):
        """
        统计各个类别下的重要单词
            [0: 该词语出现的文本数量， 1: 该word出现在各个类别下的数量, 2: 该word出现在各个类别下的比例]
        :return:
        """
        output_path = "powerful_words.pkl"
        if os.path.exists(output_path):
            with open(output_path, "rb") as pkl:
                return pickle.load(pkl)

        train_doc, train_label = self.load_data("train")
        test_doc, test_label = self.load_data("test")
        corpus, labels = [], []
        corpus.extend(train_doc)
        corpus.extend(test_doc)
        labels.extend(train_label)
        labels.extend(test_label)
        assert len(corpus) == len(labels)

        words_power = dict()
        cat_counts = dict()

        for i in range(len(corpus)):
            word_set = set(corpus[i])
            label = labels[i]
            cat_counts[label] = cat_counts.get(label, 0) + 1

            for word in word_set:
                if word not in words_power:
                    words_power[word] = [0 for _ in range(3)]
                    words_power[word][0] = 0
                    words_power[word][1] = dict()
                    words_power[word][2] = dict()
                # word出现的文档次数
                words_power[word][0] += 1
                # word出现的类别次数
                words_power[word][1][label] = words_power[word][1].get(label, 0) + 1

        # word在各个类别下的出现比例
        for word in words_power.keys():
            for label in cat_counts.keys():
                words_power[word][2][label] = words_power[word][1].get(label, 0) / (words_power[word][0] + 1e-6)

        # words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
        with open(output_path, 'wb') as pkl:
            pickle.dump(words_power, pkl)
        return words_power

    def cat_powerful_word(self):
        def init_cat_powerful_word(pword, labels, thres_num, thres_rate):
            pword_cat = dict()
            for label in labels:
                tmp_pword = filter(lambda x: x[1][1].get(label, 0) >= thres_num, pword)
                pword_cat[label] = set(list(map(lambda x:x[0], filter(lambda x: x[1][2].get(label, 0) > thres_rate, tmp_pword))))
            return pword_cat

        _, test_label = self.load_data("test")
        label_set = set(test_label)
        assert len(label_set) == 20

        words_power = self.generate_powerful_word()
        # 感觉用不上, 50和0.6才能卡出少量结果，感觉用了也没啥用， 可能数据没洗好
        return init_cat_powerful_word(words_power.items(), label_set, 50, 0.6)

    def get_powerful_word_feature(self):
        def extract_row(line, p_words, labels_set):
            is_hit = [0. ] * len(labels_set)
            for i in range(len(labels_set)):
                for word in set(line):
                    if word in p_words[labels_set[i]]:
                        is_hit[i] = 1
                        break
            return is_hit

        path = "./powerful_word_hit_feature.pkl"
        if os.path.exists(path):
            with open(path, "rb") as pkl:
                return pickle.load(pkl)

        p_words = self.cat_powerful_word()

        train_doc, _ = self.load_data("train")
        test_doc, test_label = self.load_data("test")
        labels_set = list(set(test_label))  # for index
        assert len(labels_set) == 20

        train_feature, test_feature = [], []
        for line in tqdm(train_doc):
            train_feature.append(extract_row(line, p_words, labels_set))
        for line in tqdm(test_doc):
            test_feature.append(extract_row(line, p_words, labels_set))

        train_feature = np.array(train_feature)
        test_feature = np.array(test_feature)

        with open(path, "wb") as pkl:
            pickle.dump((train_feature, test_feature), pkl)

        return train_feature, test_feature


    def get_prob_feature(self):
        def extract_row(line, pword_dict, labels_set):
            # 换成sum_logprob不知道会不会好点
            num_least = 50
            rate = [1.0] * len(labels_set)
            for i in range(len(labels_set)):
                for word in set(line):
                    if word not in pword_dict:
                        continue
                    if pword_dict[word][0]  < num_least:
                        continue
                    rate[i] *= (1. - pword_dict[word][2].get(labels_set[i], 0))
            rate = [1 - r for r in rate]
            return rate

        path = "./pword_prob.pkl"
        if os.path.exists(path):
            with open(path, "rb") as pkl:
                return pickle.load(pkl)

        pword_dic = self.generate_powerful_word()

        train_doc, _ = self.load_data("train")
        test_doc, test_label = self.load_data("test")
        labels_set = list(set(test_label)) # for index
        assert  len(labels_set) == 20

        train_feature, test_feature = [], []
        for line in tqdm(train_doc):
            train_feature.append(extract_row(line, pword_dic, labels_set))
        for line in tqdm(test_doc):
            test_feature.append(extract_row(line, pword_dic, labels_set))

        train_feature = np.array(train_feature)
        test_feature = np.array(test_feature)

        with open(path, "wb") as pkl:
            pickle.dump((train_feature, test_feature), pkl)

        return train_feature, test_feature

class SentenceFeature():
    def __init__(self):
        pass

class EmbeddingFeature():
    def __init__(self):
        pass

if __name__ == "__main__":
    feature = Feature()
    feature.train_tfidf_model("word")
    feature.train_tfidf_model("char")
    feature.get_tfidf("word")
    feature.get_tfidf("char")
    word_feature = WordFeature()
    word_feature.generate_powerful_word()
    # word_feature.get_prob_feature()
    word_feature.get_powerful_word_feature()

