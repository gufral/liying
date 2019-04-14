import string
import numpy as np
from os import listdir
from os.path import join
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

my_path = '20_newsgroups'

folders = [f for f in listdir(my_path)]

files = []

for folder_name in folders:
    folder_path = join(my_path, folder_name)
    files.append([f for f in listdir(folder_path)])

#list of pathnames
pathname_list = []
for fo in range(len(folders)):
    for fi in files[fo]:
        pathname_list.append(join(my_path, join(folders[fo], fi)))

#classes of each documents
Y = []
for folder_name in folders:
    folder_path = join(my_path, folder_name)
    num_of_files= len(listdir(folder_path))
    for i in range(num_of_files):
        Y.append(folder_name)

#train test split
doc_train, doc_test, Y_train, Y_test = train_test_split(pathname_list, Y, random_state=0, test_size=0.25)

f = open("stop_words.txt", "r")
stopwords = f.read()
f.close()

# preprocess the words list
def preprocess(words):
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]
    punctuations = (string.punctuation).replace("'", "")
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]
    words = [str for str in stripped_words if str]
    p_words = []
    for word in words:
        if (word[0] and word[len(word) - 1] == "'"):
            word = word[1:len(word) - 1]
        elif (word[0] == "'"):
            word = word[1:len(word)]
        else:
            word = word
        p_words.append(word)
    words = p_words.copy()
    words = [word for word in words if not word.isdigit()]
    words = [word for word in words if not len(word) == 1]
    words = [str for str in words if str]
    words = [word.lower() for word in words]
    words = [word for word in words if len(word) > 2]

    return words

#remove stopwords
def remove_stopwords(words):
    words = [word for word in words if not word in stopwords]
    return words

# convert sentence
def tokenize_sentence(line):
    words = line[0:len(line) - 1].strip().split(" ")
    words = preprocess(words)
    words = remove_stopwords(words)
    return words

#remove metadata
def remove_metadata(lines):
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines


# convert document
def tokenize(path):
    f = open(path, 'r', encoding='gb18030', errors='ignore')
    text_lines = f.readlines()
    text_lines = remove_metadata(text_lines)
    doc_words = []
    for line in text_lines:
        doc_words.append(tokenize_sentence(line))
    return doc_words

def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list

list_of_words = []
for document in doc_train:
        list_of_words.append(flatten(tokenize(document)))

np_list_of_words = np.asarray(flatten(list_of_words))
words, counts = np.unique(np_list_of_words, return_counts=True)
freq, wrds = (list(i) for i in zip(*(sorted(zip(counts, words), reverse=True))))

f_o_w = []
n_o_w = []
for f in sorted(np.unique(freq), reverse=True):
    f_o_w.append(f)
    n_o_w.append(freq.count(f))

#deciding the no. of words to use as feature
n = 5000
features = wrds[0:n]

#creat dictionary
dictionary = {}
doc_num = 1
for doc_words in list_of_words:
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    dictionary[doc_num] = {}
    for i in range(len(w)):
        dictionary[doc_num][w[i]] = c[i]
    doc_num = doc_num + 1

X_train = []
for k in dictionary.keys():
    row = []
    for f in features:
        if(f in dictionary[k].keys()):
            row.append(dictionary[k][f])
        else:
            row.append(0)
    X_train.append(row)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

list_of_words_test = []
for document in doc_test:
        list_of_words_test.append(flatten(tokenize(document)))

dictionary_test = {}
doc_num = 1
for doc_words in list_of_words_test:
    np_doc_words = np.asarray(doc_words)
    w, c = np.unique(np_doc_words, return_counts=True)
    dictionary_test[doc_num] = {}
    for i in range(len(w)):
        dictionary_test[doc_num][w[i]] = c[i]
    doc_num = doc_num + 1

X_test = []
for k in dictionary_test.keys():
    row = []
    for f in features:
        if(f in dictionary_test[k].keys()):
            row.append(dictionary_test[k][f])
        else:
            row.append(0)
    X_test.append(row)

X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

#lsa
lsa = TruncatedSVD(n_components=500)
X_train = lsa.fit_transform(X_train)
X_test = lsa.transform(X_test)

# GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, Y_train)
Y_predict = gbc_clf.predict(X_test)
gbc_clf.score(X_test, Y_test)
print(classification_report(Y_test, Y_predict))
