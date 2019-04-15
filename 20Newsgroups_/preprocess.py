import os, pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tokenizer import Tokenizer
import config

my_path = '20_newsgroups'

def get_file_label():
    folders = [f for f in os.listdir(my_path)]

    files = []

    for folder_name in folders:
        folder_path = os.path.join(my_path, folder_name)
        files.append([f for f in os.listdir(folder_path)])

    #list of pathnames
    pathname_list = []
    for fo in range(len(folders)):
        for fi in files[fo]:
            pathname_list.append(os.path.join(my_path, os.path.join(folders[fo], fi)))

    #classes of each documents
    Y = []
    for folder_name in folders:
        folder_path = os.path.join(my_path, folder_name)
        num_of_files= len(os.listdir(folder_path))
        for i in range(num_of_files):
            Y.append(folder_name)

    return pathname_list, Y

def split(pathname_list, Y):
    doc_train, doc_test, Y_train, Y_test = train_test_split(pathname_list, Y, random_state=0, test_size=0.25)
    return doc_train, doc_test, Y_train, Y_test


def preprocess():
    path_name_list, labels = get_file_label()
    doc_train, doc_test, Y_train, Y_test = split(path_name_list, labels)

    clean_doc_train, clean_doc_test = [], []
    tokenizer = Tokenizer()
    is_print = True
    for doc_path in tqdm(doc_train):
        if is_print:
            print(doc_path)
            is_print = False
        text_news = []
        with open(doc_path, "r", encoding="gb18030", errors="ignore") as fr:
            for idx, line in enumerate(fr):
                line = line.strip()
                if len(line.split()) < 4:
                    continue
                if idx < 25 and (":" in line or ".edu" in line or "@" in line):
                    continue
                text_news.extend(tokenizer(line))
        clean_doc_train.append(text_news)

    for doc_path in tqdm(doc_test):
        text_news = []
        with open(doc_path, "r", encoding="gb18030", errors="ignore") as fr:
            for idx, line in enumerate(fr):
                line = line.strip()
                if len(line.split()) < 4:
                    continue
                if idx < 25 and (":" in line or ".edu" in line or "@" in line):
                    continue
                text_news.extend(tokenizer(line))
        clean_doc_test.append(text_news)

    assert  len(clean_doc_train) == len(Y_train) and len(clean_doc_test) == len(Y_test)

    with open(config.clean_doc_train_path, "wb") as pkl:
        pickle.dump((clean_doc_train, Y_train), pkl)
    with open(config.clean_doc_test_path, "wb") as pkl:
        pickle.dump((clean_doc_test, Y_test), pkl)

if __name__ == "__main__":
    preprocess()
