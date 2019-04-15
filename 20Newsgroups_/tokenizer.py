import string, re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

class Tokenizer():
    def __init__(self):
        self.punc = string.punctuation
        self.stop_words = []
        self.wnl = WordNetLemmatizer()

        with open("stop_words.txt", "r", encoding="utf-8") as fr:
            for line in fr:
                self.stop_words.append(line.strip())

    def __call__(self, text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        text= text.strip().lower()
        text = ' ' + text + ' '

        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)cm ", lambda m: m.group(1) + ' cm ', text)
        text = re.sub(r"(\d+)h ", lambda m: m.group(1) + ' hour ', text)
        text = re.sub(r"(\d+)hr ", lambda m: m.group(1) + ' hour ', text)
        text = re.sub(r"(\d+)hrs ", lambda m: m.group(1) + ' hour ', text)
        text = re.sub(r"(\d+)day ", lambda m: m.group(1) + ' day ', text)
        text = re.sub(r"(\d+)days ", lambda m: m.group(1) + ' day ', text)
        text = re.sub(r"(\d+)week ", lambda m: m.group(1) + ' week ', text)
        text = re.sub(r"(\d+)weeks ", lambda m: m.group(1) + ' week ', text)
        text = re.sub(r"(\d+)% ", lambda m: m.group(1) + ' percent ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"didn’t", "do not", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)
        text = re.sub(r"\<", " < ", text)
        text = re.sub(r"\>", " > ", text)
        text = re.sub(r"/\t", "", text)
        text = re.sub(r"``", "", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)  # 测试！
        text = re.sub(r"\$", " dollar ", text)



        # ordinal numeral
        text = re.sub(r"1st", " first ", text)
        text = re.sub(r"2nd", " second ", text)
        text = re.sub(r"3rd", " third ", text)
        text = re.sub(r"4th", " fourth ", text)
        text = re.sub(r"5th", " fifth ", text)
        text = re.sub(r"6th", " sixth ", text)
        text = re.sub(r"7th", " seventh ", text)
        text = re.sub(r"8th", " eighth ", text)
        text = re.sub(r"9th", " ninth ", text)
        text = re.sub(r"10th", " tenth ", text)
        text = re.sub(r"11th", " eleventh ", text)
        text = re.sub(r"12th", " twelfth ", text)
        text = re.sub(r"20th", " twentieth ", text)
        text = re.sub(r"29th", " twenty-ninth ", text)

        word_list = nltk.word_tokenize(text.strip())
        tagged_sent = nltk.pos_tag(word_list)

        word_list = [self.wnl.lemmatize(word, get_wordnet_pos(tag) or wordnet.NOUN) for word,tag in tagged_sent]
        return [word for word in word_list if word not in self.punc  and "@" not in word and word not in self.stop_words and not word.isdigit()]