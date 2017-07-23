import re
import numpy as np
import glob
import math
from collections import defaultdict, Counter
from process import process


def tokenizer(text, map=False, preprocess=False):
    """ Returns a set/dict of all words in text """
    if preprocess==True:
        lis_words = Counter(process(text))
    else:
        lis_words = Counter(text.split())
    if map==False:
        return list(lis_words.keys())
    else:
        return lis_words


def generate_dictionary(path):
    '''Generates a dictionary of words using the words in mails inside
    the directory specified by the path.
    Returns dictionary of words along with their counts
    CAUTION : Path must be raw string'''
    dictionary = Counter()
    for path_name in glob.iglob(path, recursive=True):
        with open(path_name, "r") as f:
            for word, count in tokenizer(f.read(), map=True).items():
                dictionary[word] += count
    # words, _ = zip(*dictionary.most_common(N))
    return dictionary


class multivariateNBClassifier:
    """Multivariate Bernoulli Naive Bayes Spam Classifier"""

    def __init__(self, dictionary):
        '''
        dictionary : list of the most frequent words
        N = No. of words in the dictionary
        word_cnt : 2xN matrix, stores word counts in spam/non-spam mails
        cnt = [count_spam_mails, count_non_spam_mails]
        '''
        self.dictionary = np.asarray(dictionary)
        self.N = len(dictionary)     
        self.word_cnt = np.ones((2, self.N))
        self.cnt = np.array([[2], [2]])        

    
    def train(self, path, is_spam, preprocess=False):
        ''' Train the classifier on all the email files with matching path
        is_spam = 1 if all the emails are spam
        is_spam = 0 if all the emails are not spam
        CAUTION : path must be a raw string 
        '''
        for path_name in glob.iglob(path, recursive=True):
            with open(path_name, "r") as f:
                text = np.asarray(tokenizer(f.read(), preprocess=preprocess))
                self.word_cnt[is_spam] += np.in1d(self.dictionary, text)
                self.cnt[is_spam,0] += 1


    def classify(self, message, preprocess=True):
        '''Classifies message as spam(1) or non spam(0)'''
        text = np.asarray(tokenizer(message, preprocess=preprocess))
        # log of probability of message being a spam
        log_prob_spam = 0.0
        #log of probability of message not being a spam
        log_prob_not_spam = 0.0
        # Precomputing logarithms
        word_cnt = np.log(self.word_cnt)
        cnt = np.log(self.cnt)
        word_cnt_comp = np.log(self.cnt-self.word_cnt) 
        index = np.in1d(self.dictionary, text)
        
        log_prob_spam = np.sum(np.where(index, word_cnt[1]-cnt[1,0], word_cnt_comp[1]-cnt[1,0]))
        log_prob_not_spam = np.sum(np.where(index, word_cnt[0]-cnt[0,0], word_cnt_comp[0]-cnt[0,0]))

        log_prob_spam += cnt[1,0] - np.log(np.sum(self.cnt))
        log_prob_not_spam += cnt[0,0] - np.log(np.sum(self.cnt))
        return int(log_prob_spam > log_prob_not_spam)


class multinomialNBClassifier:
    """Multinomial Naive Bayes Spam Classifier"""

    def __init__(self, dictionary):
        '''
        dictionary : list of the most frequent words
        N = No. of words in the dictionary
        word_cnt : 2xN matrix, stores word counts in spam/non-spam mails
        cnt = [count_spam_mails, count_non_spam_mails]
        '''
        self.dictionary = list(dictionary)
        self.N = len(dictionary)     
        self.word_cnt = np.ones((2, self.N))
        self.cnt = np.r_[self.N, self.N]        

    
    def train(self, path, is_spam, preprocess=False):
        ''' Train the classifier on all the email files with matching path
        is_spam = 1 if all the emails are spam
        is_spam = 0 if all the emails are not spam
        CAUTION : path must be a raw string 
        '''
        for path_name in glob.iglob(path, recursive=True):
            with open(path_name, "r") as f:
                # text : Dict containing term frequencies
                text = tokenizer(f.read(), map=True, preprocess=preprocess)
                self.word_cnt[is_spam, :] += np.array([text[word] for word in self.dictionary])
                self.cnt[is_spam] += sum(list(text.values()))


    def classify(self, message, preprocess=True):
        '''Classifies message as spam(1) or non spam(0)'''
        map_words = tokenizer(message, map=True, preprocess=preprocess)
        # log of probability of message being a spam
        log_prob_spam = 0.0
        #log of probability of message not being a spam
        log_prob_not_spam = 0.0
        # Precomputing logarithms
        word_cnt = np.log(self.word_cnt)
        cnt = np.log(self.cnt)
        for i, word in enumerate(self.dictionary):
            # if word appears in the message
            if word in map_words:
                # log_prob_spam += map_words[word]*(math.log(self.word_cnt[1, i]) 
                #                  - math.log(self.cnt[1]))
                # log_prob_not_spam += map_words[word]*(math.log(self.word_cnt[0, i])
                #                      - math.log(self.cnt[0]))
                log_prob_spam += map_words[word]*(word_cnt[1, i] - cnt[1])
                log_prob_not_spam += map_words[word]*(word_cnt[0, i] - cnt[0])

        # log_prob_spam += math.log(self.cnt[1]) - math.log(self.cnt[0] + self.cnt[1])
        # log_prob_not_spam += math.log(self.cnt[0]) - math.log(self.cnt[0] + self.cnt[1])

        log_prob_spam += cnt[1] - np.log(np.sum(cnt))
        log_prob_not_spam += cnt[0] - np.log(np.sum(cnt))

        return int(log_prob_spam > log_prob_not_spam)