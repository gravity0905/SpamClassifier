import sys
import re
import pickle
import glob
from stemmer import PorterStemmer

def process(text):
    '''Returns a list of words after carying out the
    following text preprocessing and normalization steps'''
    # Convert text to lower case
    text = text.lower()
    #Remove 'Subject'
    text = re.sub(r'^sub(ject)?', ' ', text)
    # Strip HTML
    text = re.sub(r'<.*?>', ' ', text)
    # Normalize URLs
    text = re.sub(r'(http|https|ftp)://\S*', ' httpaddr ', text)
    # Normalize email addresses
    text = re.sub(r'[\w.+-]+@[\w.-]+', ' emailaddr ', text)
    # Normalize numbers
    text = re.sub(r'\b\d[\d,]*[.]*[\d]*\b', ' number ', text)
    # Normalize Dollars/Rupees
    text = re.sub(r'(\$|\brs\b|₹|£)+', ' dollar ', text)
    # Remove non-word characters
    text = re.sub(r'[^a-z]+', ' ', text)
    # Strip all whitespace characters and generate list of words
    # Stop Word Removal
    # stop_words = pickle.load(open('stopwords_set.pyset', 'rb'))
    text = [word for word in text.split() if word not in process.stop_words and len(word)>2]
    # Word Stemming
    p = PorterStemmer()
    result = []
    for word in text:
        try:
            stem_word = p.stem(word, 0, len(word)-1)
            if stem_word not in process.stop_words:
                result.append(stem_word)
        except:
            pass
    return result
process.stop_words = pickle.load(open('stopwords_set.pyset', 'rb'))


def preprocess():
    path = r'.\lingspam_public\processed\**\*.txt'
    for path_name in glob.iglob(path, recursive=True):
        with open(path_name, "r") as f:
            text = process(f.read())
        with open(path_name, "w") as f:
            f.write(" ".join(text))
    return 0


def main():
    if len(sys.argv) != 2:
        print('usage: ./process.py file|--p')
        sys.exit(1)
    if(sys.argv[1]=="--p"):
        preprocess()
    else:
        filename = sys.argv[1]
        with open(filename, 'r') as f:
            print(' '.join(process(f.read())))


if __name__ == '__main__':
    main()