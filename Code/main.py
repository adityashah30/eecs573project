import sys
import pandas as pd
import gensim
import nltk

def load_data(fname):
    data = pd.read_csv(fname)
    return data

def model_summary(data):
    summary = data['Summary']
    sentences = [nltk.word_tokenize(sentence) for sentence in summary]

def main():
    if len(sys.argv) != 2:
        print "Usage: python main.py <data.csv>"
        sys.exit(1)
    data = load_data(sys.argv[1])
    model_summary(data)

if __name__ == "__main__":
    main()
