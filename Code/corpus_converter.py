from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

class CorpusConverter:
    '''
    Converts a list of sentences into a gensim corpus.
    '''

    stopwords = map(unicode, stopwords.words('english')  
                + list(string.punctuation))

    def tokenize(self, sentences):
        return map(lambda sent: filter(lambda tok:
                                       tok not in CorpusConverter.stopwords,
                                      word_tokenize(sent.lower())), sentences)

    def convert(self, sentences):
        tokens = self.tokenize(sentences)
        dictionary = corpora.Dictionary(tokens)
        corpus = map(lambda token: dictionary.doc2bow(token), tokens)
        return corpus, dictionary
