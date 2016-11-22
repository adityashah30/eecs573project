'''
The LdaModel class.

Creates a LDA model from given data and number of topics.
'''
from gensim.models import LdaModel
from gensim.utils import tokenize
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.manifold import TSNE
import matplotlib.pylab as plt

class _CorpusConverter:
    '''
    Converts a list of sentences into a gensim corpus.
    '''

    def __init__(self):
        self.stopwords = map(unicode, stopwords.words('english') 
                                      + list(string.punctuation))

    def _tokenize(self, sentences):
        return map(lambda sent: filter(lambda tok: tok not in self.stopwords,
                                      word_tokenize(sent.lower())) , sentences)

    def convert(self, sentences):
        tokens = self._tokenize(sentences)
        dictionary = corpora.Dictionary(tokens)
        corpus = map(lambda token: dictionary.doc2bow(token), tokens)
        return corpus, dictionary

class LDAModel:

    def __init__(self, sentences=None, num_topics=2):
        self.corpus, self.id2word = _CorpusConverter().convert(sentences)
        self.num_topics = num_topics
        self.lda_model = LdaModel(self.corpus, self.num_topics, self.id2word)

    def get_model_topics(self):
        return self.lda_model.print_topics(-1)
