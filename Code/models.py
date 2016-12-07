from gensim.models import LdaModel
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE

from corpus_converter import CorpusConverter

class LDAModel:
    '''
    Creates a LDA model using Gensim's LdaModel class.
    '''

    def __init__(self, sentences=None, num_topics=2):
        self.converter = CorpusConverter()
        self.corpus, self.id2word = self.converter.convert(sentences)
        self.num_topics = num_topics
        self.lda_model = LdaModel(self.corpus, self.num_topics, self.id2word)

    def get_model_topics(self):
        return self.lda_model.print_topics(-1)

class LDAModelSklearn:
    '''
    Creates a LDA model using Scikit-learn's (Sklearn) 
    LatentDirichletAllocation class
    '''

    def __init__(self, sentences=None, num_topics=2):
        self.sentences = sentences
        self.num_topics = num_topics
        self.lda_model = LatentDirichletAllocation(self.num_topics)
        self.converter = CorpusConverter()

    def fit_transform(self):
        vectorizer = CountVectorizer(stop_words=CorpusConverter.stopwords)
        matrix_vectorized = vectorizer.fit_transform(self.sentences)
        return self.lda_model.fit_transform(matrix_vectorized)

    def plot(self, matrix_reduced, plot_=False):
        if self.num_topics == 2:
            plt.scatter(matrix_reduced[:,0], matrix_reduced[:,1])
        elif self.num_topics == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(matrix_reduced[:, 0], matrix_reduced[:, 1], 
                       matrix_reduced[:, 2])
        if plot_:
            plt.show()
        else:
            plt.savefig("Errata_LDA-"+str(self.num_topics)+"D.png")

class LSIModelSklearn:
    '''
    Creates a LSI(LSA) model using Scikit-learn's (Sklearn) TruncatedSVD class.
    '''

    def __init__(self, sentences=None, num_topics=2):
        self.sentences = sentences
        self.num_topics = num_topics
        self.lsi_model = TruncatedSVD(num_topics)

    def fit_transform(self):
        vectorizer = TfidfVectorizer(stop_words=CorpusConverter.stopwords)
        matrix_vectorized = vectorizer.fit_transform(self.sentences)
        return self.lsi_model.fit_transform(matrix_vectorized)

    def plot(self, matrix_reduced, plot_=False):
        if self.num_topics == 2:
            plt.scatter(matrix_reduced[:,0], matrix_reduced[:,1])
        elif self.num_topics == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(matrix_reduced[:, 0], matrix_reduced[:, 1], 
                       matrix_reduced[:, 2])
        if plot_:
            plt.show()
        else:
            plt.savefig("Errata_LSI-"+str(self.num_topics)+"D.png")
