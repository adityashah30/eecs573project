import os
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from corpus_converter import CorpusConverter

class Visualizer:

    def __init__(self):
        pass

    def visualize(self, sentences, n_dimensions=2, plot_=False):
        vectorizer = TfidfVectorizer(stop_words=CorpusConverter.stopwords)
        matrix_vectorized = vectorizer.fit_transform(sentences).toarray()
        dim_reducer = TSNE(self.n_dimensions)
        matrix_reduced = dim_reducer.fit_transform(matrix_vectorized)
        self.plot(matrix_reduced, dim_reducer.__class__.__name__, 
                  n_dimensions, plot_)

    def plot(self, matrix, model_name, n_dimensions=2, plot_=False):
        if n_dimensions == 2:
            plt.scatter(matrix[:,0], matrix[:,1])
        elif n_dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(matrix[:, 0], matrix[:, 1], 
                       matrix[:, 2])
        if plot_:
            plt.show()
        else:
            plot_path = os.path.join("..", "Plots", 
                                     "Errata-{}-{}D.png".format(model_name, 
                                                                n_dimensions))
            plt.savefig(plot_path)
