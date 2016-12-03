class Visualizer:

    def __init__(self, sentences=None, num_dimensions=3, plot=False):
        self.sentences = sentences
        self.tokens = _CorpusConverter().tokenize(self.sentences)
        self.num_dimensions = num_dimensions
        self.plot = plot

    def visualize(self):
        corpus_converter = _CorpusConverter()
        vectorizer = TfidfVectorizer(stop_words=corpus_converter.stopwords)
        matrix_vectorized = vectorizer.fit_transform(self.sentences).toarray()
        tsne = TSNE(self.num_dimensions)
        matrix_reduced = tsne.fit_transform(matrix_vectorized)
        if self.num_dimensions == 2:
            plt.scatter(matrix_reduced[:,0], matrix_reduced[:,1])
        elif self.num_dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(matrix_reduced[:, 0], matrix_reduced[:, 1], 
                       matrix_reduced[:, 2])
        if self.plot:
            plt.show()
        else:
            plt.savefig("Errata-"+str(self.num_dimensions)+"D.png")
