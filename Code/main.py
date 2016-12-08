import sys
from datahandler import DataHandler
from models import LDAModelGensim, LDAModelSklearn, LSIModelSklearn
from visualizer import Visualizer

'''
The main file to run the tests on processor data for ARM Cortex-A9.
We use the following algorithms to extract data from processor errata

1. Topic Modelling using LDA (Latent Dirichlet Allocation)
2. Topic Modelling using LSA (Latent Semantic Analysis)
3. Using clustering algorithms such as KMeans to group similar errata
4. Use Hierarchical clustering methods to first cluster on a field and then
   clustering subsequently on those clusters.
'''

def load_data(fname):
    data_handler = DataHandler(fname)
    try:
        data = data_handler.load_data()
        return data
    except (TypeError, IOError) as detail:
        print "Error: ", detail
        sys.exit(1)

def build_lda_model(data, field, num_topics=2):
    data = data[field]
    lda_model = LDAModel(data, num_topics)
    return lda_model

def build_lda_model_sklearn(data, field, num_topics=2):
    data = data[field]
    lda_model = LDAModelSklearn(data, num_topics)
    lda_model.plot(lda_model.fit_transform())
    print lda_model.get_model_topics()

def build_lsi_model_sklearn(data, field, num_topics=2):
    data = data[field]
    lsi_model = LSIModelSklearn(data, num_topics)
    lsi_model.plot(lsi_model.fit_transform())

def visualize_data(data, field, num_dimensions=3):
    visualizer = Visualizer(data[field], num_dimensions, True)
    visualizer.visualize()

def main():
    if len(sys.argv) != 2:
        print "Usage: python main.py <data.csv/data.xlsx>"
        sys.exit(1)
    data = load_data(sys.argv[1])
    # lda_model = build_lda_model(data, "Summary", 10)
    # print lda_model.get_model_topics()
    # visualize_data(data, "Summary", 3)
    build_lda_model_sklearn(data, "Summary", 3)
    # build_lsi_model_sklearn(data, "Summary", 3)

if __name__ == "__main__":
    main()
