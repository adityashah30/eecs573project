import sys
from datahandler import DataHandler
from model_lda import LDAModel

def load_data(fname):
    data_handler = DataHandler(fname)
    try:
        data = data_handler.load_data()
        return data
    except (TypeError, IOError) as detail:
        print "Error:", detail
        sys.exit(1)

def build_lda_model(data, field, num_topics=2):
    data = data[field]
    lda_model = LDAModel(data, num_topics)
    return lda_model

def main():
    if len(sys.argv) != 2:
        print "Usage: python main.py <data.csv>"
        sys.exit(1)
    data = load_data(sys.argv[1])
    lda_model = build_lda_model(data, "Workaround", 3)
    print lda_model.get_model_topics()


if __name__ == "__main__":
    main()
