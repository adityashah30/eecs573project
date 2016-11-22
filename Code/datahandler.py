'''
The DataHandler class.

Loads a CSV file given the path
'''
import pandas as pd

class DataHandler:

    def __init__(self, fname=None):
        self.fname = fname

    def load_data(self, fname=None):
        if not self.fname:
            self.fname = fname
        if not self.fname:
            raise TypeError("Enter a non-empty file name")
        try:
            return pd.read_csv(self.fname)
        except IOError as detail:
            raise IOError("Pandas read_csv failed")
