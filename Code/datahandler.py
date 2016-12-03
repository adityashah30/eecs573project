'''
The DataHandler class.

Loads a CSV file given the path
'''
import pandas as pd

class DataHandler:

    def __init__(self, fname=None, is_csv=False):
        self.fname = fname
        self.is_csv = is_csv

    def load_data(self, fname=None):
        if not self.fname:
            self.fname = fname
        if not self.fname:
            raise TypeError("Enter a non-empty file name")
        try:
            if self.is_csv:
                return pd.read_csv(self.fname)
            else:
                return pd.read_excel(self.fname)
        except IOError as detail:
            raise detail
