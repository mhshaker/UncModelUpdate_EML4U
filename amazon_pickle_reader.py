# Reader for preprocessed Amazon movie reviews.
# Reads pickle files and holds data in memory.
#
# The preprocessed data was sorted by time and rounded by hour [1], resulting in new identifiers "raw_id".
# The BoW-50 data does not contain 84,090 entries, mainly (by rounding) from reviews between 1997 and 1999.
#
# Example:
# data = AmazonPickleReader('/tmp/')
# print(data.get_text(84090))
# print(data.get_bow50(84090))
# print(data.get_text(84090, metadata=True))
# print(data.get_bow50(84090, metadata=True))

import pickle
import os.path

class AmazonPickleReader:
    
    META_HELPFULNESS = 0
    META_SCORE = 1
    META_DATE = 2
    META_ORIGINAL_NO = 3
    META_ID = 4
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.filename_raw   = 'amazon_raw.pickle'
        self.filename_bow50 = 'amazon_bow_50.pickle'
        self.originalno_to_rawid = None
        self.data_raw   = None
        self.data_bow50 = None

    def set_filename_raw(self, filename):
        self.filename_raw = filename
        
    def set_filename_bow_50(self, filename):
        self.filename_bow50 = filename
        
    def get_all_raw(self):
        if(self.data_raw is None):
            print('AmazonPickleReader: Reading raw data')
            with open(os.path.join(self.data_directory, self.filename_raw), 'rb') as handle:
                self.data_raw = pickle.load(handle)
        return self.data_raw
    
    def get_all_bow50(self):
        if(self.data_bow50 is None):
            print('AmazonPickleReader: Reading bow50 data')
            with open(os.path.join(self.data_directory, self.filename_bow50), 'rb') as handle:
                self.data_bow50 = pickle.load(handle)
        return self.data_bow50

    def get_text(self, raw_id, metadata=False):
        if metadata:
            return self.get_all_raw()[1][raw_id]
        else:
            return self.get_all_raw()[0][raw_id]
    
    def get_bow50(self, raw_id, metadata=False):
        # 1997 to 1999 not included
        bow50_id = raw_id - 84090
        if(bow50_id < 0):
            raise IndexError('list index out of range: ' + str(raw_id))
        if metadata:
            return self.get_all_bow50()["data"][1][bow50_id]
        else:
            return self.get_all_bow50()["data"][0][bow50_id]

    def get_raw_id(self, original_no):
        if(self.originalno_to_rawid is None):
            self.originalno_to_rawid = {}
            for tup in self.get_all_raw()[1]:
                self.originalno_to_rawid[tup[self.META_ORIGINAL_NO]] = tup[self.META_ID]
        return self.originalno_to_rawid[original_no]