"""
This class implements data provider for reading various data
"""
import dill
import os
import pandas as pd
import ast
import json


class DataProvider(object):
    """
    This class implements data provider for reading various data
    """

    def __init__(self, DATA_DIR):
        """
        Constructor
        :param DATA_DIR:
        :return:
        """
        self.DATA_DIR = DATA_DIR

    def read_city_states(self, filename='city_states.dill'):
        """
        Reads the city states dill file
        :param filename:
        :return city_states:
        """
        DATA_DIR = self.DATA_DIR
        city_states_file = os.path.join(DATA_DIR, 'city_states', filename)
        with open(city_states_file, 'rb') as f:
            city_states = dill.load(f)
        return city_states

    def read_hex_bin_attributes(self):
        """
        Reads the csv file containing the hex bin attributes
        :param:
        :return df:
        """
        DATA_DIR = self.DATA_DIR
        attr_file = os.path.join(DATA_DIR, "hex_bins", "hex_bin_attributes.csv")
        df = pd.read_csv(
                attr_file,
                header=0,
                index_col=False,
                converters={'east': ast.literal_eval,
                            'north_east': ast.literal_eval,
                            'north_west': ast.literal_eval,
                            'south_east': ast.literal_eval,
                            'south_west': ast.literal_eval,
                            'west': ast.literal_eval})
        return df

    def read_hex_bin_distances(self):
        """
        Reads the csv file containing the hex bin distances
        :return df:
        """
        DATA_DIR = self.DATA_DIR
        dist_file = os.path.join(
                DATA_DIR,
                "hex_bins",
                "hex_distances.csv")
        df = pd.read_csv(
                dist_file,
                header=0,
                index_col=False)
        return df

    def read_neighborhood_data(self, filename='neighborhood.dill'):
        """
        Reads the neighborhood data dill file
        :param filename:
        :return neighborhood:
        """
        DATA_DIR = self.DATA_DIR
        neighborhood_file = os.path.join(DATA_DIR, filename)
        with open(neighborhood_file, 'rb') as f:
            neighborhood = dill.load(f, encoding='bytes')
        return neighborhood
