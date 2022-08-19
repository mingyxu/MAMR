"""
This class implements data exporter for storing various data
"""
import logging
import dill
import os
import pyaml


class DataExporter(object):
    """
    This class implements data exporter for storing various data
    """

    def __init__(self, DATA_DIR):
        """
        Constructor
        """
        self.DATA_DIR = DATA_DIR

    def export_city_state(self, city_states, filename='city_states.dill'):
        """
        Exports a dill file containing city states
        :param city_states:
        :param filename:
        :return:
        """
        DATA_DIR = self.DATA_DIR
        filepath = os.path.join(DATA_DIR, 'city_states', filename)
        with open(filepath, 'wb') as f:
            dill.dump(city_states, f)
        self.logger.info("Exported city states to {}".format(filepath))

    def export_bin_distances(self, dist_df, filename='hex_distances.csv'):
        """
        Exports a csv file containing the distances between hex bins
        :param dist_df:
        :param filename:
        :return:
        """
        DATA_DIR = self.DATA_DIR
        filepath = os.path.join(DATA_DIR, 'hex_bins', filename)
        dist_df.to_csv(filepath, sep=',', header=True, index=False)

    def export_neighborhood_data(self, data, filename='neighborhood.dill'):
        """
        Exports a dill file containing neighborhood bins at various radius
        :param data:
        :param filename:
        :return:
        """
        DATA_DIR = self.DATA_DIR
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'wb') as f:
            dill.dump(data, f)
        self.logger.info("Exported neighborhood to {}".format(filepath))
