"""
This class implements a job to export the neighborhood data
"""

import logging
import numpy as np
from data.data_provider import DataProvider
from data.data_exporter import DataExporter
from hex_bins.hex_bin_utils import *

class NeighborhoodDataExportJob(object):
    """
    This class implements a job to export the neighborhood data
    """

    def __init__(self, DATA_DIR):
        """
        :param DATA_DIR:
        :return:
        """
        self.DATA_DIR = DATA_DIR
        self.logger = logging.getLogger("cuda_logger")
        self.radius = 1

    def run(self):
        """
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: NeighborhoodDataExportJob\n")
        data_provider = DataProvider(self.DATA_DIR)
        data_exporter = DataExporter(self.DATA_DIR)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_bins = hex_attr_df['hex_id'].values

        data = {}
        for r in range(self.radius + 1):
            data[r] = {}
            for hex_bin in hex_bins:
                neighbors = hex_neighborhood(hex_bin, hex_attr_df, r)
                zero_vector = np.zeros(len(hex_bins))
                np.put(zero_vector, neighbors, 1)
                one_hot_encoding_vector = zero_vector
                data[r][hex_bin] = one_hot_encoding_vector

        data_exporter.export_neighborhood_data(data)
        self.logger.info("Finished job: NeighborhoodDataExportJob")

if __name__ == "__main__":
    DATA_DIR = "./data/"
    job = NeighborhoodDataExportJob(DATA_DIR)
    job.run()