"""
This class implements a job for creating city state
"""

import logging
from data.data_exporter import DataExporter
from city_states.create_city_state import CityStateCreator


class CreateCityStateJob(object):
    """
    This class implements a job for creating city state
    """

    def __init__(self, DATA_DIR, DATA_DIR_origin):
        """
        Constructor
        :param DATA_DIR:
        :return:
        """
        self.DATA_DIR = DATA_DIR
        self.DATA_DIR_origin = DATA_DIR_origin
        self.logger = logging.getLogger("cuda_logger")

    def run(self):
        """
        This method executes the job
        :param:
        :return:
        """
        self.logger.info("Starting job: CreateCityStateJob\n")

        city_state_creator = CityStateCreator(self.DATA_DIR, self.DATA_DIR_origin)
        city_state = city_state_creator.get_city_states()
        self.logger.info("Exporting city states\n")
        data_exporter = DataExporter(self.DATA_DIR)
        data_exporter.export_city_state(city_state)

        self.logger.info("Finished job: CreateCityStateJob")

if __name__ == "__main__":
    DATA_DIR = "./data/"
    DATA_DIR_origin = "./data/raw.csv"
    job = CreateCityStateJob(DATA_DIR, DATA_DIR_origin)
    job.run()