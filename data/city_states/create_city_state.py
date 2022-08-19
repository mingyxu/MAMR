"""
This class creates city state
"""

from __future__ import division
import logging
import pandas as pd
from datetime import timedelta
from data.data_provider import DataProvider
from multiprocess import Manager
from pathos.pools import ProcessPool
from city_states.generate_state_matrices import StateData


class CityStateCreator(object):
    """
    Creates city state structure
    """

    def __init__(self, DATA_DIR, DATA_DIR_origin):
        """
        Constructor
        :returns:
        """
        self.DATA_DIR = DATA_DIR
        self.DATA_DIR_origin = DATA_DIR_origin
        self.logger = logging.getLogger("cuda_logger")
        self.start_time = "2016-01-01 00:00:00"
        self.end_time = "2016-01-31 23:59:59"
        self.time_slice_duration = 10
        self.time_unit_duration = 10
        data_provider = DataProvider(self.DATA_DIR)
        hex_attr_df = data_provider.read_hex_bin_attributes()
        hex_dist_df = data_provider.read_hex_bin_distances()
        self.hex_bins = hex_attr_df['hex_id'].values
        self.hex_dist = hex_dist_df[['pickup_bin', 'dropoff_bin', 'straight_line_distance']]

    def create_parallel_args(self, index, time_slice_starts, time_slice_ends):
        """
        Creates argument dictionary for parallelization
        :param index:
        :param time_slice_starts:
        :param time_slice_ends:
        """
        args = {}
        N = len(index.values)
        for t in range(N):
            args[t] = {
                    'time': index[t],
                    'time_slice_start': time_slice_starts[t],
                    'time_slice_end': time_slice_ends[t]}

        return args

    def get_city_state(self, arg):
        """
        Fill up city states dictionary for time t
        :param arg:
        :return:
        """
        city_states = arg[0]
        t = arg[1]

        state = {}
        state['time'] = self.parallel_args[t]['time']
        state['time_slice_start'] = self.parallel_args[t]['time_slice_start']
        state['time_slice_end'] = self.parallel_args[t]['time_slice_end']
        state['time_slice_duration'] = self.time_slice_duration
        state['time_unit_duration'] = self.time_unit_duration

        # Logging
        self.logger.info(
                "Creating state for time: {}\n".format(state['time']))

        # Create state data
        state_data = StateData(
                    start_time=state['time_slice_start'],
                    end_time=state['time_slice_end'],
                    time_slice_duration=self.time_slice_duration,
                    time_unit_duration=self.time_unit_duration,
                    hex_bins=self.hex_bins,
                    DATA_DIR_origin=self.DATA_DIR_origin)

        # Populate state data matrices and vectors

        # Transition and ride count matrices
        state_data.create_transition_matrix()
        state['transition_matrix'] = state_data.transition_matrix
        state['ride_count_matrix'] = state_data.ride_count_matrix

        # Pickup and dropoff vectors
        state_data.create_pickup_vector()
        state_data.create_dropoff_vector()
        state['pickup_vector'] = state_data.pickup_vector
        state['dropoff_vector'] = state_data.dropoff_vector

        # Distance matrix
        state_data.create_distance_matrix()
        state['distance_matrix'] = state_data.distance_matrix

        # Travel time matrix
        state_data.create_travel_time_matrix()
        state['travel_time_matrix'] = state_data.travel_time_matrix

        # Reward matrix
        state_data.create_reward_matrix()
        state['reward_matrix'] = state_data.reward_matrix

        # Straight line distance matrix
        state_data.create_geodesic_matrix(self.hex_dist)
        state['geodesic_matrix'] = state_data.geodesic_matrix

        # Assign city state
        city_states[t] = state

    def get_city_states(self):
        """
        Creates city states from start time to end time
        :param:
        :return:
        """
        city_states = []
        start_time = self.start_time
        end_time = self.end_time

        # Create array of time slice values between the start and end time
        business_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] 
        business_hours_start = 0
        business_hours_end = 23
        index = pd.date_range(start=start_time, end=end_time, freq=str(self.time_unit_duration)+'min')

        # Filter only the required days and hours
        index = index[index.day_name().isin(business_days)]
        index = index[(index.hour >= business_hours_start) & (index.hour <= business_hours_end)]
        time_slice_starts = index - timedelta(minutes=self.time_slice_duration/2)
        time_slice_ends = index + timedelta(minutes=self.time_slice_duration/2)

        # Create arguments dictionary for parallelization
        self.parallel_args = self.create_parallel_args(index, time_slice_starts, time_slice_ends)

        # Create city states
        manager = Manager()
        city_states = manager.dict()
        N = len(index.values)

        # Create parallel pool
        self.logger.info("Creating parallelization pool")
        pool = ProcessPool(nodes=25)
        pool.map(self.get_city_state, ([city_states, t] for t in range(N)))
        pool.close()
        pool.join()
        pool.clear()
        self.logger.info("Finished creating city states")

        return dict(city_states)
