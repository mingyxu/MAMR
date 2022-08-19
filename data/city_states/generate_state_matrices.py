"""
This class generates and returns data structures that describe a city state
"""

from __future__ import division
import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
from datetime import datetime

def timestamp_datetime(value):
    d = datetime.fromtimestamp(value)
    t = dt.datetime(d.year,d.month,d.day,d.hour,d.minute,0)
    return t


class StateData(object):
    """
    Class for generating data structures that describe a city state
    """
    def __init__(self,
                 start_time,         # Start time of time slice
                 end_time,           # End time of time slice
                 time_slice_duration, # Time slice length in real minutes
                 time_unit_duration,  # 1 time unit = 1 real minutes
                 hex_bins,
                 DATA_DIR_origin):    # Raw data

        
        Order = pd.read_csv(DATA_DIR_origin)
        Order['PU_timestamp'] = Order['PU_time'].apply(timestamp_datetime)
        df = Order.loc[(Order['PU_timestamp']>=start_time)&(Order['PU_timestamp']<=end_time)]

        # Assign instance variables
        self.df                  = df
        self.start_time          = start_time
        self.end_time            = end_time
        self.time_slice_duration = time_slice_duration
        self.time_unit_duration  = time_unit_duration
        self.hex_bins            = hex_bins

    def create_transition_matrix(self):
        """
        Creates ride_count_matrix and transition_matrix
        """
        # Create a networkx graph
        count_df = (self.df[['pickup_bin', 'dropoff_bin', 'weight']]
                        .groupby(by=['pickup_bin', 'dropoff_bin'])
                        .sum()
                        .reset_index())

        G = nx.from_pandas_edgelist(
                        df=count_df,
                        source='pickup_bin',
                        target='dropoff_bin',
                        edge_attr=['weight'],
                        create_using=nx.DiGraph(attr='weight'))

        # Create ride_count_matrix
        ride_count_matrix = nx.to_numpy_matrix(
                        G,
                        nodelist=self.hex_bins,
                        weight='weight')

        self.ride_count_matrix = np.squeeze(np.asarray(ride_count_matrix))

        # Create transition matrix
        G = nx.stochastic_graph(G, weight='weight')
        transition_matrix = nx.to_numpy_matrix(
                        G,
                        nodelist=self.hex_bins,
                        weight='weight')

        transition_matrix = np.squeeze(np.asarray(transition_matrix))

        # Remove 0 values
        transition_matrix[transition_matrix == 0] = 0.001
        transition_matrix = (transition_matrix/transition_matrix
                            .sum(axis=1)[:, None])
        self.transition_matrix = transition_matrix

    def create_pickup_vector(self):
        """
        Creates pickup vector
        """
        self.pickup_vector = self.ride_count_matrix.sum(axis=1, dtype=int)

    def create_dropoff_vector(self):
        """
        Creates dropoff vector
        """
        self.dropoff_vector = self.ride_count_matrix.sum(axis=0, dtype=int)

    def create_distance_matrix(self):
        """
        Creates distance matrix
        """
        # Create networkx graph
        dist_df = (self.df[['pickup_bin', 'dropoff_bin', 'trip_distance']]
                       .groupby(by=['pickup_bin', 'dropoff_bin'])
                       .mean()
                       .reset_index())

        G = nx.from_pandas_edgelist(
                        df=dist_df,
                        source='pickup_bin',
                        target='dropoff_bin',
                        edge_attr=['trip_distance'],
                        create_using=nx.DiGraph(attr='trip_distance'))

        # Create distance matrix
        dist_matrix = nx.to_numpy_matrix(
                        G,
                        nodelist=self.hex_bins,
                        weight='trip_distance')

        distance_matrix = np.squeeze(np.asarray(dist_matrix))
        self.distance_matrix = distance_matrix

    def create_travel_time_matrix(self):
        """
        Creates travel time matrix
        """
        # Create networkx graph
        travel_df = (self.df[['pickup_bin', 'dropoff_bin', 'duration_seconds']]
                         .groupby(by=['pickup_bin', 'dropoff_bin'])
                         .mean()
                         .reset_index())

        # Convert to time units
        travel_df['duration_seconds'] = travel_df['duration_seconds']/60
        travel_df['duration_seconds'] = np.ceil(
                        travel_df['duration_seconds']/self.time_unit_duration)

        G = nx.from_pandas_edgelist(
                        df=travel_df,
                        source='pickup_bin',
                        target='dropoff_bin',
                        edge_attr=['duration_seconds'],
                        create_using=nx.DiGraph(attr='duration_seconds'))

        # Create travel time matrix
        travel_matrix = nx.to_numpy_matrix(
                        G,
                        nodelist=self.hex_bins,
                        weight='duration_seconds')

        travel_matrix = np.squeeze(np.asarray(travel_matrix))
        self.travel_time_matrix = travel_matrix

    def create_reward_matrix(self):
        """
        Creates reward matrix
        """
        # Create networkx graph
        reward_df = (self.df[['pickup_bin', 'dropoff_bin', 'fare_amount']]
                         .groupby(by=['pickup_bin', 'dropoff_bin'])
                         .mean()
                         .reset_index())

        G = nx.from_pandas_edgelist(
                        df=reward_df,
                        source='pickup_bin',
                        target='dropoff_bin',
                        edge_attr=['fare_amount'],
                        create_using=nx.DiGraph(attr='fare_amount'))

        # Create reward matrix
        reward_matrix = nx.to_numpy_matrix(
                        G,
                        nodelist=self.hex_bins,
                        weight='fare_amount')

        reward_matrix = np.squeeze(np.asarray(reward_matrix))
        self.reward_matrix = reward_matrix

    def create_geodesic_matrix(self, hex_dist):
        """
        Creates geodesic distance matrix
        :param hex_dist:
        """
        # Crate networkx graph
        G = nx.from_pandas_edgelist(
                        df=hex_dist,
                        source='pickup_bin',
                        target='dropoff_bin',
                        edge_attr=['straight_line_distance'],
                        create_using=nx.DiGraph(attr='straight_line_distance'))

        # Create geodesic distance matrix
        geodesic_matrix = nx.to_numpy_matrix(
                        G,
                        nodelist=self.hex_bins,
                        weight='straight_line_distance')

        geodesic_matrix = np.squeeze(np.asarray(geodesic_matrix))
        self.geodesic_matrix = geodesic_matrix

