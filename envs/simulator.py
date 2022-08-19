"""
These methods help match passengers to drivers during each step
for one taxi
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os


def get_degree(S, D):
    if S==0 or D==0:
        r = 1
    else:
        r = 1 - (abs(S-D) / max(S,D))
    return r

def get_order(city_state_hexbin, current_demand):
    """ 
    get next order in current gird at time step t
    :param city_state_hexbin: number of orders from file "city_state"
    :param current_demand: number of fulfilled orders
    :return next_order_hexbin: next order waiting to be completed in which hexbin
    """
    accu = np.add.accumulate(city_state_hexbin)
    i = 0
    while (current_demand + accu[i]) < accu[-1]:
        i += 1
    return i

def get_order2(hex_demand):
    return (hex_demand!=0).argmax(axis=0)

def initialize_driver_distribution(num_drivers):
    """
    Creates driver distribution at time 0
    :param num_drivers:
    :return driver_distribution:
    """
    count_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/envs/drivers_d30.csv")
    drivers_counts = pd.read_csv(count_dir)
    drivers = np.arange(num_drivers)
    driver_distribution = np.split(drivers, np.cumsum(drivers_counts['counts_9']))[:-1]
    return [x.tolist() for x in driver_distribution]

def get_neighbor(hex_attr_df, hex_bin, direction):
    """
    Get neighbor
    :param hex_attr_df:
    :param hex_bin:
    :param direction:
    :return neighbor:
    """
    attributes = hex_attr_df.iloc[hex_bin]
    try:
        neighbor = int(attributes[direction])
    except ValueError:
        neighbor = None
    return neighbor

def take_dispatch_action(t, T, S, drivers, hex_bin, city_state, driver_distribution_matrix, driver_state, index=None):
    """
    Takes dispatch action for all drivers in the current hex_bin
    :param t:
    :patam T:
    :patam S:
    :param drivers: 
    :param action:
    :param hex_bin:
    :param city_state:
    :param driver_distribution_matrix:
    :return driver_distribution_matrix, reward, hex_state_demand:
    """
    reward = 0
    total_travel = 0
    hex_demand = []
    pax_destination_vector = np.array(city_state['ride_count_matrix'][hex_bin].astype(int))
    # Assign drivers to passengers
    if index is None: # S>D >> D=0
        driver_destination_vector = np.split(drivers, np.cumsum(pax_destination_vector))
        driver_distribution_matrix[t][hex_bin] = driver_destination_vector[-1]
        driver_destination_vector = driver_destination_vector[:-1]
    else:
        we_pax_destination_vector = pax_destination_vector[index]
        destination = np.split(drivers, np.cumsum(we_pax_destination_vector))
        destination1 = destination[:-1]
        recovery_arr = np.zeros_like(destination1)
        for idx, num in enumerate(destination1):
            recovery_arr[index[idx]] = num
        driver_distribution_matrix[t][hex_bin] = destination[-1]
        driver_destination_vector = recovery_arr
    for i in range(S):
        hex_demand.append(pax_destination_vector[i] - len(driver_destination_vector[i]))

    fulfilled_order = len(drivers) - len(driver_distribution_matrix[t][hex_bin])
    for i in range(len(driver_distribution_matrix[t][hex_bin])):
        dirver_id = driver_distribution_matrix[t][hex_bin][i].astype(int)
        driver_state[dirver_id][0] = 0
        driver_state[dirver_id][1] = hex_bin
        driver_state[dirver_id][2] = t
        driver_state[dirver_id][3] = hex_bin

    hex_state_demand = city_state['pickup_vector'][hex_bin].astype(int) - fulfilled_order

    # Update next driver distribution and update total rewards
    for i in range(S):
        if len(driver_destination_vector[i]) > 0:  
            if i != hex_bin:  # If the driver was matched with a passenger
                travel_time = int(city_state['travel_time_matrix'][hex_bin][i])
                total_travel += len(driver_destination_vector[i])*travel_time
                if travel_time == 0:
                    travel_time = 1
                reward += len(driver_destination_vector[i])*city_state['distance_matrix'][hex_bin][i]*1.5
            else:
                travel_time = 1
                reward += 0
            t_prime = t + travel_time
            if t_prime < T:
                try:
                    driver_distribution_matrix[t_prime][i] = (
                        driver_distribution_matrix[t_prime][i].tolist() + driver_destination_vector[i].tolist())
                except:
                    driver_distribution_matrix[t_prime][i] = (
                        driver_distribution_matrix[t_prime][i] + driver_destination_vector[i].tolist())
            for j in range(len(driver_destination_vector[i])):
                dirver_id = driver_destination_vector[i][j].astype(int)
                driver_state[dirver_id][0] = 1
                driver_state[dirver_id][1] = hex_bin
                driver_state[dirver_id][2] = t_prime
                driver_state[dirver_id][3] = i
    return driver_distribution_matrix, reward, hex_state_demand, driver_state, hex_demand, total_travel


def take_relocate_action(t, T, driver_id, action, hex_bin, hex_attr_df, city_state, driver_distribution_matrix, city_state_demand, next_city_demand, driver_state_i, hex_demand):
    """
    Takes relocate action for one driver 
    :param t:
    :param T:
    :param drivers:
    :param driver_id:
    :param action:
    :param hex_bin:
    :param hex_attr_df:
    :param city_state:
    :param driver_distribution_matrix:
    :param city_state_demand 
    :return driver_distribution_matrix, reward, city_state_demand:
    """
    reward = 0
    trv_time = 0
    income = 0
    act_dict = {0: "north_east_neighbor",
                1: "north_neighbor",
                2: "north_west_neighbor",
                3: "south_east_neighbor",
                4: "south_neighbor",
                5: "south_west_neighbor"
                }
    neighbor = get_neighbor(hex_attr_df, hex_bin, act_dict[action])

    # If the relocate action is infeasible because the hex bin is on the map boundary
    if neighbor is None:
        # wait
        travel_time = 1
        neighbor = hex_bin
    else:
        # relocate to neighbor
        reward -= city_state['distance_matrix'][hex_bin][neighbor] * 0.5 # cost from c to n
        if city_state_demand[neighbor] > 0: 
            destination_hexbin = get_order2(hex_demand[neighbor])
            hex_demand[neighbor][destination_hexbin] -= 1
            city_state_demand[neighbor] -= 1
            travel_time = int(city_state['travel_time_matrix'][neighbor][destination_hexbin])
            reward += city_state['distance_matrix'][neighbor][destination_hexbin] * 1.5
            trv_time += travel_time
            driver_state_i[0] = 1
            driver_state_i[3] = destination_hexbin
        else:
            travel_time = int(city_state['travel_time_matrix'][hex_bin][neighbor])
            driver_state_i[0] = 0
            driver_state_i[3] = neighbor
        if travel_time == 0:  # travel_time is at least 1 (this condition is for sanity)
            travel_time = 1
    income = reward
    if t < (T-2):
        reward += get_degree(next_city_demand[hex_bin], len(driver_distribution_matrix[t+1][hex_bin])) # current gird
    t_prime = t + travel_time
    driver_distribution_matrix[t][hex_bin] = np.delete(driver_distribution_matrix[t][hex_bin], np.where(driver_distribution_matrix[t][hex_bin] == driver_id))
    if t_prime < T:
        driver_distribution_matrix[t_prime][driver_state_i[3]] = np.append(driver_distribution_matrix[t_prime][driver_state_i[3]], driver_id)

    # driver_state_i[1] = neighbor
    driver_state_i[1] = hex_bin
    driver_state_i[2] = t_prime
            
    return driver_distribution_matrix, reward, city_state_demand, driver_state_i, hex_demand, trv_time, income


def take_wait_action(t, T, driver_id, hex_bin, driver_distribution_matrix, next_city_demand, driver_state_i):
    """
    Takes wait action for all drivers in the current hex_bin
    :param t:
    :patam T:
    :param drivers:
    :param driver_id: 
    :param hex_bin:
    :param driver_distribution_matrix:
    :return driver_distribution_matrix, reward:
    """
    reward = 0
    travel_time = 1
    t_prime = t + travel_time
    if t_prime < T:
        driver_distribution_matrix[t_prime][hex_bin] = np.append(driver_distribution_matrix[t_prime][hex_bin], driver_id)
    driver_state_i[0] = 0 # idle
    driver_state_i[1] = hex_bin # c grid
    driver_state_i[2] = t_prime # d time
    driver_state_i[3] = hex_bin # d grid
    if t < (T-2):
        reward += get_degree(next_city_demand[hex_bin], len(driver_distribution_matrix[t+1][hex_bin])) # current gird
    
    return driver_distribution_matrix, reward, driver_state_i


def take_action(t, _action, city_state, hex_attr_df, driver_distribution_matrix, T, driver_id, hex_bin, city_state_demand, next_city_demand, driver_state_i, hex_demand):
    """
    Take the chosen action for one driver
    :param t:
    :param action:
    :param city_state:
    :param hex_attr_df:
    :param driver_distribution_matrix:
    :param T:
    :param driver_id:
    :param hex_bin:
    :return driver_distribution_matrix, reward, city_state_demand:
    """
    reward = 0
    trv_time = 0
    income = 0
    action = np.nonzero(_action)[0][0]
    hex_demand_all = hex_demand

    if action <= 5:       # relocate to neighbor bin  
        driver_distribution_matrix, reward, city_state_demand, driver_state_i, hex_demand_all, trv_time, income = take_relocate_action(t, T, driver_id, action, hex_bin, hex_attr_df, city_state,
                                                                    driver_distribution_matrix, city_state_demand, next_city_demand, driver_state_i, hex_demand)

    else:  # action == 6  # wait action
        driver_distribution_matrix, reward, driver_state_i = take_wait_action(t, T, driver_id, hex_bin, driver_distribution_matrix, next_city_demand, driver_state_i)

    return driver_distribution_matrix, reward, city_state_demand, driver_state_i, hex_demand_all, trv_time, income
