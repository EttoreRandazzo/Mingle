"""

    MOSTLY FOR HORSES,
    general data manipulation functions


"""




import pandas as pd
import glob, os
from math import radians, cos, sin, asin, sqrt
import itertools
import random
from haversine import haversine
from datetime import datetime, timedelta, date, time


def havm(p1,p2,miles = False):
    # use this to have it in meters..
    return haversine(p1,p2,miles) * 1000


def estimate_positions_with_interpolation(data,indexes,clock):
    positions = []
    for animal,index in enumerate(indexes):
        if 0 < index < len(data[animal]):
            # this is the common case. it means that we have data.
            # we want to interpolate the actual index with the previous one and get that position
            prev = data[animal][index-1]
            futr = data[animal][index]
            range_t = (futr[1]-prev[1]).total_seconds()
            #print("prev: "+ str(prev[1]))
            #print("futr: "+ str(futr[1]))
            #print("clock: "+ str(clock))
            #print("range"+str(range_t))
            interval_t = (futr[1]-clock).total_seconds()
            #print("interval"+str(interval_t))
            t = interval_t/range_t
            #print(t)
            pos = (prev[0][0]*t + futr[0][0]*(1-t), prev[0][1]*t + futr[0][1]*(1-t))
            positions.append(pos)
        else: positions.append(None)

    return positions

def estimate_positions(data,indexes,clock):
    positions = []
    for animal,index in enumerate(indexes):
        if 0 < index < len(data[animal]):
            # this is the common case. it means that we have data.
            # for starters we just give the previous position (no interpolation)
            positions.append(data[animal][index-1][0])
        else: positions.append(None)

    return positions


def create_network_from_GPS_data(data,max_distance,min_time,interval = 5):
    """

    :param data: a list (classification) of list (times) of positions.
        final line: [(lat,lon),time]. time is a Datetime object
    :param max_distance: threshold. To have an interaction, they must be that close
    :param min_time: threshold. To have an interaction, they must be close for that time
    :param interval: how many seconds we sample
    :return: list (one for each interval) of list of edges of interactions
    """

    # interval is going to be a timedelta
    inter = timedelta(0,interval)

    # we need to have a starting time.
    clock_time = min(list(map(lambda times: times[0][-1],data)))

    # as well as an ending time.
    end_time = max(map(lambda times: times[-1][-1],data))

    # we need to save the edges of every interval!
    result = []

    # we hold different indexes for every horse (every horse has a different timestamp) and update the index
    # when the clock_time is >= the actual time.
    actual_indexes = [0 for horse in data]

    # we need to keep track for how long animals are next to each other. Yep, dictionary.
    # at the beginning the interaction time is 0 for all of them
    interact_time = {}
    for animal1,we1 in enumerate(actual_indexes):
        for animal2,we2 in enumerate(actual_indexes):
            #refactor this, will you?
            if animal1 < animal2:
                interact_time[(animal1,animal2)] = 0

    while clock_time <= end_time:
        # the edges interacting at this interval
        act_interactions = []
        # we want to update our position at time clock_time
        for animal,index in enumerate(actual_indexes):
            if index >= len(data[animal]):
                continue
            if data[animal][index][1] <= clock_time:
                actual_indexes[animal] +=1

        # we want to estimate our animal positions!
        act_pos = estimate_positions_with_interpolation(data,actual_indexes,clock_time)

        for an1,pos1 in enumerate(act_pos):
            for an2,pos2 in enumerate(act_pos):
                #refactor this, will you?
                if an1 < an2:
                    if pos1 == None or pos2 == None: continue
                    if havm(pos1,pos2) <= max_distance:
                        interact_time[(an1,an2)] += interval
                        if interact_time[(an1,an2)] >= min_time:
                            act_interactions.append((an1,an2))
                    else:
                        interact_time[(an1,an2)] = 0

        # act_interactions MIGHT BE EMPTY! (this way we can also get the time ^^)
        # ATM I DONT NEED IT, so..
        #if len(act_interactions) > 0:
        result.append(act_interactions)

        clock_time += inter


    return result


def sort_tuple(tupl,names):
    return tuple(
            sorted([names.index(el) if el in names else -1 for el in tupl]))



def update_network(nw,to_add,to_remove,names):
    for t in to_add:
        temp = t.split("-")
        if len(temp) > 1:
            nw.add(sort_tuple(temp,names))
        else:
            temp = sorted(t.split("+"))
            for cpl in itertools.combinations(temp, 2):
                nw.add((sort_tuple(cpl,names)))

    for t in to_remove:
        temp = t.split("-")
        if len(temp) > 1:
            cpl = sort_tuple(temp,names)
            if cpl in nw:
                nw.remove(cpl)
        else:
            temp = sorted(t.split("+"))
            for cpl in itertools.combinations(temp, 2):
                cpl = sort_tuple(cpl,names)
                if cpl in nw:
                    nw.remove(cpl)



def create_ground_truth_network_from_data(data,names,interval=5):
    """

    :param data: list of datetime,new edges,toRemove edges
    :param names: the name of the animals. Used to give ids equal everywhere
    :param interval: how many seconds we sample
    :return: a list of list couples of interactions for every interval
    """
    # interval is going to be a timedelta
    inter = timedelta(0,interval)

    # we need to have a starting time.
    clock_time = data[0][0]

    # we need to save the edges of every interval!
    result = []

    # as well as the actual network!
    act_nw = set()

    # we modify the actual network every time the clock time is > than the next row time.
    # so we start from row 0 and end after the last
    act_row = 0
    row_len = len(data)

    # the first time we manually do it.
    update_network(act_nw,data[act_row][1],data[act_row][2],names)

    while act_row < row_len:
        # we add the actual network.
        result.append(list(act_nw))
        if act_row + 1 >= row_len: break;
        clock_time += inter
        while act_row+1 < row_len and data[act_row+1][0] <= clock_time:
            act_row += 1
            update_network(act_nw,data[act_row][1],data[act_row][2],names)

    return result


def quantize_distance(distance,max_distance,granularity):
    if distance >= max_distance:
        return granularity +1
    return int((distance //(max_distance //granularity)) + 1)


def create_input_from_GPS_data(data,interval = 5,max_distance = 100, dist_granularity = 5):
    """

    :param data: a list (classification) of list (times) of positions.
        final line: [(lat,lon),time]. time is a Datetime object
    :param interval: how many seconds we sample
    :param max_distance: the maximum distance before the highest possible distance.
    :param dist_granularity: how many ranges there are between [0,max_distance)
    :return: list (one for each interval) of distances between classification
    """

    # interval is going to be a timedelta
    inter = timedelta(0,interval)

    # we need to have a starting time.
    clock_time = min(list(map(lambda times: times[0][-1],data)))

    # as well as an ending time.
    end_time = max(map(lambda times: times[-1][-1],data))

    # we need to save the edges of every interval!
    result = []

    # we hold different indexes for every horse (every horse has a different timestamp) and update the index
    # when the clock_time is >= the actual time.
    actual_indexes = [0 for horse in data]

    while clock_time <= end_time:
        # the edges interacting at this interval
        act_input = []
        # we want to update our position at time clock_time
        for animal,index in enumerate(actual_indexes):
            if index >= len(data[animal]):
                continue
            if data[animal][index][1] <= clock_time:
                actual_indexes[animal] +=1

        # we want to estimate our animal positions!
        act_pos = estimate_positions_with_interpolation(data,actual_indexes,clock_time)

        for an1,pos1 in enumerate(act_pos):
            for an2,pos2 in enumerate(act_pos):
                #refactor this, will you?
                if an1 < an2:
                    if pos1 is None or pos2 is None:
                        act_input.append(dist_granularity+1)
                    else:
                        act_input.append(quantize_distance(havm(pos1,pos2),max_distance,dist_granularity))


        # act_interactions MIGHT BE EMPTY! (this way we can also get the time ^^)
        # ATM I DONT NEED IT, so..
        #if len(act_interactions) > 0:
        result.append(act_input)

        clock_time += inter

    return result


def transform_list_to_rank(lst,max_dist = 6):
    result = lst[:]
    # now we need to transform the rankings from 1 to whatever, without gaps
    existing_ranks = sorted(list(set(filter(lambda x: x != max_dist,result))))
    for i, rank in enumerate(existing_ranks):
        result = [i + 1 if val == rank else val for val in result]
    return result


def create_relative_input_from_distance(data, dist_granularity = 5):
    """

    :param data: a list (classification) of list (times) of positions.
        final line: [(lat,lon),time]. time is a Datetime object
    :param dist_granularity: the rankings: from 1 to dist_granularity, + 1 for not seen.
    :return the input as relative distances
    """
    result = []
    for line in data:
        result.append(transform_list_to_rank(line,dist_granularity+1))

    return result


def get_edge_index(edge,n_elems):
    """

    :param edge:
    :param n_elems:
    :return: the position of the array sorted like: [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    """
    ind = 0
    for i in range(edge[0]):
        ind += n_elems - i - 1
    ind += edge[1] - edge[0] - 1
    if ind >= n_elems*(n_elems-1)//2:
        print("Index out of bounds! edge: (%d,%d), index: %d" % (edge[0],edge[1],ind))
    return ind


def transform_list_to_matrix_representation(data,n_elems):
    """

    :param data: a list of list of edges
    :param n_elems: the number of nodes. Important because we have to make n*(n-1)/2 values
    :return: for every starting list, a list of n*(n-1)/2 values
    """
    default_row = [0]* (n_elems*(n_elems-1)//2)
    result = []
    for row in data:
        new_row = default_row[:]
        for edge in row:
            if edge[0] >= n_elems or edge[1] >= n_elems:
                print("Unexpected edge: (%d,%d)" % (edge[0],edge[1]))
                continue
            new_row[get_edge_index(edge,n_elems)] = 1
        result.append(new_row)

    return result


def sample_balanced_dataset(X,y):
    """

    :param X: input data
    :param y: list of outputs. WE SUPPOSE IT IS ONLY 0 AND 1!
    :return: a random balanced sample of the dataset. It returns ALL the elements of the minority class.
    """
    # we create a mask to split zeros and ones
    mask = [True if x == 1 else False for x in y]
    # we split our dataset:
    X1 = list(map(lambda z: z[1],filter(lambda x: mask[x[0]],enumerate(X))))
    X0 = list(map(lambda z: z[1],filter(lambda x: not mask[x[0]],enumerate(X))))

    bigger = X1 if len(X1) > len(X0) else X0
    sampling_size = min(len(X1),len(X0))

    # These elements remain unchanged
    new_X = X0[:] if len(X1) > len(X0) else X1[:]
    new_X += random.sample(bigger, sampling_size)
    # we sort these results
    new_y = [0]*sampling_size + [1]*sampling_size if len(X1) > len(X0) else [1]*sampling_size + [0]*sampling_size

    return new_X,new_y

