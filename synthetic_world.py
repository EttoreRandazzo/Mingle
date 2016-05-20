from data_manipulation import *



world_info = {}

world_info["x_range"] = (0,499)
world_info["y_range"] = (0,499)

world_info["food_spot"] = (0,400)


def add_agent(b, world = world_info):
    if "bunnies" not in world:
        world["bunnies"] = [b]
    else:
        world["bunnies"].append(b)


def add_pillar(r, world = world_info):
    if "pillars" not in world:
        world["pillars"] = [r]
    else:
        world["pillars"].append(r)


def create_timestamp(world = world_info):
    """

    creates a timestamp list of the world
    :param world:
    :return: a list of inputs
    """

    # first, I look at which bunnies actually transmit something
    transmit = []
    for agent in world["bunnies"]:
        transmit += agent.transmit()

    # now I create the matrix from the receivers
    result = []
    for agent in world["bunnies"]:
        result += agent.compute_abs_and_rel_distances(transmit)
    """for pillar in world["pillars"]:
        result += pillar.compute_abs_and_rel_distances(transmit)"""

    # finally, I add the speed information
    speeds = [b.speed for b in world["bunnies"]]
    result += speeds
    return result


def create_timestamp_raw(world = world_info):
    """

    creates a timestamp list of the world, with only input its raw data
    :param world:
    :return: a list of inputs of raw data
    """

    # first, I look at which bunnies actually transmit something
    transmit = []
    for agent in world["bunnies"]:
        transmit += agent.transmit()

    # now I create the matrix from the receivers
    result = []
    for agent in world["bunnies"]:
        result += agent.compute_raw_distances(transmit)
    """for pillar in world["pillars"]:
        result += pillar.compute_abs_and_rel_distances(transmit)"""

    return result

def create_timestamp_all(world = world_info):
    """

    creates a timestamp list of the world
    :param world:
    :return: a list of inputs and a list of raw inputs
    """

    # first, I look at which bunnies actually transmit something
    transmit = []
    for agent in world["bunnies"]:
        transmit += agent.transmit()

    # now I create the matrix from the receivers
    result = []
    result_raw = []
    for agent in world["bunnies"]:
        ts = agent.compute_abs_rel_raw_distances(transmit)
        result += ts[0]
        result_raw += ts[1]
    if "pillars" in world:
        for pillar in world["pillars"]:
            # raw distances here are not needed because we don't use such information for our baseline.
            result += pillar.compute_abs_and_rel_distances(transmit)

    # finally, I add the speed information
    speeds = [b.speed for b in world["bunnies"]]
    result += speeds
    return result,result_raw

def create_timestamp_all_split(world = world_info):
    """

    creates a timestamp list of the world
    :param world:
    :return: a list of abs inputs, a list of rel inputs, a list of speeds and a list of raw inputs
    """

    # first, I look at which bunnies actually transmit something
    transmit = []
    for agent in world["bunnies"]:
        transmit += agent.transmit()

    # now I create the matrix from the receivers
    result_abs = []
    result_raw = []
    result_rel = []
    for agent in world["bunnies"]:
        ts = agent.compute_abs_rel_raw_distances_split(transmit)
        result_abs += ts[0]
        result_rel += ts[1]
        result_raw += ts[2]
    if "pillars" in world:
        for pillar in world["pillars"]:
            # raw distances here are not needed because we don't use such information for our baseline.
            ts = pillar.compute_abs_and_rel_distances_split(transmit)
            result_abs += ts[0]
            result_rel += ts[1]

    # finally, I add the speed information
    speeds = [b.speed for b in world["bunnies"]]
    return result_abs,result_rel,speeds,result_raw


def create_labels_timestamp_list(world = world_info):
    """

    :param world:
    :return: a list of interactions (edges). to transform it to matrix use classification.data_manipulation 'transform_list_to_matrix_representation
        for all the labels altogether
    """
    bunnies = world['bunnies']
    # we first create a set of tuple interactions for the bunnies, tuples are sorted.
    interactions = set()

    for i,b1 in enumerate(bunnies):
        if b1.state == 'INTERACTING':
            interacting_with = b1.state_info['active_interactions']
            for b2 in interacting_with:
                j = bunnies.index(b2)
                if i == j:
                    print("EQUAL INDEXES FOR INTERACTION?!?!")
                    continue
                interactions.add(tuple(sorted([i,j])))

    return list(interactions)

def move_bunnies(world = world_info):
    for agent in world["bunnies"]:
        agent.move(world)


def create_multiple_timestamp_raw_input_and_output(time,sampling_step=1,world = world_info):
    matrix = []
    matrix_raw = []
    labels = []
    for _ in range(time):
        ts = create_timestamp_all(world)
        matrix.append(ts[0])
        matrix_raw.append(ts[1])
        labels.append(create_labels_timestamp_list(world))
        for _ in range(sampling_step):
            move_bunnies(world)
    return matrix,matrix_raw,transform_list_to_matrix_representation(labels,len(world['bunnies']))


def create_multiple_timestamp_raw_input_and_output_all_split(time,sampling_step=1,world = world_info):
    """
     same as create_multiple_timestamp_raw_input_and_output but returns stuff separated
    :param time:
    :param sampling_step:
    :param world:
    :return: abs,rel,speed,raw,labels
    """
    matrix_abs = []
    matrix_rel =[]
    matrix_speed = []
    matrix_raw = []
    labels = []
    for _ in range(time):
        ts = create_timestamp_all_split(world)
        matrix_abs.append(ts[0])
        matrix_rel.append(ts[1])
        matrix_speed.append(ts[2])
        matrix_raw.append(ts[3])
        labels.append(create_labels_timestamp_list(world))
        for _ in range(sampling_step):
            move_bunnies(world)
    return matrix_abs,matrix_rel,matrix_speed,matrix_raw,transform_list_to_matrix_representation(labels,len(world['bunnies']))


def split_input(input,n_animals):
    """
        :return abs_distances,rel_distances,speeds
    """
    abs_dist = []
    rel_dist = []
    speeds = []
    #n_distances = n_animals*(n_animals-1)//2
    for line in input:
        speeds.append(line[-n_animals:])
        for an in range(n_animals):
            abs_dist.append(line[2*an*(n_animals-1):(2*an+1)*(n_animals-1)])
            rel_dist.append(line[(2*an+1)*(n_animals-1):2*(an+1)*(n_animals-1)])

    return abs_dist,rel_dist,speeds
