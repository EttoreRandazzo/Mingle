import random


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


def transform_input_to_individual_based(abs_input,rel_input,raw_input,speed_input):
    """

    :param abs_input: timestamps of absolute distances of all n individuals (it contains k pillar distances too) (size n*(n-1) + k*n)
    :param rel_input: timestamps of ranking distances of all n individuals (it contains k pillar distances too) (size n*(n-1) + k*n)
    :param raw_input: timestamps of raw distances of all n individuals (size n*(n-1))
    :param speed_input: timestamps of speeds for all n individuals (size n)
    :return: a dictionary of (individuals) dictionary of inputs, ex dic[individual]['absolute'] returns a dictionary of absolute distances, where
        each key has all the timestamps regarding its key. ex dic[0]['absolute'][1] = [[1],[2],[1],..]
        dictionary for each individual: absolute, ranking, raw, speed values
    """

    n_individuals = len(speed_input[0])
    n_diadic_dist = n_individuals*(n_individuals-1)
    n_pillars = (len(abs_input[0]) - n_diadic_dist) // n_individuals
    individual_dictionary = {}
    for i in range(n_individuals):
        individual_dictionary[i] = {}
        temp_dic = individual_dictionary[i]
        temp_dic['absolute'] = {}
        temp_dic['ranking'] = {}
        temp_dic['raw'] = {}
        temp_dic['speed'] = []

    for abs_ts in abs_input:
        diadic_dist = abs_ts[:n_diadic_dist]
        pillars_dist = abs_ts[n_diadic_dist:]

        considered_input = 'absolute'
        for i in range(n_individuals):
            personal_input = diadic_dist[(i*(n_individuals-1)):((i+1)*(n_individuals-1))]
            other_id = 0
            # every signal regarding another individual has its own voice in the dictionary, containing all timestamps togheter. ex ts_individual_dict[1] = [2,2,1...]
            for sensed_signal in personal_input:
                if other_id == i: other_id += 1
                # I get the corresponding list
                if other_id not in individual_dictionary[i][considered_input]: individual_dictionary[i][considered_input][other_id] = []
                individual_dictionary[i][considered_input][other_id].append([sensed_signal])
                other_id += 1

            # we add all pillars inputs regarding this individual
            pillars_inputs = []
            for k in range(n_pillars):
                pillars_inputs.append(pillars_dist[k*n_individuals + i])
            if 'pillars' not in individual_dictionary[i][considered_input]: individual_dictionary[i][considered_input]['pillars'] = []
            individual_dictionary[i][considered_input]['pillars'].append(pillars_inputs)

    for rel_ts in rel_input:
        diadic_dist = rel_ts[:n_diadic_dist]
        pillars_dist = rel_ts[n_diadic_dist:]

        considered_input = 'ranking'
        for i in range(n_individuals):
            personal_input = diadic_dist[(i*(n_individuals-1)):((i+1)*(n_individuals-1))]
            other_id = 0
            # every signal regarding another individual has its own voice in the dictionary, containing all timestamps togheter. ex ts_individual_dict[1] = [2,2,1...]
            for sensed_signal in personal_input:
                if other_id == i: other_id += 1
                # I get the corresponding list
                if other_id not in individual_dictionary[i][considered_input]: individual_dictionary[i][considered_input][other_id] = []
                individual_dictionary[i][considered_input][other_id].append([sensed_signal])
                other_id += 1

            # we add all pillars inputs regarding this individual
            pillars_inputs = []
            for k in range(n_pillars):
                pillars_inputs.append(pillars_dist[k*n_individuals + i])
            if 'pillars' not in individual_dictionary[i][considered_input]: individual_dictionary[i][considered_input]['pillars'] = []
            individual_dictionary[i][considered_input]['pillars'].append(pillars_inputs)

    for raw_ts in raw_input:
        diadic_dist = raw_ts[:]

        considered_input = 'raw'
        for i in range(n_individuals):
            personal_input = diadic_dist[(i*(n_individuals-1)):((i+1)*(n_individuals-1))]
            other_id = 0
            # every signal regarding another individual has its own voice in the dictionary, containing all timestamps togheter. ex ts_individual_dict[1] = [2,2,1...]
            for sensed_signal in personal_input:
                if other_id == i: other_id += 1
                # I get the corresponding list
                if other_id not in individual_dictionary[i][considered_input]: individual_dictionary[i][considered_input][other_id] = []
                individual_dictionary[i][considered_input][other_id].append([sensed_signal])
                other_id += 1

    for speed_ts in speed_input:
        for i in range(n_individuals):
            individual_dictionary[i]['speed'].append([speed_ts[i]])

    return individual_dictionary


