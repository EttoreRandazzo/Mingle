from classification.classifiers import *

def create_markov_data_independence_assumption(data,order):
    """

    :param data: inputs
    :param order: the order of the markov chain
    :return: a new data with markov assumptions. num rows: len(data) - order.
    """
    if order <= 0: return data

    result = {}
    for key, value in data.items():
        actual_dict = {}
        result[key] = actual_dict

        for key2, value2 in value.items():
            actual_list = []
            actual_dict[key2] = actual_list
            for i in range(order,len(value2)):
                row = []
                for step in range(0,order + 1):
                    row += value2[i-step]
                actual_list.append(row)

    return result


def get_number_of_individuals(n_targets):
    """

    :param n_targets: the number of output targets, considering an undirected graph
    :return: the number of individuals
    """
    n_individuals = 2
    while n_individuals*(n_individuals-1) < n_targets*2 : n_individuals += 1

    if n_individuals*(n_individuals-1) != n_targets*2: raise ValueError('Targets size is inconsistent: there is no natural n such that n*(n-1)/2 == len(targets[0]) AKA %d' % (n_targets))

    return n_individuals



def train_dts_independence(observations,targets,method='bagging'):
    """Trains a decision tree for each output

    :param observations: our train dataset, it must be a dictionary of individuals
    :param targets: multiple target variables.
    :param method: bagging,random_forest,boosting
    :return: the dt models in a list, one for each target variable
    """
    n_targets = len(targets[0])
    # we need to know the number of individuals n such that n*(n-1)/2 = n_targets
    # we do that iteratively
    n_individuals = get_number_of_individuals(n_targets)

    tars = np.array(targets)

    # we want to use all training data available for only one classifier, so we create data from all pairs
    training_input = []
    training_output = []
    for i in range(n_individuals-1):
        for j in range(i+1,n_individuals):
            # basically, for each pair
            # we get individual values AND pairwise values
            i_general = observations[i]['general']
            j_general = observations[j]['general']
            i_senses_j = observations[i][j]
            j_senses_i = observations[j][i]
            for timestamp in range(len(i_general)):
                # THE ORDER IS EXTREMELY IMPORTANT (to be consistent)
                training_input.append(i_general[timestamp] + j_general[timestamp] + i_senses_j[timestamp] + j_senses_i[timestamp])

            # also, we add the output, ordered
            pair_targets = tars[:,get_edge_index((i,j),n_individuals)].tolist()
            training_output += pair_targets

    # now we simply train

    dt = None
    if method == 'bagging': dt = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=100,max_samples=0.5, max_features=1.)
    elif method == 'random_forest': dt = RandomForestClassifier(n_estimators=100)
    elif method == 'boosting': dt = AdaBoostClassifier(n_estimators=100)
    else: dt = tree.DecisionTreeClassifier()
    # the dt cannot be trained if the outputs are all equal. In that case, we create a fake dt
    if len(set(training_output)) > 1:
        # We want to have a balanced data set while training.
        bal_observations, bal_tar = sample_balanced_dataset(training_input,training_output) #from data_manipulation
        dt.fit(bal_observations,bal_tar)
    else:
        dt = FakeClassifier(training_output[0])
    return dt


def predict_dts_independence(observations,model):
    """

    :param observations:
    :param model: must be a tuple with (classifier, number of individuals)
    :return:
    """

    n_individuals = model[1]
    res = []
    for i in range(n_individuals-1):
        for j in range(i+1,n_individuals):
            # basically, for each pair
            pair_input = []
            # we get individual values AND pairwise values
            i_general = observations[i]['general']
            j_general = observations[j]['general']
            i_senses_j = observations[i][j]
            j_senses_i = observations[j][i]
            for timestamp in range(len(i_general)):
                # THE ORDER IS EXTREMELY IMPORTANT (to be consistent)
                pair_input.append(i_general[timestamp] + j_general[timestamp] + i_senses_j[timestamp] + j_senses_i[timestamp])

            # also, we add the output, ordered
            pair_targets = model[0].predict(pair_input)
            res.append(pair_targets)

    res = np.array(res)
    res = res.transpose()
    return res.tolist()