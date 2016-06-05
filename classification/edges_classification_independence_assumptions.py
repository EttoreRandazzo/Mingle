from classification.classifiers_independence_assumptions import *

def extract_best_configuration_dt_independence(data,results,evaluation_cost,method= 'bagging', train_percentage = 0.7, markov_alpha = 0.05):
    """Decision Tree model extractor

    :param data: all the observed input as a dictionary
    :param results: all the observed output
    :param method: bagging,random_forest,boosting
    :param train_percentage: the percentage of [data,results] to be used as train data

    :return: (the best model to be used, its markov order)
    """

    markov_order = 0
    padding = [0]*len(results[0]) # used to pad markov orders
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    # we also need to know the number of individuals
    n_individuals = get_number_of_individuals(len(results[0]))

    best_model = None
    best_eval = 1e50
    best_pred = None
    best_conf = None
    train_rows = int(len(data[0][1])*train_percentage)
    # splitting here requires to create two different dictionaries
    train_d = {}
    test_d = {}
    for key, value in data.items():
        new_individual_dict_train = {}
        new_individual_dict_test = {}
        train_d[key] = new_individual_dict_train
        test_d[key] = new_individual_dict_test

        for key2, actual_value in value.items():
            new_individual_dict_train[key2] = actual_value[:train_rows]
            new_individual_dict_test[key2] = actual_value[train_rows:]

    train_r = results[:train_rows]
    test_r = results[train_rows:]

    while True: # Do While loop.

        # the data must be transformed depending on the markov assumption.
        # num inputs: len(data[0])*(1 + markov_order)
        markov_train_d = create_markov_data_independence_assumption(train_d,markov_order)
        markov_test_d = create_markov_data_independence_assumption(test_d,markov_order)
        # the first markov_order results cannot be used, for lack of input
        markov_train_r = train_r[markov_order:]
        markov_test_r = test_r[:] #test_r[markov_order:]

        # WE MIGHT NEED TO DEEPCOPY DICTIONARIES HERE
        act_model = train_dts_independence(markov_train_d,markov_train_r,method), n_individuals , markov_order
        act_pred = predict_dts_independence(markov_test_d,act_model)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(markov_order):
            act_pred.insert(0,padding[:])

        #print(act_pred[:30])
        #print(markov_test_r[:30])

        act_conf = compute_confusion_matrix(act_pred,markov_test_r)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        # printing info
        print("Markov order %d:" % markov_order)
        print_result_info(act_conf,act_eval)

        if act_eval < best_eval and (markov_order == 0 or perform_multiple_comparison_stat(act_pred,best_pred[1:],markov_alpha)):
            best_eval = act_eval
            best_model = act_model
            best_pred = act_pred
            best_conf = act_conf
        else:
            break
        markov_order += 1

    # we want to print the info of the best one
    print("The best configuration turned out to be with markov order %d:" % (markov_order-1))
    print_result_info(best_conf,best_eval)

    print("Now training with ALL the available data")
    markov_data = create_markov_data_independence_assumption(data,markov_order-1)

    best_model = train_dts_independence(markov_data,results[(markov_order-1):]), n_individuals, markov_order-1

    return best_model