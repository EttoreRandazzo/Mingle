
from classification.classifiers import *


def extract_best_configuration_svm(data,results,evaluation_cost, train_percentage = 0.7, markov_alpha = 0.05,
                                   model='svm',markov_assumption='sliding_window'):
    """

    :param data: all the observed input.
    :param results: all the observed output
    :param train_percentage: the percentage of [data,results] to be used as train data
    :param markov_alpha: alpha for statistical significant difference of markov orders
    :param evaluation_function: the function used to evaluate the results. must take as input: tp,tn,fp,fn
    :param model: the model to use
    :param markov_assumption: either sliding_window for multiple inputs, or HMM for actual input and previous outcomes.
    :return: (the best model to be used, its markov order)
    """

    markov_order = 0
    padding = [0]*len(results[0]) # used to pad markov orders
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    best_model = None
    best_eval = 1e50
    best_pred = None
    best_conf = None
    train_rows = int(len(data)*train_percentage)
    train_d = data[:train_rows]
    test_d = data[train_rows:]
    train_r = results[:train_rows]
    test_r = results[train_rows:]
    while True: # Do While loop.

        if markov_assumption == 'sliding_window':
            # the data must be transformed depending on the markov assumption.
            # num inputs: len(data[0])*(1 + markov_order)
            markov_train_d = create_markov_data(train_d,markov_order)
            markov_test_d = create_markov_data(test_d,markov_order)
            # the first markov_order results cannot be used, for lack of input
            markov_train_r = train_r[markov_order:]
            markov_test_r = test_r[:]  #test_r[markov_order:]
        elif markov_assumption == 'HMM':
            # the data is created on the go, but for the first ones..
            markov_train_d = train_d[:]
            markov_test_d = test_d[:]
            # we put zeroes where we don't have previous predictions:
            for i in range(markov_order):
                markov_train_d[i] += [0]*len(results[0])*(markov_order-i)
                markov_test_d[i] += [0]*len(results[0])*(markov_order-i)
            markov_train_r = train_r[:]
            markov_test_r = test_r[:]
        else:
            raise NameError('Unsupported markov assumption '+markov_assumption)

        if model == 'svm':
            act_model = train_svms(markov_train_d[:],markov_train_r,evaluation_cost,model='svm',markov_assumption=markov_assumption,markov_order=markov_order)
            act_pred = predict_svms(markov_test_d[:],act_model,model='svm',markov_assumption=markov_assumption,markov_order=markov_order)
        elif model == 'structured_svm':
            act_model = train_structured_svm(markov_train_d,markov_train_r)
            act_pred = predict_structured_svm(markov_test_d,act_model)
            # for now, models of order greater than 0, remove test cases. we 0 pad them.
            for _ in range(markov_order):
                act_pred.insert(0,padding[:])
        else:
            raise NameError('Unsupported model '+model)

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
    markov_data = create_markov_data(data,markov_order-1)

    best_model = train_svms(markov_data,results[(markov_order-1):],evaluation_cost,model='svm',markov_assumption=markov_assumption,markov_order=markov_order)


    return best_model,markov_order-1


def extract_best_configuration_dt(data,results,evaluation_cost,method= 'bagging', train_percentage = 0.7, markov_alpha = 0.05):
    """Decision Tree model extractor

    :param data: all the observed input.
    :param results: all the observed output
    :param method: bagging,random_forest,boosting
    :param train_percentage: the percentage of [data,results] to be used as train data

    :return: (the best model to be used, its markov order)
    """

    markov_order = 0
    padding = [0]*len(results[0]) # used to pad markov orders
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    best_model = None
    best_eval = 1e50
    best_pred = None
    best_conf = None
    train_rows = int(len(data)*train_percentage)
    train_d = data[:train_rows]
    test_d = data[train_rows:]
    train_r = results[:train_rows]
    test_r = results[train_rows:]
    while True: # Do While loop.

        # the data must be transformed depending on the markov assumption.
        # num inputs: len(data[0])*(1 + markov_order)
        markov_train_d = create_markov_data(train_d,markov_order)
        markov_test_d = create_markov_data(test_d,markov_order)
        # the first markov_order results cannot be used, for lack of input
        markov_train_r = train_r[markov_order:]
        markov_test_r = test_r[:] #test_r[markov_order:]

        act_model = train_dts(markov_train_d[:],markov_train_r,method)
        act_pred = predict_dts(markov_test_d[:],act_model)
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
    markov_data = create_markov_data(data,markov_order-1)

    best_model = train_dts(markov_data[:],results[(markov_order-1):])


    return best_model,markov_order-1


def extract_best_configuration_baseline_distance(data,results, evaluation_cost,n_nodes, max_distance_sensed, INF_TOKEN, train_percentage = 0.7):
    """

    :param data: all the observed input.
    :param results: all the observed output
    :param train_percentage: the percentage of [data,results] to be used as train data
    :param evaluation_cost: the cost weights of: [false_positive,false_negative]
    :param n_ndoes: the number of nodes of the graph
    :param max_distance_sensed: the maximum distance we consider
    :param INF_TOKEN: a token that means 'not sensed'
    :param train_percentage: how much of data to be used as train.
    :return: distance_treshold
    """

    # our cost function to minimize
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    best_distance = max_distance_sensed
    best_eval = 1e50
    best_conf = []
    train_rows = int(len(data)*train_percentage)
    train_d = data[:train_rows]
    test_d = data[train_rows:]
    train_r = results[:train_rows]
    test_r = results[train_rows:]

    # This time we do a bruteforce search from 0 to max_distance_sensed
    left, right = 0, max_distance_sensed

    for act_distance in range(max_distance_sensed):
        # the training here uses a file. So we need to save the file model

        act_pred = predict_threshold(test_d,n_nodes,act_distance,INF_TOKEN)

        print(act_pred[:30])
        print(test_r[:30])
        act_conf = compute_confusion_matrix(act_pred,test_r)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        # printing info
        print("Distance used :%d" % act_distance)
        print_result_info(act_conf,act_eval)

        if act_eval < best_eval:
            best_eval = act_eval
            print("Found a better distance: %d" % act_distance)
            best_distance = act_distance
            best_conf = act_conf

    # we want to print the info of the best one
    print("The best configuration turned out to be with distance %d:" % best_distance)
    print_result_info(best_conf,best_eval)

    return best_distance


def extract_best_configuration_baseline_distance_and_time(data,results, evaluation_cost,n_nodes, max_distance_sensed,max_time, INF_TOKEN, train_percentage = 0.7):
    """

    :param data: all the observed input.
    :param results: all the observed output
    :param train_percentage: the percentage of [data,results] to be used as train data
    :param evaluation_cost: the cost weights of: [false_positive,false_negative]
    :param n_ndoes: the number of nodes of the graph
    :param max_distance_sensed: the maximum distance we consider
    :param max_time: the maximum previous time assumption we consider
    :param INF_TOKEN: a token that means 'not sensed'
    :param train_percentage: how much of data to be used as train.
    :return: distance_treshold, time_threshold
    """

    # our cost function to minimize
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    best_distance = max_distance_sensed
    best_time = -1
    best_eval = 1e50
    best_conf = []
    train_rows = int(len(data)*train_percentage)
    train_d = data[:train_rows]
    test_d = data[train_rows:]
    train_r = results[:train_rows]
    test_r = results[train_rows:]

    # This time we do a bruteforce search from 0 to max_distance_sensed
    left, right = 0, max_distance_sensed

    for act_distance, act_time in ((x, y) for x in range(max_distance_sensed+1) for y in range(max_time+1)):
        # the training here uses a file. So we need to save the file model

        act_pred = predict_threshold_with_time(test_d,n_nodes,act_distance,act_time,INF_TOKEN)
        #print(act_pred[:30])
        #print(test_r[:30])
        act_conf = compute_confusion_matrix(act_pred,test_r)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        # printing info
        """
        print("Distance used: %d, time considered: %d" % (act_distance, act_time))
        print_result_info(act_conf,act_eval)
        """

        if act_eval < best_eval:
            best_eval = act_eval
            #print("Found a better couple: distance: %d, time: %d" % (act_distance,act_time))
            best_distance = act_distance
            best_time = act_time
            best_conf = act_conf


    # we want to print the info of the best one
    print("The best configuration turned out to be with distance %d and time %d:" % (best_distance,best_time))
    print_result_info(best_conf,best_eval)

    return best_distance, best_time


"""
# Sample usage
import random

data_len = 1000
transposed = []
transposed.append([random.randrange(1,5,1) for _ in range(data_len)])
transposed.append([random.randrange(1,3,1) for _ in range(data_len)])
transposed.append([random.randrange(1,6,1) for _ in range(data_len)])
transposed = np.array(transposed).transpose().tolist()

results = [[random.randrange(0,2) for _ in range(data_len)],[random.randrange(0,2) for _ in range(data_len)]]
results = np.array(results).transpose().tolist()

model = extract_best_configuration_svm(transposed,results)

print(model)
"""


"""
predictions = [[0,0,0,1,0],[0,1,0,1,0],[1,0,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[1,0,1,1,0]]
ground_truth = [[1,0,0,1,0],[0,1,1,1,0],[1,0,1,1,0],[1,1,0,1,0],[0,0,0,1,0],[1,0,1,0,0]]
print(adjust_network_with_markov_optimized(predictions,ground_truth,1,0.4,0.5))
"""