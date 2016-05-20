
from classification.nn_classifier import *


def extract_best_configuration_nn(data,results, evaluation_cost, train_percentage = 0.7, markov_alpha = 0.05
                                   ,markov_assumption='sliding_window'):
    """

    :param data: all the observed input.
    :param results: all the observed output
    :param train_percentage: the percentage of [data,results] to be used as train data
    :param markov_alpha: alpha for statistical significant difference of markov orders
    :param evaluation_cost: the cost weights of: [false_positive,false_negative]
    :param markov_assumption: either sliding_window for multiple inputs, or HMM for actual input and previous outcomes.
    :return: (the best model to be used, its markov order)
    """

    markov_order = 0
    padding = [0]*len(results[0]) # used to pad markov orders
    # our cost function to minimize
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    best_model = None
    best_eval = 1e50
    best_conf = None
    best_pred = None
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
            markov_test_r = test_r[:] #test_r[markov_order:]
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

        # the training here uses a file. So we need to save the file model
        act_model = "./models/network_%d.ckpt"%(markov_order)

        train_nn(markov_train_d[:],markov_train_r,act_model,evaluation_cost=evaluation_cost)

        act_pred = predict_nn(markov_test_d,len(markov_test_r[0]),act_model)
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
            best_conf = act_conf
            best_pred = act_pred
        else:
            break
        markov_order += 1

    # we want to print the info of the best one
    print("The best configuration turned out to be with markov order %d:" % (markov_order-1))
    print_result_info(best_conf,best_eval)

    print("Now training with ALL the available data")
    markov_data = create_markov_data(data,markov_order-1) #modifiable. whatever

    train_nn(markov_data,results[(markov_order-1):],best_model,evaluation_cost=evaluation_cost)

    return best_model,markov_order-1


def extract_best_configuration_nn_for_rare_data(data,results, evaluation_cost, train_percentage = 0.7, markov_alpha = 0.05
                                   ,markov_assumption='sliding_window'):
    """

    :param data: all the observed input.
    :param results: all the observed output
    :param train_percentage: the percentage of [data,results] to be used as train data
    :param markov_alpha: alpha for statistical significant difference of markov orders
    :param evaluation_cost: the cost weights of: [false_positive,false_negative]
    :param markov_assumption: either sliding_window for multiple inputs, or HMM for actual input and previous outcomes.
    :return: (the best model to be used, its markov order)
    """

    markov_order = 0
    padding = [0]*len(results[0]) # used to pad markov orders
    # our cost function to minimize
    evaluation_function = cost_ev(evaluation_cost[0],evaluation_cost[1])

    best_model = None
    best_eval = 1e50
    best_pred = None
    train_rows = int(len(data)*train_percentage)
    train_d = data[:train_rows]
    test_d = data[train_rows:]
    train_r = results[:train_rows]
    test_r = results[train_rows:]
    while True: # Do While loop.

        if markov_assumption == 'sliding_window':
            # the data must be transformed depending on the markov assumption.
            # num inputs: len(data[0])*(1 + markov_order)
            t_markov_train_d = create_markov_data(train_d,markov_order)
            markov_test_d = create_markov_data(test_d,markov_order)
            # the first markov_order results cannot be used, for lack of input
            t_markov_train_r = train_r[markov_order:]
            markov_test_r = test_r[:] #test_r[markov_order:]
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

        # DIFFERENT PART: here we remove ALL train cases where there is no 1 in there.
        markov_train_d = []
        markov_train_r = []
        for i in range(len(t_markov_train_r)):
            if 1 in t_markov_train_r[i]:
                markov_train_d.append(t_markov_train_d[i])
                markov_train_r.append(t_markov_train_r[i])

        # the training here uses a file. So we need to save the file model
        act_model = "./models/network_%d.ckpt"%(markov_order)

        train_nn(markov_train_d[:],markov_train_r,act_model,evaluation_cost=evaluation_cost)

        act_pred = predict_nn(markov_test_d,len(markov_test_r[0]),act_model)
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
        else:
            break
        markov_order += 1

    return best_model,markov_order-1

