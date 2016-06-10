
from synthetic_world import *
from classification.edges_classification import *
from classification.nn_edges_classification import *
from sensors import *
import numpy as np

from classification.edges_classification_independence_assumptions import *


# CODE TO CREATE THE WORLD
custom_world = {}
# world boundaries
custom_world["x_range"] = (0,499)
custom_world["y_range"] = (0,499)
# where is the food
custom_world["food_spot"] = (0,400)

# sensors parameters
transmitter_reliability = 0.95 # default 0.95
transmitter_error = 10 #standad deviation 10
receiver_reliability = 0.95 # default 0.95
receiver_error = 15 # standard deviation 15
receiver_range = 100 # up to where it can sense
receiver_precision = 20 # when he considers ranks equal
distance_quanta = 5 # absolute distance values (0,1,2,3,4,...,distance_quanta,distance_quanta+1 (INF))

# set it to true if we want pillars
world_with_pillars = False
if world_with_pillars:
    # sensors and agents instantiation
    x_step = world_info["x_range"][1] // 5
    y_step = world_info["y_range"][1] // 5

    for x in range(x_step,world_info["x_range"][1],x_step):
        for y in range(y_step,world_info["y_range"][1],y_step):
            receiv = Receiver(receiver_reliability,receiver_error,receiver_range,receiver_precision,distance_quanta)
            add_pillar(Pillar((x,y),[receiv]),world=custom_world)


# Agent info
custom_agent_info = {}
custom_agent_info["size"] = 30
custom_agent_info["move_chance"] = 0.3
custom_agent_info["max_movement"] = 45


# These are critical info we need to modify!
normal_transition_sleep = 0.05 # default was 0.05
normal_transition_hungry = 0.05 # default was 0.05
normal_transition_interacting = 0.01 # default was 0.2

normal_adjust_sleep = normal_transition_sleep
normal_adjust_hungry = normal_adjust_sleep + normal_transition_hungry
normal_adjust_interacting = normal_adjust_hungry + normal_transition_interacting

# BY DEFAULT A BUNNY CAN HOLD A STATE AND ITS STATE INFO. GIVEN THAT, HERE IS MY FSM SETTINGS
# here I create a state info (for example for 'NORMAL')
custom_normal_state_info = {'transition_probabilities':(normal_adjust_sleep,normal_adjust_hungry,normal_adjust_interacting),'movement_range': (0,1),'movement_error': 15,
                     'interaction_range': 45,'movement_chance': 0.2}
# I assign this info here, so that I can use it as a default copy to use for this agent any time I get into that state.
custom_agent_info["normal_state_info"] = custom_normal_state_info

custom_sleeping_state_info = {'sleeping_turns_range': (3,21)}
custom_agent_info["sleeping_state_info"] = custom_sleeping_state_info

custom_hungry_state_info = {'eating_range': 20, 'eating_turns_range': (1,5),'movement_range': (10,20),'movement_error': 2 }
custom_agent_info["hungry_state_info"] = custom_hungry_state_info

custom_interacting_state_info = {'movement_range': (10,30),'movement_error': 5, 'dropout_chance': 0.4, 'interact_chance': 0.3,
                          'interaction_range': 30}
custom_agent_info["interacting_state_info"] = custom_interacting_state_info

# finally, I assign the move behavior to the custom  agent
custom_agent_info["move"] = fsa_movement # refer to sensors.py to read how it works
# NOTICE that it will get info from both the world and the agent info.


# instantiate the agents
number_agents = 6 # default 6

for _ in range(number_agents):
    transm = Transmitter(transmitter_reliability,transmitter_error)
    receiv = Receiver(receiver_reliability,receiver_error,receiver_range,receiver_precision,distance_quanta)
    b_pos = (random.randrange(custom_world["x_range"][1] + 1),random.randrange(custom_world["y_range"][1] + 1))
    # a agent can also have custom behavior. IN THAT CASE, CREATE A PRIVATE BUNNY INFO HERE
    add_agent(Agent(b_pos,[transm],[receiv],custom_agent_info),custom_world)



# MODEL SELECTIONS
use_svm = False
use_dt_bag = True
use_dt_boost = False
use_dt_random = False
use_nn = False
use_bs = False

use_dt_bag_independence = True
# MODEL EVALUATION
cost_mispredictions = [1,10] # default [1,3]
input_size = 100000
test_size = 2000
number_repetitions = 5 # how many times we have to run this model to estimate an accurate average.


# END PARAMETERS SELECTION

"""
input_abs,input_rel,input_speed,input_raw,output = create_multiple_timestamp_raw_input_and_output_all_split(input_size,world=custom_world)

individual_dictionary = transform_input_to_individual_based(input_abs,input_rel,input_raw,input_speed)

#print(individual_dictionary)

standardized_dictionary = standardize_individual_based_dictionary(individual_dictionary)

#print(standardized_dictionary)

extract_best_configuration_dt_independence(standardized_dictionary,output,[1,3])

"""




evaluations = {} # a dict of list of results for each model.

if use_svm:
    evaluations['SVM'] = []
if use_dt_bag:
    evaluations['Bagging'] = []
if use_dt_boost:
    evaluations['Boosting'] = []
if use_dt_random:
    evaluations['Random Forest'] = []
if use_nn:
    evaluations['NN'] = []
if use_bs:
    evaluations['Baseline'] = []
if use_dt_bag_independence:
    evaluations['Bagging Independence'] = []

# WE WANT A LOGGER
logger = open('logger.txt','w')

# we store every result
for repetition in range(number_repetitions):
    print("\nREPETITION %d:\n" % (repetition+1))

    input_abs,input_rel,input_speed,input_raw,output = create_multiple_timestamp_raw_input_and_output_all_split(input_size,world=custom_world)
    test_abs,test_rel,test_speed,test_raw,test_output = create_multiple_timestamp_raw_input_and_output_all_split(test_size,world=custom_world)

    # Independence ones, we have booleans to understand what we want
    train_dictionary = standardize_individual_based_dictionary(transform_input_to_individual_based(input_abs,input_rel,input_raw,input_speed),True,False,False,True)
    test_dictionary = standardize_individual_based_dictionary(transform_input_to_individual_based(test_abs,test_rel,test_raw,test_speed),True,False,False,True)

    input_target = []
    for i in range(len(input_abs)):
        input_target.append(input_abs[i]+input_speed[i])


    test_target = []
    for i in range(len(test_abs)):
        test_target.append(test_abs[i]+test_speed[i])

    if use_svm:
        print("TRAINING SVM")
        svm_model, svm_markov = extract_best_configuration_svm(input_target,output,evaluation_cost=cost_mispredictions)

    if use_dt_bag:
        print("TRAINING DT BAG")
        dt_bag_model, dt_bag_markov = extract_best_configuration_dt(input_target,output,evaluation_cost=cost_mispredictions,method='bagging')

    if use_dt_boost:
        print("TRAINING DT BOOST")
        dt_boost_model, dt_boost_markov = extract_best_configuration_dt(input_target,output,evaluation_cost=cost_mispredictions,method='boosting')

    if use_dt_random:
        print("TRAINING DT RANDOM FOREST")
        dt_random_model, dt_random_markov = extract_best_configuration_dt(input_target,output,evaluation_cost=cost_mispredictions,method='random_forest')

    if use_nn:
        print("TRAINING NN")
        nn_model, nn_markov = extract_best_configuration_nn(input_target,output,evaluation_cost=cost_mispredictions)

    # BASELINE!
    if use_bs:
        print("TRAINING BASELINE")
        max_time = 2
        bl_dist, bl_time = extract_best_configuration_baseline_distance_and_time(input_raw,output,cost_mispredictions,number_agents,receiver_range,max_time,INF_TOKEN)


    # INDEPENDENCE ASSUMPTIONS
    if use_dt_bag_independence:
        print("TRAINING DT BAG INDEPENDENCE")
        dt_bag_model_independence = extract_best_configuration_dt_independence(train_dictionary,output,evaluation_cost=cost_mispredictions,method='bagging')


    # TESTING THE RESULTS!
    print("EVALUATING WITH TEST DATA:")
    evaluation_function = cost_ev(cost_mispredictions[0],cost_mispredictions[1])
    # the input data of that order is stored in a dictionary.
    input_order = {0: test_target}
    input_order_independence = {0: test_dictionary}
    padding = [0]*len(test_output[0])

    if use_svm:
        print("SVM EVALUATION:")
        if svm_markov not in input_order:
            input_order[svm_markov] = create_markov_data(test_target,svm_markov)
        act_order_i = input_order[svm_markov]

        act_pred = predict_svms(act_order_i[:],svm_model)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(svm_markov):
            act_pred.insert(0,padding[:])

        act_conf = compute_confusion_matrix(act_pred,test_output)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['SVM']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Markov order %d:" % svm_markov)
        print_result_info(act_conf,act_eval)


    if use_dt_bag:
        print("DT BAG EVALUATION:")
        if dt_bag_markov not in input_order:
            input_order[dt_bag_markov] = create_markov_data(test_target,dt_bag_markov)
        act_order_i = input_order[dt_bag_markov]

        act_pred = predict_dts(act_order_i[:],dt_bag_model)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(dt_bag_markov):
            act_pred.insert(0,padding[:])

        act_conf = compute_confusion_matrix(act_pred,test_output)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['Bagging']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Markov order %d:" % dt_bag_markov)
        print_result_info(act_conf,act_eval)


    if use_dt_boost:
        print("DT BOOST EVALUATION:")
        if dt_boost_markov not in input_order:
            input_order[dt_boost_markov] = create_markov_data(test_target,dt_boost_markov)
        act_order_i = input_order[dt_boost_markov]

        act_pred = predict_dts(act_order_i[:],dt_boost_model)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(dt_boost_markov):
            act_pred.insert(0,padding[:])

        act_conf = compute_confusion_matrix(act_pred,test_output)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['Boosting']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Markov order %d:" % dt_boost_markov)
        print_result_info(act_conf,act_eval)


    if use_dt_random:
        print("DT RANDOM FOREST EVALUATION:")
        if dt_random_markov not in input_order:
            input_order[dt_random_markov] = create_markov_data(test_target,dt_random_markov)
        act_order_i = input_order[dt_random_markov]

        act_pred = predict_dts(act_order_i[:],dt_random_model)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(dt_random_markov):
            act_pred.insert(0,padding[:])

        act_conf = compute_confusion_matrix(act_pred,test_output)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['Random Forest']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Markov order %d:" % dt_random_markov)
        print_result_info(act_conf,act_eval)


    if use_nn:
        print("NN EVALUATION:")
        if nn_markov not in input_order:
            input_order[nn_markov] = create_markov_data(test_target,nn_markov)
        act_order_i = input_order[nn_markov]

        act_pred = predict_nn(act_order_i[:],len(test_output[0]),nn_model)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(nn_markov):
            act_pred.insert(0,padding[:])

        act_conf = compute_confusion_matrix(act_pred,test_output)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['NN']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Markov order %d:" % nn_markov)
        print_result_info(act_conf,act_eval)


    if use_bs:
        print("BASELINE EVALUATION:")

        act_pred = predict_threshold_with_time(test_raw,number_agents,bl_dist,bl_time,INF_TOKEN)

        act_conf = compute_confusion_matrix(act_pred,test_output)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['Baseline']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Distance used: %d, time considered: %d" % (bl_dist, bl_time))
        print_result_info(act_conf,act_eval)

    if use_dt_bag_independence:
        print("DT BAG INDEPENDENCE EVALUATION:")
        dt_bag_independence_markov = dt_bag_model_independence[2]
        if dt_bag_independence_markov not in input_order_independence:
            input_order_independence[dt_bag_independence_markov] = create_markov_data_independence_assumption(test_dictionary,dt_bag_independence_markov)
        act_order_i = input_order_independence[dt_bag_independence_markov]

        act_pred = predict_dts_independence(act_order_i,dt_bag_model_independence)
        # for now, models of order greater than 0, remove test cases. we 0 pad them.
        for _ in range(dt_bag_independence_markov):
            act_pred.insert(0,padding[:])

        act_conf = compute_confusion_matrix(act_pred,test_output)   #test_svms(markov_test_d,markov_test_r,act_model)
        act_eval = evaluation_function(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        act_acc = accuracy_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_prec = precision_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_rec = recall_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])
        act_f1 = f1_ev(act_conf[0][0],act_conf[1][1],act_conf[1][0],act_conf[0][1])

        # adding the info to the dictionary
        eval_list = evaluations['Bagging Independence']
        eval_list.append([act_eval,act_acc,act_prec,act_rec,act_f1])

        # printing info
        print("Markov order %d:" % dt_bag_independence_markov)
        print_result_info(act_conf,act_eval)

print("PRINTING FINAL RESULTS")
print(evaluations)
def mean(a):
    return sum(a) / len(a)

for key, value in evaluations.items():
    # we compute average
    average_evals = list(map(np.mean,zip(*value)))
    # then variance
    #variance_evals = list(map(mean,map(lambda x: x-,zip(*value))))
    #finally standard deviation
    std_evals = list(map(np.std,zip(*value)))

    string_info = "Model %s -> cost: %f +- %f, acc: %f +- %f, prec: %f +- %f, reca: %f +- %f, f1:%f +- %f" % (key,average_evals[0],std_evals[0],average_evals[1],std_evals[1],average_evals[2],std_evals[2],average_evals[3],std_evals[3],average_evals[4],std_evals[4])
    print(string_info)
    logger.write(string_info+"\n")

string_info = "Parameters:\nInput size %d\nNumber of agents: %d\ninteraction_likelihood: %f\nhasPillars: %r\ntransmitter_reliability: %f\ntransmitter_error: %f\nreceiver_reliability: %f\nreceiver_error: %f" % (input_size,number_agents,normal_transition_interacting,world_with_pillars,transmitter_reliability,transmitter_error,receiver_reliability,receiver_error)
print(string_info)
logger.write("\n"+string_info+"\n")

# this is for Latex tables!

logger.write("\n"+"LATEX TABLE FORMAT:"+"\n")
for key, value in evaluations.items():
    average_evals = list(map(mean,zip(*value)))
    std_evals = list(map(np.std,zip(*value)))
    string_info = "%s & %d \pm %d & %f \pm %f & %f \pm %f & %f \pm %f & %f \pm %f\\\\\n\\hline" % (key,average_evals[0],std_evals[0],average_evals[1],std_evals[1],average_evals[2],std_evals[2],average_evals[3],std_evals[3],average_evals[4],std_evals[4])
    print(string_info)
    logger.write(string_info+"\n")


#CLOSE THE LOGGER!
logger.close()