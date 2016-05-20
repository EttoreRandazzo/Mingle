"""
    this is an example of classification with our classifiers.
    We first create our own synthetic data, then we train our classifiers
"""


from synthetic_world import *
from classification.edges_classification import *
from sensors import *


# We want to synthesize data, so we create a world
custom_world = {}
# world boundaries
custom_world["x_range"] = (0,499)
custom_world["y_range"] = (0,499)
# where is the food
custom_world["food_spot"] = (0,400)


# We decide the parameters of our sensors, that is, the noise of our input
transmitter_reliability = 0.95
transmitter_error = 10 #standad deviation 10
receiver_reliability = 0.95
receiver_error = 15 # standard deviation 15
receiver_range = 100 # up to where it can sense
receiver_precision = 20 # when he considers ranks equal
distance_quanta = 5 # absolute distance values (0,1,2,3,4,...,distance_quanta,distance_quanta+1 (INF))

# set it to true if we want beacons. In this example we will NOT have beacons
world_with_pillars = False
if world_with_pillars:
    # sensors and bunnies instantiation
    x_step = world_info["x_range"][1] // 5
    y_step = world_info["y_range"][1] // 5

    for x in range(x_step,world_info["x_range"][1],x_step):
        for y in range(y_step,world_info["y_range"][1],y_step):
            receiv = Receiver(receiver_reliability,receiver_error,receiver_range,receiver_precision,distance_quanta)
            add_pillar(Pillar((x,y),[receiv]),world=custom_world)


# We create animal behavior
custom_agent_info = {}
custom_agent_info["size"] = 30
custom_agent_info["move_chance"] = 0.3
custom_agent_info["max_movement"] = 45


# These are critical info we need to modify! how likely is the animal to go into that state
normal_transition_sleep = 0.05 # default was 0.05
normal_transition_hungry = 0.05 # default was 0.05
normal_transition_interacting = 0.2 # default was 0.2

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

# finally, I assign the move behavior to the custom agent
custom_agent_info["move"] = fsa_movement # refer to sensors.py to read how it works
# NOTICE that it will get info from both the world and the agent info.


# instantiate the bunnies
number_bunnies = 6 # default 6

for _ in range(number_bunnies):
    transm = Transmitter(transmitter_reliability,transmitter_error)
    receiv = Receiver(receiver_reliability,receiver_error,receiver_range,receiver_precision,distance_quanta)
    b_pos = (random.randrange(custom_world["x_range"][1] + 1),random.randrange(custom_world["y_range"][1] + 1))
    # a agent can also have custom behavior. IN THAT CASE, CREATE A PRIVATE BUNNY INFO HERE
    add_agent(Agent(b_pos,[transm],[receiv],custom_agent_info),custom_world)


# We want to create some data
input_size = 10000
test_size = 2000
"""
    the very long-named function create_multiple_timestamp_raw_input_and_output_all_split (because we have other less complete functions which would be faster)
    takes as input the number of time frames we want to create and a custom world where to take info AND modify it (bunnies move)
    it returns 5 elements: the first 4 are inputs, the final is output
        absolute_distances: discrete directed distances between animals (we have n*(n-1) distances each time frame). When not sensed a signal, it gives max_distance + 1
        ranking_distances: directed ranking distances starting from 1. When not sensed a signal, it gives act_max_distance + 1
        animal_speed: speed of the animal in that time frame
        raw_distances: unprocessed directed distances (in cm). When not sensed a signal, it gives INF_TOKEN
        interactions: undirected interactions between animals (0 is not interacting, 1 is interacting) (we have n*(n-1)/2 outputs each time frame)
"""
input_abs,input_rel,input_speed,input_raw,input_output = create_multiple_timestamp_raw_input_and_output_all_split(input_size,world=custom_world)
test_abs,test_rel,test_speed,test_raw,test_output = create_multiple_timestamp_raw_input_and_output_all_split(test_size,world=custom_world)


# Now we want to create a model and classify.

# To actually evaluate, we need to have a cost matrix to understand which model is best.
# cost_mispredictions is [false_positive_cost, false_negative_cost], while we assume true positives and negatives to have cost 0
cost_mispredictions = [1,3] # default [1,3]
# we create a cost function out of it. It is a weighted sum of False Positives and False Negatives. The lower, the better
evaluation_function = cost_ev(cost_mispredictions[0],cost_mispredictions[1])

# We want to use our close enough for long enough baseline first. We train our model with the input_raw and input_output data

print("TRAINING BASELINE")
# max_time is the max range we believe the best long enough could be. This is usually VERY low (tipically 0) thus, we put it to 3,
# but if it were to turn out 3 to be the best, you might want to increase it to see if something goes well
max_time = 2
"""
    This baseline needs raw data as input and of course the output to train. It minimizes the cost function from the cost matrix cost_mispredictions.
    To speed up the process, it requires to know the number of bunnies interacting (to properly split input_raw and input_output)
    it bruteforcely searches over the space [0,receiver_range]X[0,max_time]. It is still very fast even if quadratic.
    INF_TOKEN must be passed to understand when a signal is not sensed.

    it returns the two optimal thresholds: best_max_dist,best_min_time

"""
bl_dist, bl_time = extract_best_configuration_baseline_distance_and_time(input_raw,input_output,cost_mispredictions,number_bunnies,receiver_range,max_time,INF_TOKEN)

# Once we have got the best model, we want to evaluate it with some test data.
print("TESTING BASELINE")
# we infer data from our test raw data. We give the number of bunnies, the two thresholds and the INF_TOKEN
bl_pred = predict_threshold_with_time(test_raw,number_bunnies,bl_dist,bl_time,INF_TOKEN)

# we create a confusion matrix and feed it to the evaluation function
bl_conf = compute_confusion_matrix(bl_pred,test_output)
bl_eval = evaluation_function(bl_conf[0][0],bl_conf[1][1],bl_conf[1][0],bl_conf[0][1])

# We print all the results
print("Distance used: %d, time considered: %d" % (bl_dist, bl_time))
print_result_info(bl_conf,bl_eval)


# Not bad, but we can do better. Let's try with an ensemble method!

# However, we want to use only a small portion of the input data.
# We decided to only use absolute distances and speed
input_target = []
for i in range(len(input_abs)):
    input_target.append(input_abs[i]+input_speed[i])

test_target = []
for i in range(len(test_abs)):
    test_target.append(test_abs[i]+test_speed[i])

# input_target and test_target are going to be the input data we use to train and test respectively
print("TRAINING ENSEMBLE METHOD")
"""
    extract_best_configuration_dt takes as input whatever input data we want and as output our output.
    it needs our cost matrix and it can work with different methods: bagging,random_forest,boosting

    its output is our classifier and its markov assumption (it finds the best markov assumption)
"""
ensemble_model, ensemble_markov = extract_best_configuration_dt(input_target,input_output,evaluation_cost=cost_mispredictions,method='bagging')

# Now we evaluate it
print("TESTING ENSEMBLE METHOD:")
actual_order_test_input = test_target
# if the actual ensemble_markov is greater than 0, it means we need to transform our input data to work with that assumption
if ensemble_markov > 0:
    actual_order_test_input = create_markov_data(test_target,ensemble_markov)

# we would need to pad in case of a higher than 0 order
padding = [0]*len(test_output[0])

# Then, we simply infer
ensemble_pred = predict_dts(actual_order_test_input[:],ensemble_model)
# for now, models of order greater than 0, remove test cases. we 0 pad them.
for _ in range(ensemble_markov):
    ensemble_pred.insert(0,padding[:])

ensemble_conf = compute_confusion_matrix(ensemble_pred,test_output)
ensemble_eval = evaluation_function(ensemble_conf[0][0],ensemble_conf[1][1],ensemble_conf[1][0],ensemble_conf[0][1])

# printing info
print("Markov order %d:" % ensemble_markov)
print_result_info(ensemble_conf,ensemble_eval)