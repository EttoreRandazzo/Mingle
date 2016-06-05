
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
transmitter_reliability = 1. # default 0.95
transmitter_error = 0 #standad deviation 10
receiver_reliability = 1. # default 0.95
receiver_error = 0 # standard deviation 15
receiver_range = 100 # up to where it can sense
receiver_precision = 20 # when he considers ranks equal
distance_quanta = 5 # absolute distance values (0,1,2,3,4,...,distance_quanta,distance_quanta+1 (INF))

# set it to true if we want pillars
world_with_pillars = True
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
use_svm = True
use_dt_bag = True
use_dt_boost = True
use_dt_random = True
use_nn = True
use_bs = True
# MODEL EVALUATION
cost_mispredictions = [1,3] # default [1,3]
input_size = 500
test_size = 2000
number_repetitions = 10 # how many times we have to run this model to estimate an accurate average.


# END PARAMETERS SELECTION


input_abs,input_rel,input_speed,input_raw,output = create_multiple_timestamp_raw_input_and_output_all_split(input_size,world=custom_world)

individual_dictionary = transform_input_to_individual_based(input_abs,input_rel,input_raw,input_speed)

#print(individual_dictionary)

standardized_dictionary = standardize_individual_based_dictionary(individual_dictionary)

#print(standardized_dictionary)

extract_best_configuration_dt_independence(standardized_dictionary,output,[1,3])