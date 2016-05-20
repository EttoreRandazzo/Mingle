from synthetic_world import *
from sensors import *


# CODE TO CREATE THE WORLD
custom_world = {}
# world boundaries
custom_world["x_range"] = (0,499) #inclusive
custom_world["y_range"] = (0,499)
# where is the food
custom_world["food_spot"] = (0,400) # needed for my Agent FSM

# sensors parameters
transmitter_reliability = 0.95
transmitter_error = 10 # standard deviation
receiver_reliability = 0.95
receiver_error = 15 # standard deviation
receiver_range = 100 # up to where it can sense
receiver_precision = 20 # when he considers ranks equal
distance_quanta = 5 # absolute distance values (0,1,2,3,4,...,distance_quanta,distance_quanta+1 (INF))

# set it to true if we want beacons (FIXED POSITION RECEIVERS)
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

# BY DEFAULT A BUNNY CAN HOLD A STATE AND ITS STATE INFO. GIVEN THAT, HERE IS MY FSM SETTINGS
# here I create a state info (for example for 'NORMAL')
custom_normal_state_info = {'transition_probabilities':(0.05,0.1,0.3),'movement_range': (0,1),'movement_error': 15,
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
number_agents = 6

for _ in range(number_agents):
    transm = Transmitter(transmitter_reliability,transmitter_error)
    receiv = Receiver(receiver_reliability,receiver_error,receiver_range,receiver_precision,distance_quanta)
    b_pos = (random.randrange(custom_world["x_range"][1] + 1),random.randrange(custom_world["y_range"][1] + 1))
    # a agent can also have custom behavior. IN THAT CASE, CREATE A PRIVATE BUNNY INFO HERE
    add_agent(Agent(b_pos,[transm],[receiv],custom_agent_info),custom_world)


# size of the data you want to synthesize
input_size = 2000
test_size = 2000


"""LEGEND: CREATES A input_size NUMBER OF TIMESTAMPS. the data is split (so you have to merge it if you wish, or look at another function)
input_abs: absolute pairwise distances
input_rel: ranking pairwise distances
input_speed: absolute speed of a agent
input_raw: unprocessed pairwise distances (has inf values)
output: labels written in MATRIX REPRESENTATION!

I thought I had written the function to move from matrix to list representation, but apparently I was wrong (or I deleted it)
To do that you should simply know how they are sorted and invert it:
Bunnies are numbered from 0 to n-1 (they are in a list in custom_world)
every timestamp is a matrix with these undirected interactions: [(0,1),(0,2),(0,3),...(0,n-1),(1,2),(1,3),...(1,n-1),....(n-2,n-1)]
there is of course a simple pattern to invert it. If you want me to do that, just tell me :)
"""
input_abs,input_rel,input_speed,input_raw,output = create_multiple_timestamp_raw_input_and_output_all_split(input_size,world=custom_world)
test_abs,test_rel,test_speed,test_raw,test_output = create_multiple_timestamp_raw_input_and_output_all_split(test_size,world=custom_world)

print(output)