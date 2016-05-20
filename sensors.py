import math
import random
import numpy as np
import copy

# defining the infinite value
inf = float('inf')
# defining the format for infinity in the files
INF_TOKEN = "INF"

#dist lambda function of two points
dist = lambda p1,p2: math.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

def magnitude(v):
    return math.sqrt(sum(map(lambda x: x**2,v)))

def normalize(v):
    mag = magnitude(v)
    if mag == 0: return (0,0)
    return tuple(list(map(lambda x: x/mag,v)))

def noise_position(pos,stdev):
    return tuple(map(lambda x: x + np.random.normal(0,stdev),pos))

def tup_sum(t1,t2):
    result = []
    for i in range(len(t1)):
        result.append(t1[i]+t2[i])
    return tuple(result)

def tup_diff(t1,t2):
    result = []
    for i in range(len(t1)):
        result.append(t1[i]-t2[i])
    return tuple(result)

def tup_prod(t1,scalar):
    result = []
    for i in range(len(t1)):
        result.append(t1[i]*scalar)
    return tuple(result)

class Sensor:
    """

    common class base for every sensor
    """

    def __init__(self, reliability,error):
        self.reliability = reliability
        self.error = error # standard deviation

    def noise_result(self,pos):
        if self.error > 0.:
            return tuple(map(lambda x: x + np.random.normal(0,self.error),pos))
        return pos


class Transmitter(Sensor):

    def transmits(self,pos):
        return self.noise_result(pos) if  random.random() <= self.reliability else INF_TOKEN


class Receiver(Sensor):

    def __init__(self,reliability,error,range,precision,distance_quatisation):
        super(Receiver,self).__init__(reliability,error)
        # range is for both abs and rel distances
        self.range = range
        # precision is for relative distances
        self.precision = precision
        # distance_quantisation is for abs distances
        self.distance_quantisation = distance_quatisation

    def receives(self,pos,transm):
        """Default implementation is solely based on crisp values: Too far <=> False

        :param pos: the position of the receiver
        :param transm: the transmitter position which is transmitting
        :return: False if the receiver does not receive that signal, else a noisy distance
        """

        if random.random() > self.reliability:
            return False

        act_dist = dist(pos,transm)

        if act_dist >= self.range:
            return False

        #distance = self.noise_result((act_dist,))[0]
        distance = act_dist + (np.random.normal(0,self.error*(act_dist/self.range)) if self.error > 0. else 0)
        while distance < 0 or distance >= self.range:
            # we create a noise which increases linearly the further the two sensors are
            distance = act_dist + np.random.normal(0,self.error*(act_dist/self.range))
        return distance

    def estimate_distances(self, holder, transmitters):
        """

        :param holder: the agent this receiver is attached to: we don't want to add useless self signals
        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of unprocessed distances for each transmitter BUT the ones of the same id
        """

        pos = holder.pos

        result = []
        for cp in transmitters:
            # We ignore the same ids
            if cp[0] == holder: continue
            t = cp[1]
            # infinite distances remain infinite distances
            if t == INF_TOKEN:
                result.append(t)
            else:
                dist = self.receives(pos,t)
                result.append(INF_TOKEN if dist == False else dist)

        return result


    def compute_rankings(self, distances):
        """

        :param distances: a list of distances. If a transmitter does not transmit, then its value is INF_TOKEN
        :return: a list of rankings
        """
        # infinite distances remain infinite distances
        #result = distances[:]
        result = []
        for t in distances:
            if t == INF_TOKEN:
                result.append(t)
            else:
                # we want a quantized distance
                result.append(int(t//self.precision) +1)

        # now we need to transform the rankings from 1 to whatever, without gaps
        existing_ranks = sorted(list(set(filter(lambda x: x != INF_TOKEN,result))))
        for i, rank in enumerate(existing_ranks):
            result = [i + 1 if val == rank else val for val in result]

        # finally we want to transform inf_tokens to "furthest", thus
        filtered_results = list(filter(lambda x: x != INF_TOKEN,result))
        if len(filtered_results) == 0:
            max_rank = 1
        else:
            max_rank = max(filtered_results) + 1
        result = [max_rank if x == INF_TOKEN else x for x in result]

        return result


    def quantise_distances(self, distances):
        """

        :param distances: a list of distances. If a transmitter does not transmit, then its value is INF_TOKEN
        :return: a list of quantised distances
        """

        result = distances[:]

        # We find out what is the sensitivity first
        gap = self.range / self.distance_quantisation

        # Value for too far away (inf) are going to be distance_quantisation
        # now we just divide the distances we have
        result = [self.distance_quantisation if x == INF_TOKEN else int(x//gap) for x in result]

        return result

    def compute_abs_and_rel_distances(self,holder,transmitters):
        """

        :param holder: the agent this receiver is attached to: we don't want to add useless self signals
        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and rel distances for each transmitter BUT the ones of the same id
        """
        distances = self.estimate_distances(holder,transmitters)
        result = self.quantise_distances(distances)
        result += self.compute_rankings(distances)

        return result

    def compute_abs_and_rel_distances_from_raw(self,distances):
        """

        :param distances: the raw distances
        :return: a list of abs and rel distances for each transmitter BUT the ones of the same id
        """
        result = self.quantise_distances(distances)
        result += self.compute_rankings(distances)

        return result

    def compute_abs_and_rel_distances_from_raw_split(self,distances):
        """

        :param distances: the raw distances
        :return: a list of abs and a list of rel distances for each transmitter BUT the ones of the same id
        """
        result_abs = self.quantise_distances(distances)
        result_rel = self.compute_rankings(distances)

        return result_abs,result_rel

# A pillar is a fixed position in the environment where we can have receivers. We don't support transmitters yet.
class Pillar:

    def __init__(self,pos,receivers):

        self.pos = pos
        self.receivers = receivers

    def compute_abs_and_rel_distances(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and rel distances for each transmitter BUT the ones of the same id of the agent
        """
        result = []
        for receiver in self.receivers:
            result += receiver.compute_abs_and_rel_distances(self,transmitters)

        return result

    def compute_abs_and_rel_distances_split(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and a list of rel distances for each transmitter BUT the ones of the same id of the agent
        """
        result_abs = []
        result_rel = []

        for receiver in self.receivers:
            act_raw = receiver.estimate_distances(self,transmitters)

            tmp = receiver.compute_abs_and_rel_distances_from_raw_split(act_raw)
            result_abs += tmp[0]
            result_rel += tmp[1]

        return result_abs,result_rel


    def compute_raw_distances(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of raw distances for each transmitter BUT the ones of the same id of the agent. It can have INF_TOKEN
        """
        result = []
        for receiver in self.receivers:
            result += receiver.estimate_distances(self,transmitters)

        return result

    def compute_abs_rel_raw_distances(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and rel distances for each transmitter BUT the ones of the same id of the agent
            and a list of raw distances
        """
        result = []
        result_raw = []
        for receiver in self.receivers:
            act_raw = receiver.estimate_distances(self,transmitters)
            result_raw += act_raw
            result += receiver.compute_abs_and_rel_distances_from_raw(act_raw)

        return result,result_raw


#default agent info
agent_info = {}

class Agent:

    def __init__(self,pos,transmitters,receivers,info = agent_info):

        self.pos = pos
        self.prev_pos = pos
        self.speed = 0.

        self.transmitters = transmitters
        self.receivers = receivers
        self.info = info
        self.state = 'NORMAL'
        self.state_info =  copy.deepcopy(self.info["normal_state_info"])

    def move(self,world):
        self.prev_pos = self.pos
        self.info["move"](self,world)
        self.speed = dist(self.pos,self.prev_pos)

    def compute_abs_and_rel_distances(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and rel distances for each transmitter BUT the ones of the same id of the agent
        """
        result = []
        for receiver in self.receivers:
            result += receiver.compute_abs_and_rel_distances(self,transmitters)

        return result

    def compute_raw_distances(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of raw distances for each transmitter BUT the ones of the same id of the agent. It can have INF_TOKEN
        """
        result = []
        for receiver in self.receivers:
            result += receiver.estimate_distances(self,transmitters)

        return result

    def compute_abs_rel_raw_distances(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and rel distances for each transmitter BUT the ones of the same id of the agent
            and a list of raw distances
        """
        result = []
        result_raw = []
        for receiver in self.receivers:
            act_raw = receiver.estimate_distances(self,transmitters)
            result_raw += act_raw
            result += receiver.compute_abs_and_rel_distances_from_raw(act_raw)

        return result,result_raw

    def compute_abs_rel_raw_distances_split(self,transmitters):
        """

        :param transmitters: list of couples (id,transmission). the transmission is INF_TOKEN or a point
        :return: a list of abs and a list of rel distances for each transmitter BUT the ones of the same id of the agent
            and a list of raw distances
        """
        result_abs = []
        result_rel = []
        result_raw = []
        for receiver in self.receivers:
            act_raw = receiver.estimate_distances(self,transmitters)
            result_raw += act_raw
            tmp = receiver.compute_abs_and_rel_distances_from_raw_split(act_raw)
            result_abs += tmp[0]
            result_rel += tmp[1]

        return result_abs,result_rel,result_raw

    def transmit(self):
        """

        :return: a list of tuples (self, transmit result) for every transmitter
        """
        return [(self,x.transmits(self.pos)) for x in self.transmitters]

#custom agent info

agent_info["size"] = 30
agent_info["move_chance"] = 0.3
agent_info["max_movement"] = 45

#custom agent movement:
def random_movement(self,world):

    pos = self.pos

    if random.random() <= self.info["move_chance"]:
        max_mov = self.info["max_movement"]
        if random.random() < 0.5:
            new_x = pos[0] + random.randrange(max_mov) + 1
        else:
            new_x = pos[0] - random.randrange(max_mov) - 1
        world_x = world["x_range"]
        if not(world_x[0] <= new_x <= world_x[1]):
            new_x = pos[0]

        if random.random() < 0.5:
            new_y = pos[1] + random.randrange(max_mov) + 1
        else:
            new_y = pos[1] - random.randrange(max_mov) - 1

        world_y = world["y_range"]
        if not (world_y[0] <= new_y <= world_y[1]):
            new_y = pos[1]

        self.pos = (new_x,new_y)


def adjust_to_world_bounds(pos,world):
    new_x,new_y = pos
    x_range = world['x_range']
    if new_x < x_range[0]: new_x = x_range[0]
    elif new_x > x_range[1]: new_x = x_range[1]
    y_range = world['y_range']
    if new_y < y_range[0]: new_y = y_range[0]
    elif new_y > y_range[1]: new_y = y_range[1]

    return new_x,new_y

def compute_position_from_direction(agent,dir,world):
    dir = normalize(dir)
    dist_range = agent.state_info["movement_range"]
    distance = random.randrange(dist_range[0],dist_range[1])
    dir = tup_prod(dir,distance)

    pos = tup_sum(agent.pos,dir)
    # we want to mess up a bit the position, thus we add noise
    pos = noise_position(pos,agent.state_info["movement_error"])
    # we finally adjust the position for the world bounds
    pos = adjust_to_world_bounds(pos,world)
    return pos


normal_state_info = {'transition_probabilities':(0.05,0.1,0.3),'movement_range': (0,1),'movement_error': 15,
                     'interaction_range': 45,'movement_chance': 0.2}
agent_info["normal_state_info"] = normal_state_info

sleeping_state_info = {'sleeping_turns_range': (3,21)}
agent_info["sleeping_state_info"] = sleeping_state_info

hungry_state_info = {'eating_range': 20, 'eating_turns_range': (1,5),'movement_range': (10,20),'movement_error': 2 }
agent_info["hungry_state_info"] = hungry_state_info

interacting_state_info = {'movement_range': (10,30),'movement_error': 5, 'dropout_chance': 0.4, 'interact_chance': 0.3,
                          'interaction_range': 30}
agent_info["interacting_state_info"] = interacting_state_info


def fsa_movement(self,world):
    """State driven movement

    :param self: a agent
    :param world: all the world info
    """
    act_state = self.state
    act_state_info = self.state_info

    if act_state == 'NORMAL':
        # Here we can switch to ANY other state, or stay normal.

        # this is a tuple of numbers between 0 and 1. (sleep_prob,hungry_prob,interact_prob)
        #example: (0.2,0.4,0.7) means we have 20% chance to go sleep, 20% to go hungry, 30% to go interact, 30% to stay normal.
        state_transitions = act_state_info["transition_probabilities"]
        ran_num = random.random()
        if ran_num < state_transitions[0]:
            # We sleep
            self.state = 'SLEEPING'
            self.state_info = copy.deepcopy(self.info["sleeping_state_info"])
            sleep_range = self.state_info['sleeping_turns_range']
            self.state_info['sleeping_turns'] = random.randrange(sleep_range[0],sleep_range[1])

        elif ran_num < state_transitions[1]:
            # We eat
            self.state = 'HUNGRY'
            self.state_info = copy.deepcopy(self.info["hungry_state_info"])
            eat_range = self.state_info['eating_turns_range']
            self.state_info['eating_turns'] = random.randrange(eat_range[0],eat_range[1])
            # Also, we call move again.
            fsa_movement(self,world)

        elif ran_num < state_transitions[2]:
            # We TRY to interact
            # We check ONE agent which is in our interaction range. if there isn't any, we do nothing.
            interaction_range = act_state_info['interaction_range']
            for agent in world['bunnies']:
                if agent == self: continue
                if dist(self.pos,agent.pos) < interaction_range:
                    # we found one!
                    self.state = 'INTERACTING'
                    self.state_info = copy.deepcopy(self.info["interacting_state_info"])
                    self.state_info['active_interactions'] = [agent]

                    # the first move is done here. without losing the interaction!
                    self.pos = compute_position_from_direction(self, tup_diff(agent.pos, self.pos),world)
                    break

        else:
            # we behave normally
            # we randomly move with a certain probability
            if random.random() < act_state_info['movement_chance']:
                #it randomly moves thanks to the error.
                self.pos = compute_position_from_direction(self,(0,0),world)

    elif act_state == 'SLEEPING':
        turns_to_sleep = act_state_info["sleeping_turns"]
        turns_to_sleep -= 1
        if turns_to_sleep <= 0:
            # we go back to normal
            self.state = 'NORMAL'
            self.state_info = copy.deepcopy(self.info["normal_state_info"])
        else:
            act_state_info["sleeping_turns"] = turns_to_sleep
    elif act_state == 'HUNGRY':
        # we want to go there
        eat_spot = world["food_spot"]
        # if we are in range we eat
        if dist(self.pos,eat_spot) < act_state_info["eating_range"]:
            turns_to_eat = act_state_info["eating_turns"]
            turns_to_eat -= 1
            if turns_to_eat <= 0:
                # we go back to normal
                self.state = 'NORMAL'
                self.state_info = copy.deepcopy(self.info["normal_state_info"])
            else:
                act_state_info["eating_turns"] = turns_to_eat
        else:
            # we move!
            new_dir = tup_diff(eat_spot, self.pos)
            self.pos = compute_position_from_direction(self,new_dir,world)
    elif act_state == 'INTERACTING':
        # We have a certain chance to drop an active interaction
        drop_chance = act_state_info["dropout_chance"]
        act_interactions = act_state_info["active_interactions"]
        # removes from the list with drop_chance chance any agent
        act_interactions[:] = [tup for tup in act_interactions if random.random() < drop_chance]

        # We have a certain chance to add new interactions with the nearby bunnies!
        add_chance = act_state_info["interact_chance"]
        # This can add an interaction just removed. Whatever!
        for agent in set(world["bunnies"]) - set(act_interactions) - set([self]):
            if random.random() < add_chance and dist(self.pos,agent.pos) < act_state_info['interaction_range']:
                act_interactions.append(agent)

        # If we still have at least one interaction, we pick one random agent among them and move somewhere around there!
        if len(act_interactions) > 0:
            target = random.choice(act_interactions)
            self.pos = compute_position_from_direction(self,tup_diff(target.pos, self.pos),world)
        else:
            # Else we go back to NORMAL
            self.state = 'NORMAL'
            self.state_info = copy.deepcopy(self.info["normal_state_info"])


agent_info["move"] = fsa_movement

