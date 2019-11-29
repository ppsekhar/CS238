import sys
import numpy as np
import copy

import argparse

import pdb
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--outfilename', metavar='N', type=str,help='where to write policy')
args = parser.parse_args()

# 1 unit of $ = 1 unit of E
MIN_UNIT = -1
MAX_UNIT = 1

POSSIBLE_ACTIONS = [0, 1, 2, 3]

TRADE_ORDER = {
    0: 1,
    1: 3,
    3: 2,
    2: 0
}

# TODO: Enable trading with more than one partner? 
# TRADE_ORDER = {
#     0: [1,2],
#     1: [3,0],
#     3: [2,1],
#     2: [0,3]
# }

DISCOUNT_FACTOR = 0.5

GRID_LOCATION = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1)
}

CONSUMER_CONSUMPTION = 1
PRODUCER_PRODUCTION = 1


# B = 0 in deficit
# otherwise A and B both 1
# TODO: consider end states (out of bounds)
def reward(state):
    total_reward = 0
    for r in state:
        for microgrid in r:
            energy = microgrid[2]
            money = microgrid[3]
            a = 1
            b = 1
            if energy < 0:
                b = 0
            total_reward += a * energy + b * money
    return total_reward


def calculate_next_state(curr_state, action):
    buyer, seller, amount = action
    next_state = copy.deepcopy(curr_state)

    # update buyer
    buyer_loc = GRID_LOCATION[buyer]
    microgrid1 = curr_state[buyer_loc[0]][buyer_loc[1]]

    # Assume consumers consume 1 energy, producers produce 1
    new_energy1 = microgrid1[2] + amount - microgrid1[1]*CONSUMER_CONSUMPTION \
                    + microgrid1[0]*PRODUCER_PRODUCTION
    microgrid1_next = (microgrid1[0], microgrid1[1], new_energy1, microgrid1[3] - amount)
    next_state[buyer_loc[0]][buyer_loc[1]] = microgrid1_next

    # update seller
    seller_loc = GRID_LOCATION[seller]
    microgrid2 = curr_state[seller_loc[0]][seller_loc[1]]

    new_energy2 =  microgrid2[2] - amount - microgrid2[1]*CONSUMER_CONSUMPTION \
                    + microgrid2[0]*PRODUCER_PRODUCTION
    microgrid2_next = (microgrid2[0], microgrid2[1], new_energy2, microgrid2[3] + amount)
    next_state[seller_loc[0]][seller_loc[1]] = microgrid2_next

    return next_state


def generate_states(u_vals, initial_state):
    valid_energy_money_values = [i for i in range(MIN_UNIT, MAX_UNIT + 1)]
    energy_money_combos = list(product(valid_energy_money_values, valid_energy_money_values))

    # Assume initial state is 2 X 2
    mg_combos = []
    for row in initial_state:
        for mg in row:
            print(mg)
            possible_mg_values = []
            for combo in energy_money_combos:
                new_tuple = (mg[0], mg[1], combo[0], combo[1])
                possible_mg_values.append(new_tuple)

            mg_combos.append(possible_mg_values)

    state_space = list(product(mg_combos[0], mg_combos[1], mg_combos[2], mg_combos[3]))
    print(state_space[7])
    return state_space



def value_iteration(initial_state):

    # state: (action, utility)
    # action: (buyer, seller, $ amount)
    state = copy.deepcopy(initial_state)
    u_vals = {}
    num_states_visited = 0
    viable_next_states = []
    for trader in TRADE_ORDER:
        for action_amount in POSSIBLE_ACTIONS:
            # Sell 
            curr_action1 = (trader, TRADE_ORDER[trader], action_amount)
            next_state1 = calculate_next_state(state, curr_action1)
            viable_next_states.append(next_state1)

            # Buy 
            curr_action2 = (TRADE_ORDER[trader], trader, action_amount)
            next_state2 = calculate_next_state(state, curr_action2)
            viable_next_states.append(next_state2)
    
    # state_space = generate_states(u_vals, initial_state)
    while num_states_visited < 10000 and len(viable_next_states) > 0: # TODO: end at convergence
        num_states_visited+=1
        state = viable_next_states.pop(0)
        # TODO: loop through all states
        '''
        For Gauss-Seidel iteration, we need to pick a state ordering
        here and loop through all states, updating asynchronously as we go.
        We have 2 options for this:
            1. Use the MIN_UNIT and MAX_UNIT bounds to come up with all possible
            combinations of microgrid states. This will make the energy and money
            initializations of the microgrids meaningless.
            2. Only consider states that could be reachable from the initial state
            (ex. same total money as initialized, etc.). This makes the microgrid
            initializations matter, but is probably more difficult to do.
        '''
        state_key = flatten_microgrid(state)
        if state_key not in u_vals:
            # default:
                # utility: 0
                # action: top players don't trade
            u_vals[state_key] = ((0, 1, 0), 0)

        best_action = (0, 1, 0)
        best_u = 0
        first_action = True
        for trader in TRADE_ORDER:
            for action_amount in POSSIBLE_ACTIONS:

                # Sell 
                curr_action1 = (trader, TRADE_ORDER[trader], action_amount)
                next_state1 = calculate_next_state(state, curr_action1)
                viable_next_states.append(next_state1)
                next_state1_key = flatten_microgrid(next_state1)
                
                if next_state1_key not in u_vals:
                    u_vals[next_state1_key] = ((0, 1, 0), 0)
                u1 = reward(next_state1) + DISCOUNT_FACTOR*u_vals[next_state1_key][1]
                
                # Buy 
                curr_action2 = (TRADE_ORDER[trader], trader, action_amount)
                next_state2 = calculate_next_state(state, curr_action2)
                viable_next_states.append(next_state2)
                next_state2_key = flatten_microgrid(next_state2)
                
                if next_state2_key not in u_vals:
                    u_vals[next_state2_key] = ((0, 1, 0), 0)
                u2 = reward(next_state2) + DISCOUNT_FACTOR * u_vals[next_state2_key][1]

                if first_action or u1 > best_u:
                    best_action = curr_action1
                    best_u = u1
                    first_action = False
                if u2 > best_u:
                    best_action = curr_action2
                    best_u = u2
        u_vals[state_key] = (best_action, best_u)
    
    return u_vals





# TODO: deal with states not visited during value iteration
def extract_policy(u_vals, output_filename):
    pass

# Flatten microgrid into single tuple for hashing
def flatten_microgrid(state): 
    hashable_tuple = ()
    for loc in state: 
        hashable_tuple = hashable_tuple + tuple(loc)
    return hashable_tuple

# TODO: Generalize to worlds larger than 2x2
def main():
    # Each microgrid has (P, C, E, $)
    # money and energy have to be MIN_UNIT <= E, $ <= MAX_UNIT
    # Grid ids assigned left to right, top to bottom
    # TODO: read initial state from file
    initial_state = [
        [(2, 2, 1, 2), (2, 2, -1, 0)],
        [(2, 2, 1, -3), (2, 2, 1, 0)]
    ]
    flatten_microgrid(initial_state)

    # TODO: retrieve outfilename from args
    # print(reward(initial_state))
    u_vals = value_iteration(initial_state)
    for state in u_vals:
        if u_vals[state][0] != (0, 1, 0):
            print(state, u_vals[state])

    extract_policy(u_vals, args.outfilename)


if __name__ == '__main__':
    main()