import sys
import numpy as np
import copy


# 1 unit of $ = 1 unit of E
MIN_UNIT = -10
MAX_UNIT = 10

POSSIBLE_ACTIONS: [0, 1, 2, 3]

TRADE_ORDER = {
    0: 1,
    1: 3,
    3: 2,
    2: 0
}

DISCOUNT_FACTOR = 0.5

GRID_LOCATION = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1)
}

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
    microgrid1 = curr_state[buyer_loc[0], buyer_loc[1]]
    # TODO: update energy and moey based on num consumers and producers
    microgrid1_next = (microgrid1[0], microgrid1[1], microgrid1[2] + amount, microgrid1[3] - amount)
    next_state[buyer_loc[0], buyer_loc[1]] = microgrid1_next

    # update seller
    seller_loc = GRID_LOCATION[buyer]
    microgrid2 = curr_state[seller_loc[0], seller_loc[1]]
    # TODO: update energy and moey based on num consumers and producers
    microgrid2_next = (microgrid2[0], microgrid2[1], microgrid2[2] - amount, microgrid2[3] + amount)
    next_state[seller_loc[0], seller_loc[1]] = microgrid2_next

    return next_state




def value_iteration(state):
    # state: (action, utility)
    # action: (buyer, seller, $ amount)
    u_vals = {}
    for i in range(3): # TODO: end at convergence
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
        if state not in u_vals:
            # default:
                # utility: 0
                # action: top players don't trade
            u_vals[state] = ((0, 1, 0), 0)
        best_action = (0, 1, 0)
        best_u = 0
        first_action = True
        for trader in TRADE_ORDER:
            for action_amount in POSSIBLE_ACTIONS:
                curr_action1 = (trader, TRADE_ORDER[trader], action_amount)
                next_state1 = calculate_next_state(state, curr_action1)
                if next_state1 not in u_vals:
                    u_vals[next_state1] = ((0, 1, 0), 0)
                u1 = reward(next_state1) + DISCOUNT_FACTOR*u_vals[next_state1]
                curr_action2 = (TRADE_ORDER[trader], trader, action_amount)
                next_state2 = calculate_next_state(state, curr_action2)
                if next_state2 not in u_vals:
                    u_vals[next_state2] = ((0, 1, 0), 0)
                u2 = reward(next_state2) + DISCOUNT_FACTOR * u_vals[next_state2]

                if first_action or u1 > best_u:
                    best_action = curr_action1
                    best_u = u1
                    first_action = False
                if u2 > best_u:
                    best_action = curr_action2
                    best_u = u2
        u_vals[state] = (best_action, best_u)





# TODO: deal with states not visited during value iteration
def extract_policy(u_vals):
    pass


# TODO: Generalize to worlds larger than 2x2
def main():
    # Each microgrid has (P, C, E, $)
    # money and energy have to be MIN_UNIT <= E, $ <= MAX_UNIT
    # Grid ids assigned left to right, top to bottom
    # TODO: read initial state from file
    initial_state = [
        [(2, 2, 10, 10), (2, 2, 10, 10)],
        [(2, 2, 10, 10), (2, 2, 10, 10)]
    ]
    # TODO: retrieve outfilename from args
    # print(reward(initial_state))
    u_vals = value_iteration(initial_state)
    extract_policy(initial_state, outfilename)


if __name__ == '__main__':
    main()