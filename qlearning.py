
import numpy as np
import random as rand


#Sate: set of coordinates
#Initial state is [5,3]
def init_state():
    return [5,3]

#2-d array mapped to the environments
#each cell is a tile on the map, containing current best action-value (initially 0)
def init_q():
    Q = [[[0 for k in range(4)] for j in range(9)] for i in range(6)]
    return Q

#Environments are represented by 2-d arrays, cell containing 0 is valid,
# cell containing 1 is an obstacle
def init_env():
    env1 = np.zeros([6,9])
    env2 = np.array(env1)
    for j in range(8):
        env1[3][j] = 1
    for j in range(1,9):
        env2[3][j] = 1
    return env1, env2

#check all 4 possible actions, return action which corresponds to
# maximum Q
#PARAMETERS
## (Q) - action-value pairs, current (state)
def maximising_action(Q, state):
    #create list of all 4 possible actions to be taken
    possible_actions = []
    for i in range(4):
        possible_actions.append(Q[state[0]][state[1]][i])
    #return the index of the maximal action
    #Action mapping - [left, up, right, down] == [0, 1, 2, 3]
    return possible_actions.index(max(possible_actions))

#return action to be taken based on probablility e
#Action mapping - [left, up, right, down] == [0, 1, 2, 3]
def choose_action(e, Q, state):
    #based on e, choose either random action (0) or maximising action (1)
    choice = np.random.choice([0,1], 1, [e, 1-e])
    if choice == 0:
        #return random action
        return rand.randint(0,3)
    elif choice == 1:
        #return arg_max Q(s, a)
        return maximising_action(Q, state)

#Check if the selected movement is valid based on current state and environment
##PARAMETERS
#current state, proposed movement, current environment
def is_valid_move(env, action, state):
    #TODO error handling

    #Get environment bonudaries
    numrows = len(env)
    numcols = len(env[0])
    #Check boundaries of environment, if valid, check if new cell is wall
    if action == 0: #left
        if state[1] -1 < 0:
            return False
        return is_non_obstacle(env[state[0]][state[1]-1])
    elif action == 1: #up
        if state[0] - 1 < 0:
            return False
        return is_non_obstacle(env[state[0]-1][state[1]])
    elif action == 2: #right
        if state[1] + 1 >= numcols:
            return False
        return is_non_obstacle(env[state[0]][state[1]+1])
    elif action == 3: #down
        if state[0] + 1 >= numrows:
            return False
        return is_non_obstacle(env[state[0]+0][state[1]])

#updates current state based on the decided action
def update_state(state, action):
    #TODO error handling
        if action == 0: #left
            state[1] = state[1] - 1
        elif action == 1: #up
            state[0] = state[0] - 1
        elif action == 2: #right
            state[1] = state[1] + 1
        elif action == 3: #down
            state[0] = state[0] + 1
        return state

#Check if cell is a wall or not - true is wall, false if non-wall
def is_non_obstacle(cell):
    if cell == 0:
        return True
    elif cell == 1:
        return False
    #TODO error handling


#Take current state, make a movement on the current environment,
## update Q and state accordingly with time step. Recursivly calls self until
## goal location is reached
#PARAMETERS
## current (state), (Q) action-value function, current environment (env),
## current time step (T), probablity coefficient epsiolon (e), learning rate (a)
## discount factor (y)
def learning_episode(state, Q, env, T, e, a, y):
    if state == [0,8]:
        #Base case - goal rached, begin new episode
        #TODO finished current episode - return Q, T
        return Q, T
    else:
    #recursion case - continue updating
        #choose action to take
        action = choose_action(e, Q, state)
        #check if the action is valid
        valid = is_valid_move(env, action, state)
        #update the state based on action
        print(state)
        print(valid)
        print(action)
        prev_state = list(state)
        if valid:
            state = update_state(state, action)
        if state == [0,8]:
            r = 1
        else:
            r = 0
        #update Q
        Q[prev_state[0]][prev_state[1]][action] += a * (r + y * Q[state[0]][state[1]][maximising_action(Q, state)] - Q[state[0]][state[1]][action])
        return learning_episode(state, Q, env, T+1, e, a, y) #TODO recursion no work

def main():
    #initialize environments 1 and 2
    rand.seed()
    env1, env2 = init_env()
    #initialize Q array, state
    Q = init_q()
    #array to be graphed - step vs T taken to reach solution
    T_at_step = []
    #learn on environment 1, 1000 steps
    state = init_state() #move to loop
    #learning parameters
    a = 0.1
    e = 0.1
    y = 0.95
    Q, T = learning_episode(state, Q, env1, 0, e, a, y)
    #learn on environment 2, ?? steps
    #graph number of time steps taken to reach goal


if __name__ == "__main__":
    main()
