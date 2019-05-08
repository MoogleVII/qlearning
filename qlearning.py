
import numpy as np
import random as rand
import matplotlib.pyplot as plt

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
    if max(possible_actions) == 0:
        return rand.randint(0,3)
    return possible_actions.index(max(possible_actions))

#return action to be taken based on probablility e
#Action mapping - [left, up, right, down] == [0, 1, 2, 3]
def choose_action(e, Q, state):
    #based on e, choose either random action (0) or maximising action (1)
    choice = np.random.choice([0,1], 1, [e, 1-e])
    if choice == 0:
        #return random action
        # print('rand')
        return rand.randint(0,3)
    elif choice == 1:
        #return arg_max Q(s, a)
        # print('max')
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
## probablity coefficient epsiolon (e), learning rate (a)
## discount factor (y)
def learning_episode(state, Q, env, e, a, y):

    T = 0
    while state != [0,8]:
        #choose action to take
        action = choose_action(e, Q, state)
        #check if the action is valid
        valid = is_valid_move(env, action, state)
        #update the state based on action
        # print(state)
        # print(valid)
        # print(action)
        # print(Q)
        prev_state = list(state)
        # print(prev_state)
        if valid:
            state = update_state(state, action)
        if state == [0,8]:
            r = 1
        else:
            r = 0
        #update Q
        Q[prev_state[0]][prev_state[1]][action] += a * (r + (y * Q[state[0]][state[1]][maximising_action(Q, state)]) - Q[prev_state[0]][prev_state[1]][action])

        T+=1
    return Q, T

def main():
    #initialize environments 1 and 2
    rand.seed()
    env1, env2 = init_env()
    #initialize Q array, state
    Q = init_q()
    #array to be graphed - step vs T taken to reach solution
    T_at_step = []
    #learn on environment 1, 1000 steps
    #learning parameters
    a = 1
    e = 0.3
    y = 0.95

    for i in range(1001):
        state = init_state() #move to loop
        Q, T = learning_episode(state, Q, env1, e, a, y)
        T_at_step.append(T)
#note: optimal solution is 10 steps
    print('Average steps to solve env 1: '),
    print(np.average(T_at_step))
    plt.figure(1)
    plt.plot(T_at_step)
    plt.title('Environment 1 Learning')
    plt.ylabel('# steps to Solve')
    plt.xlabel('Iteration')


    #learn on environment 2, ?? steps
    T_at_step = []
    for i in range(1001):
        state = init_state() #move to loop
        Q, T = learning_episode(state, Q, env2, e, a, y)
        T_at_step.append(T)
#note: optimal solution is 16 steps
    print('Average steps to solve env 2: '),
    print(np.average(T_at_step))
    plt.figure(2)
    plt.plot(T_at_step)
    plt.title('Environment 2 Learning')
    plt.ylabel('# steps to Solve')
    plt.xlabel('Iteration')
    plt.show()
    #graph number of time steps taken to reach goal

if __name__ == "__main__":
    main()
