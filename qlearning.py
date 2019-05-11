
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

#check all 4 possible actions taken from (state), return action which corresponds to
# maximum Q(state)
#PARAMETERS
## (Q) - action-value pairs, current (state)
def maximising_action(Q, state):
    #create list of all 4 possible actions to be taken
    possible_actions = list(Q[state[0]][state[1]])

    #find maximising action in this state
    ## check if multiple maxima exist
    maximum = max(possible_actions)
    maxima = []
    i = 0
    while i < len(possible_actions):
        if possible_actions[i] == maximum:
            maxima.append(i)
        i+=1
    ## choose one of the maxima at random
    #Action mapping - [left, up, right, down] == [0, 1, 2, 3]
    return np.random.choice(maxima)

#choose action to be taken based on probablility (e)
#Action mapping - [left, up, right, down] == [0, 1, 2, 3]
def choose_action(e, Q, state):
    #based on e, choose either random action (0) or maximising action (1)
    choice = np.random.choice([0,1], 1, p=[e, 1-e])
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
    #error handling
    assert(action <= 3 and action >= 0)
    assert(len(env) > 0)
    assert(len(state) == 2)
    assert(state[0] >= 0 and state[1] >=0)

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
    #error handling
    #avoid state corruption with negative coordinates
    assert(not(state[1] == 0 and action == 0))
    assert(not(state[0] == 0 and action == 1))

    #update state based on action
    if action == 0: #left
        state[1] = state[1] - 1
    elif action == 1: #up
        state[0] = state[0] - 1
    elif action == 2: #right
        state[1] = state[1] + 1
    elif action == 3: #down
        state[0] = state[0] + 1
    return state

#Check if cell is a wall or not - false if obstacle, true if non-obstacle
def is_non_obstacle(cell):
    #input checking
    assert(cell == 0 or cell == 1)

    if cell == 0:
        return True
    elif cell == 1:
        return False


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
        prev_state = list(state)
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
    #initialize environments 1 and 2, Q array, state
    rand.seed()
    env1, env2 = init_env()
    Q = init_q()
    #array to be graphed - episode vs T taken to reach solution
    T_at_step = []
    #learn on environment 1, 1000 steps
    #learning parameters
    a = 1
    e = 0.005
    y = 0.95

    for i in range(1001):
        state = init_state() #move to loop
        Q, T = learning_episode(state, Q, env1, e, a, y)
        T_at_step.append(T)
#note: optimal solution is 10 steps
    print('Average steps to solve env 1, final 100 episodes: '),
    # print(np.average(T_at_step))
    print(np.mean(T_at_step[-100:]))
    plt.figure(1)
    plt.plot(T_at_step, label='actual')
    plt.plot([10 for i in range(len(T_at_step))], label='optimal')
    plt.title('Environment 1 Learning')
    plt.ylabel('# steps to Solve')
    plt.xlabel('Episode')
    plt.ylim(0,60)
    plt.legend()


    #learn on environment 2, 1000 steps
    T_at_step = []
    for i in range(1001):
        state = init_state() #move to loop
        Q, T = learning_episode(state, Q, env2, e, a, y)
        T_at_step.append(T)
#note: optimal solution is 16 steps
    print('Average steps to solve env 2, final 100 episodes: '),
    print(np.mean(T_at_step[-100:]))
    plt.figure(2)
    plt.plot(T_at_step, label='actual')
    plt.plot([16 for i in range(len(T_at_step))], label='opimal')
    plt.title('Environment 2 Learning')
    plt.ylabel('# steps to Solve')
    plt.xlabel('Episode')
    plt.legend()
    plt.ylim(0,60)

    plt.show()

if __name__ == "__main__":
    main()
