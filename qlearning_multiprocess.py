
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import multiprocessing as mp
import timeit

# TMAX = 2**18 #Optimal for poc <= 10
# My machine caps out at 2 processes before overhead dominates
TMAX = 2**13
PROC_MAX = 15
ASYNC_UPDATE = 5

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
def learning_worker(state, Q, env, e, y, T, q, id):
    #process specific Q updates
    del_Q = init_q()
    #process specific time steps taken
    t = 1
    #continue learning until TMAX is reached
    while T.value < TMAX:
        state = init_state()
        #learn until goal reached or TMAX reached
        while state != [0,8] and T.value < TMAX:
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
            #update del_Q
            del_Q[prev_state[0]][prev_state[1]][action] += (r + (y * Q[state[0]][state[1]][maximising_action(Q, state)]) - Q[prev_state[0]][prev_state[1]][action])
            if (T.value < TMAX and t % ASYNC_UPDATE == 0) or state == [0,8]:
                #push to manager process - del_Q, time steps taken, process id, and
                try:
                    q.put([t, del_Q, id, r], True, 1)
                except:
                    #if queue full,
                    print('killing process {}, TMAX reached'.format(id))
                    break
                #reset del_Q to 0
                del_Q = init_q()
            T.value += 1
            t+=1

    #allow processes to terminate without clearing queue - some updates may be lost
    q.cancel_join_thread()

def main():
    #check globals for sanity
    assert(TMAX >= 1)
    assert(PROC_MAX >= 1)
    assert(ASYNC_UPDATE >= 1)

    #create process manager
    mgr = mp.Manager()
    graph_ratios = []
    rand.seed(1) # seed with static value for testing purposes

    #Perform learning from scratch with 1 to 10 processes
    for proc in range(1, PROC_MAX+1):

        #timer for checking runtimes
        start_time = timeit.default_timer()
        #create process-shared Q, T
        Q_proxy = mgr.list(init_q())
        T = mgr.Value('i',1)
        #create queue for process communication
        q = mp.Queue(2*proc)
        #initialize environments 1 and 2
        env1, env2 = init_env()
        #list to track process stats: timesteps taken and total reward
        p_stats = [[0 for i in range(2)] for j in range(proc)]
        #learning parameters
        e = 0.005
        y = 0.95

        #start PROC_MAX learning worker processes
        jobs =  [mp.Process(target=learning_worker, args=(init_state(), Q_proxy, env1, e, y, T, q, i )) for i in range(proc)]
        for j in jobs:
            j.start()

        #Manager process collects process updates, propagates back to subprocesses
        ## via proxy
        while T.value < TMAX or not q.empty():
            try:
                #if timeout then workers are all done
                resp = q.get(True, 4)
            except:
                #stop trying to collect items, print stats
                print('broke at {} done',format(T.value / TMAX))
                break
            proc_t = resp[0]
            del_Q = list(resp[1])
            pid = resp[2]
            proc_r = resp[3]

            #handle Q update:
            #make copy of data in Qproxy
            Q_update = list(Q_proxy)
            #update this copy using subprocess deltas
            for i in range(len(Q_update)):
                for j in range(len(Q_update[0])):
                    for k in range(len(Q_update[0][0])):
                        Q_update[i][j][k] += del_Q[i][j][k]

            #replace top level objects in proxy list with the updated copy
            ## without doing this, changes don't propagate to subprocesses
            for i in range(len(Q_proxy)):
                Q_proxy[i] = Q_update[i]

            #update process-specific stats (total: time steps, reward)
            p_stats[pid][0] = proc_t
            p_stats[pid][1] += proc_r

        ##join processes when TMAX reached
        for j in jobs:
            j.join()

        #model statistics: average total episodic reward across all agents
        ## vs total timesteps across agents
        steps = []
        rewards = []
        for i in p_stats:
            steps.append(i[0])
            rewards.append(i[1])
        print('current number of procs: {}'.format(proc))
        print('avg steps taken by each agent: {}'.format(np.mean(steps)))
        print('avg reward across all agents: {}'.format(np.mean(rewards)))
        print('learning time ratio (lower = better): {}'.format(np.mean(steps) / np.mean(rewards)))
        print('time taken: {}'.format(timeit.default_timer() - start_time))
        #collect stats for graphing
        graph_ratios.append(np.mean(steps) / np.mean(rewards))
        proc+=1

    #Graph the statistics from env1
    plt.figure(1)
    plt.title('Environment 1 Learning with Multiple Processes')
    plt.plot(np.arange(1,PROC_MAX+1,1),graph_ratios, label='Learning Ratio (Lower is better)')
    plt.xticks(np.arange(1,PROC_MAX+1,1))
    plt.ylabel('Learning Ratio (Lower is better)')
    plt.xlabel('Number of processes')
    plt.show()



if __name__ == "__main__":
    main()
