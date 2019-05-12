## simple Q-learning project ##

### Files: ###
1. qlearning.py
  * This file can be run to see the visualization of Assignment 1/algorithm 1, the simple single-process qlearning on environment 1 and 2
2. qlearning_multiprocess.py
  * This is the multiprocessing version of qlearning using algorithm 2. When run, it will print visualization of learning performance when using between 1-15 processes.
3. testsuite.py
  * This is a simple test suite for both main scripts

### Installation ###
  * This project uses mainly internal python 2.7 modules with the exception of numpy, scipy and matplotlib
  * External packages can be installed using: 'python -m pip install --user numpy scipy matplotlib'
### Usage ###

Simply run `python <script>` to see the output to console and graph for each of the main scripts.

# NOTES: #
  * Because of the e-greedy policy, there is some stochastic varience with the multiprocessing graph. Depending on machine and random generation, infinite learning ratios may happen. In this case, reduce or change epsilon for better results.
  * There are several improvements to be made based on experimentation such as
   * Linearly or exponentially decreasing value of epsiolon as processes run, making randomness factor less as learning progresses
   * Changing the number of TMAX for multiprocessing, maybe make it based on PROC_MAX
   * Playing with the size of the queue for multiprocessing communication
   * Performing multiple simulations for averaging of results
   * multiprocessing version doesn't tranfer learning to env2 as it isn't required in the assignment, but that could be added easily


### Images ###

  * Fig 1, single process, environment 1 learning
  ![fig 1](https://github.com/MoogleVII/qlearning/blob/master/Figure_2.png)
  * Fig 2, single process, environment 2 transfer learning
  ![fig 2](https://github.com/MoogleVII/qlearning/blob/master/Figure_3.png)
  * Fig 3, multiple processes, environment 1 learning
  ![fig 3](https://github.com/MoogleVII/qlearning/blob/master/Figure_1.png)
