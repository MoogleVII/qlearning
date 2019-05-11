## simple Q-learning project ##

### Files: ###
1. qlearning.py
..* This file can be run to see the visualization of Assignment 1/algorithm 1, the simple single-process qlearning on environment 1 and 2
2. qlearning_multiprocess.py
..* This is the multiprocessing version of qlearning using algorithm 2. When run, it will print visualization of learning performance when using between 1-15 processes.
3. testsuite.py
..* This is a simple test suite for both main scripts

### Usage ###

Simply run `python <script>` to see the output to console and graph for each of the main scripts.

# NOTES: #
..* Because of the e-greedy policy, there is some stochastic varience with the multiprocessing graph. Depending on machine and random generation, infinite learning ratios may happen. In this case, reduce or change epsilon for better results.


### Images ###
