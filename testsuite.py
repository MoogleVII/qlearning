import qlearning as ql
import qlearning_multiprocess as qlm
import unittest

class TestQlearning(unittest.TestCase):

# init_state - just check the right state is given
    def test_init_state(self):
        self.assertEqual([5,3],ql.init_state())

# init_env - check boundaries, walls, difference between environments
    def test_init_env(self):
        #check boundaries
        env1, env2 = ql.init_env()
        self.assertTrue(len(env1[0]) == len(env2[0]))
        self.assertTrue(len(env1) == len(env2))
        self.assertTrue(len(env1[0]) == 9)
        self.assertTrue(len(env1) == 6)
        #check wall endpoints
        self.assertTrue(env1[3][0] != env2[3][0])
        #check wall existence
        self.assertTrue(env1[3][0] == 1)

# is_valid_move
    def test_is_valid_move(self):
        #basic environment
        env = [[1, 1], [0, 0]]
        #start at bottom left
        state = [1,0]

        #boundary movement
        self.assertFalse(ql.is_valid_move(env, 0, state)) #left
        #obstacle movement
        self.assertFalse(ql.is_valid_move(env, 1, state)) #up
        #open space movement
        self.assertTrue(ql.is_valid_move(env, 2, state)) #right
        #down and boundary movement
        self.assertFalse(ql.is_valid_move(env, 3, state)) #down

# maximising_action
    def test_maximising_action(self):
        state = [0,0]
        Q = [[[0, 1, 2, 4]]]
        #basic case, unique max
        self.assertEqual(3, ql.maximising_action(Q, state))
        #complex case, multiple maxima
        Q = [[[0,1,2,2]]]
        maxindex = ql.maximising_action(Q, state)
        self.assertTrue(maxindex == 2 or maxindex == 3)

#update_State
    def test_update_state(self):
        state = [0,0]
        #test outof bounds move
        try:
            ql.update_state(state, 0)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        state = [5,5]
        #test legal left
        self.assertEqual(ql.update_state(state, 0), [5,4])
        # '' up
        self.assertEqual(ql.update_state(state, 1), [4,4])
        # '' right
        self.assertEqual(ql.update_state(state, 2), [4,5])
        # '' down
        self.assertEqual(ql.update_state(state, 3), [5,5])

#choose action - only test deterministic behavior
    def test_choose_action(self):
        e = 0
        Q = [[[0, 1, 2, 4]]]
        state = [0,0]

        #check effect of e and propagation of Q
        # should return maximising action, since e = 0
        self.assertEqual(ql.choose_action(e, Q, state), 3)

# is_non_obstacle
    def test_is_non_obstacle(self):
        #simple, 0 is non-obstacle, 1 is obstacle
        self.assertTrue(ql.is_non_obstacle(0))
        self.assertFalse(ql.is_non_obstacle(1))

# learning_episode
#nondeterministic behavior so only test trivial case
    def test_learning_episode(self):
        state = [0,8] #goal state
        Q = [[[0, 1, 2, 4]]]
        env = [[1, 1], [0, 0]]
        a = 1
        e = 0.05
        y = 0.95

        #check if trivial case returns input arguments, 0 timestep
        rQ, rT = ql.learning_episode(state, Q, env, e, a, y)
        self.assertEqual(Q, rQ)
        self.assertEqual(0, rT)

#mock worker process for testing communication
    def worker_process(self, q):
        q.put(1)

#test multiprocessing communication
    def test_multiprocessing_communication(self):
        mgr = qlm.mp.Manager()
        #mock nested managed list to imitate actual use
        q = mgr.Queue()
        array_proxy = mgr.list([[0]])
        job =  qlm.mp.Process(target=self.worker_process, args=(q,))
        job.start()
        self.assertEqual(q.get(), 1)
        job.join()




if __name__ == '__main__':
    unittest.main()
