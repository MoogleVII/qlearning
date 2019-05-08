import qlearning as ql
import unittest

class TestQlearning(unittest.TestCase):
# init_state - just check the right state is given
    def test_init_state(self):
        self.assertEqual([5,3],ql.init_state())

# init_env - check boundaries, walls, ifference between environments
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
## TODO:
# maximising_action
#     dummy q
# choose action
# is_valid_move
# update_State
# is_non_obstacle
# learning_episode

if __name__ == '__main__':
    unittest.main()
