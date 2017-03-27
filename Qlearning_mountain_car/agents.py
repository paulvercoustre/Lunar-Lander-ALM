import numpy as np
import itertools as iter
import time


class RandomAgent(object):
    """Implements an agent that takes random actions
    independently of the current state"""

    def __init__(self, state_space, action_space, seed=1):
        self.state_space = state_space    # dimensionality of the state space
        self.action_space = action_space  # dimensionality of the action space

        np.random.seed(seed)

    def act(self, new_observation, epsilon, testmode=False):

        return np.random.randint(0, self.action_space-1)

    def update(self, observation, action, reward, new_observation, done):

        pass


class QlearningAgent(object):
    """ Implements a QLearning agent which discretises the state space
     and builds a high dimensional representation of the state with a kernel"""

    def __init__(self, state_space, action_space, seed=1):
        self.state_space = state_space  # dimensionality of the state space
        self.action_space = action_space  # dimensionality of the action space

        np.random.seed(seed)

        self.alpha = 0.2
        self.gamma = 0.9
        # self.epsilon = 0.1  # exploration/exploitation ratio

        self.Q = np.zeros(self.action_space)  # the Q values
        self.next_Q = np.zeros(self.action_space)

        self.phi_dict = {}   # the representation vector as dict
        self.phi_array = []  # the representation vector
        self.next_phi_dict = {}
        self.next_phi_array = []

        self.action = 0      # action taken by the agent

        self.bins = 1  # granularity of discretisation of state space except legs
        self.bins_legs = 1

        self.s = {}
        for i, j, k, l, m, n, o, p in iter.product(range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins_legs + 1),
                                                   range(self.bins_legs + 1)):

            self.s[i, j, k, l, m, n, o, p] = np.array([-2 + i * 4 / self.bins,
                                                       -2 + j * 4 / self.bins,
                                                       -2 + k * 4 / self.bins,
                                                       -2 + l * 4 / self.bins,
                                                       -2 + m * 4 / self.bins,
                                                       -2 + n * 4 / self.bins,
                                                       -0.5 + o * 1 / self.bins_legs,
                                                       -0.5 + p * 1 / self.bins_legs])

        self.weight = np.ones((self.action_space, ((self.bins + 1) ** 6) * ((self.bins_legs + 1) ** 2))) / \
                      ((self.bins + 1) ** self.state_space)  # set equal weights initially

    def kernel(self, x_Vx, s_ij, bins, bins_legs):
        phi_value = {}
        for i, j, k, l, m, n, o, p in iter.product(range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins_legs + 1),
                                                   range(bins_legs + 1)):

            phi_value[i, j, k, l, m, n, o, p] = np.exp(-(x_Vx[0] - s_ij[i, j, k, l, m, n, o, p][0]) ** 2) * \
                                                np.exp(-(x_Vx[1] - s_ij[i, j, k, l, m, n, o, p][1]) ** 2) * \
                                                np.exp(-(x_Vx[2] - s_ij[i, j, k, l, m, n, o, p][2]) ** 2) * \
                                                np.exp(-(x_Vx[3] - s_ij[i, j, k, l, m, n, o, p][3]) ** 2) * \
                                                np.exp(-(x_Vx[4] - s_ij[i, j, k, l, m, n, o, p][4]) ** 2) * \
                                                np.exp(-(x_Vx[5] - s_ij[i, j, k, l, m, n, o, p][5]) ** 2) * \
                                                np.exp(-(x_Vx[6] - s_ij[i, j, k, l, m, n, o, p][6]) ** 2) * \
                                                np.exp(-(x_Vx[7] - s_ij[i, j, k, l, m, n, o, p][7]) ** 2)

        return phi_value

    def dict_to_array(self, dict_input, bins, bins_legs):  # converts dictionary to array to compute dot product
        iterator = 0
        phi_as_array = np.ones(((self.bins + 1) ** 6) * ((self.bins_legs + 1) ** 2))
        for i, j, k, l, m, n, o, p in iter.product(range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins_legs + 1),
                                                   range(bins_legs + 1)):

            phi_as_array[iterator] = dict_input[i, j, k, l, m, n, o, p]
            iterator += 1
        return phi_as_array

    def act(self, observation, epsilon, testmode=False):
        # t = time.clock()
        self.phi_dict = self.kernel(observation, self.s, self.bins, self.bins_legs)
        # print(time.clock() - t, "seconds process time for kernel")

        # t1 = time.clock()
        self.phi_array = self.dict_to_array(self.phi_dict, self.bins, self.bins_legs)
        # print(time.clock() - t1, "seconds process time for converter")

        rand = np.random.uniform()
        if rand < epsilon:
            self.action = np.random.randint(0, self.action_space-1)  # hardcoded value !
        else:
            # greedy case. Take action with the maximal Q-value
            self.Q = np.dot(self.weight, self.phi_array)
            self.action = np.argmax(self.Q)
        return self.action

    def update(self, observation, action, reward, new_observation, done):
        self.next_phi_dict = self.kernel(new_observation, self.s, self.bins, self.bins_legs)
        self.next_phi_array = self.dict_to_array(self.next_phi_dict, self.bins, self.bins_legs)
        self.next_Q = np.dot(self.weight, self.next_phi_array)

        self.weight[self.action, :] += self.alpha * (reward + self.gamma * max(self.next_Q) - self.Q[self.action]) * \
                                       self.phi_array