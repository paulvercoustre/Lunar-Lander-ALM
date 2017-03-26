import numpy as np
import itertools as iter


class RandomAgent(object):
    """Implements an agent that takes random actions
    independently of the current state"""

    def __init__(self, state_space, action_space, seed=1):
        self.state_space = state_space    # dimensionality of the state space
        self.action_space = action_space  # dimensionality of the action space

        np.random.seed(seed)

    def act(self, new_observation, testmode=False):

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

        self.alpha = 0.1
        self.gamma = 0.1
        self.epsilon = 0.1  # exploration/exploitation ratio

        self.Q = np.zeros(self.action_space)  # the Q values
        self.next_Q = np.zeros(self.action_space)

        self.phi_dict = {}   # the representation vector as dict
        self.phi_array = []  # the representation vector
        self.next_phi_dict = {}
        self.next_phi_array = []

        self.action = 0      # action taken by the agent

        self.bins = 2  # granularity of discretisation of state space

        self.s = {}
        for i, j, k, l, m, n, o, p in iter.product(range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1),
                                                   range(self.bins + 1)):

            self.s[i, j, k, l, m, n, o, p] = np.array([-4 + i * 8 / self.bins,
                                                       -4 + j * 8 / self.bins,
                                                       -4 + k * 8 / self.bins,
                                                       -4 + l * 8 / self.bins,
                                                       -4 + m * 8 / self.bins,
                                                       -4 + n * 8 / self.bins,
                                                       -4 + o * 8 / self.bins,
                                                       -4 + p * 8 / self.bins])

        self.weight = np.ones((self.action_space, (self.bins + 1) **
                               self.state_space)) / ((self.bins + 1) ** self.state_space)  # set equal weights initially

    def kernel(self, x_Vx, s_ij, bins):
        phi_value = {}
        for i, j, k, l, m, n, o, p in iter.product(range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1)):

            phi_value[i, j, k, l, m, n, o, p] = np.exp(-(x_Vx[0] - s_ij[i, j, k, l, m, n, o, p][0]) ** 2) * \
                                                np.exp(-(x_Vx[1] - s_ij[i, j, k, l, m, n, o, p][1]) ** 2) * \
                                                np.exp(-(x_Vx[2] - s_ij[i, j, k, l, m, n, o, p][2]) ** 2) * \
                                                np.exp(-(x_Vx[3] - s_ij[i, j, k, l, m, n, o, p][3]) ** 2) * \
                                                np.exp(-(x_Vx[4] - s_ij[i, j, k, l, m, n, o, p][4]) ** 2) * \
                                                np.exp(-(x_Vx[5] - s_ij[i, j, k, l, m, n, o, p][5]) ** 2) * \
                                                np.exp(-(x_Vx[6] - s_ij[i, j, k, l, m, n, o, p][6]) ** 2) * \
                                                np.exp(-(x_Vx[7] - s_ij[i, j, k, l, m, n, o, p][7]) ** 2)
        return phi_value

    def dict_to_array(self, dict_input, bins):  # converts dictionary to array to compute dot product
        iterator = 0
        phi_as_array = np.ones((bins + 1) ** self.state_space)
        for i, j, k, l, m, n, o, p in iter.product(range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1),
                                                   range(bins + 1)):

            phi_as_array[iterator] = dict_input[i, j, k, l, m, n, o, p]
            iterator += 1
        return phi_as_array

    def act(self, observation, testmode=False):
        self.phi_dict = self.kernel(observation, self.s, self.bins)
        self.phi_array = self.dict_to_array(self.phi_dict, self.bins)
        rand = np.random.uniform()
        if rand < self.epsilon:
            self.action = np.random.randint(0, self.action_space-1)  # hardcoded value !
        else:
            # greedy case. Take action with the maximal Q-value
            self.Q = np.dot(self.weight, self.phi_array)
            self.action = np.argmax(self.Q)  # - 1
        return self.action

    def update(self, observation, action, reward, new_observation, done):
        self.next_phi_dict = self.kernel(new_observation, self.s, self.bins)
        self.next_phi_array = self.dict_to_array(self.next_phi_dict, self.bins)
        self.next_Q = np.dot(self.weight, self.next_phi_array)

        self.weight[self.action, :] += self.alpha * (reward + self.gamma * max(self.next_Q) - self.Q[self.action]) * \
                                       self.phi_array

        # self.current_state = next_state
