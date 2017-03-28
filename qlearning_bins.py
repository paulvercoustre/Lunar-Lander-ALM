import gym
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import itertools as iter
import operator

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}             # store Q values (action value function)
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):

        last_value = self.q.get((state, action), None)
        if last_value is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = last_value + self.alpha * (value - last_value)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        rand = np.random.uniform()
        if rand < self.epsilon:
            self.action = np.random.randint(0, 4)

        count = q.count(maxQ)

        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

def build_state(features):  # function to discretize the space  
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
        
    np.random.seed(1)
    random.seed(1)
    env.seed(1)
    n_bins = 3

    # Discretization of the space
    # Not with the 2 last dimensions because they take only 2 values: 0 or 1
    state1_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    state2_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    state3_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    state4_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    state5_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    state6_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]


    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.1, gamma=0.99, epsilon=1)

    episode = 20000
    interval = 20
    episodes_graph = range(episode)
    rewards_graph = np.empty(episode)
    mean_graph = {}
    for i_episode in range(episode):
        observation = env.reset()
        state1, state2, state3, state4, state5, state6 = observation[0:6]            
        state = build_state([to_bin(state1, state1_bins),to_bin(state2, state2_bins),
                             to_bin(state3, state3_bins),to_bin(state4, state4_bins),
			     to_bin(state5, state5_bins),to_bin(state6, state6_bins),
			     observation[6], observation[7]])

        done = False
        total_reward = 0
        if qlearn.epsilon > 0.1:
            qlearn.epsilon = qlearn.epsilon * 0.995

        while not done:	    	

            action = qlearn.chooseAction(state)
            observation, reward, done, info = env.step(action)
            state1, state2, state3, state4, state5, state6 = observation[0:6]           
            nextState = build_state([to_bin(state1, state1_bins),to_bin(state2, state2_bins),
                             to_bin(state3, state3_bins),to_bin(state4, state4_bins),
			     to_bin(state5, state5_bins),to_bin(state6, state6_bins),
			     observation[6], observation[7]])

               
            qlearn.learn(state, action, reward, nextState)
            state = nextState
            total_reward += reward
        if i_episode % 200 == 0:
            mean_graph[i_episode] = np.mean(total_reward)
        rewards_graph[i_episode] = total_reward    
        print("Episode {:d} reward score: {:0.2f}".format(i_episode, total_reward))


    print("Mean reward: {:0.2f}".format(np.mean(rewards_graph)))
    print("Median reward: {:0.2f}".format(np.median(rewards_graph)))
    print("Standard deviation of rewards: {:0.2f}".format(np.std(rewards_graph)))
    print("Max total reward: {:0.2f}".format(np.max(rewards_graph)))
    print("Min total reward: {:0.2f}".format(np.min(rewards_graph)))
    _ = plt.plot(episodes_graph,rewards_graph, alpha=.5)
    _ = plt.plot(sorted(mean_graph.keys()), [value for (key, value) in sorted(mean_graph.items())])
    _ = plt.plot(episodes_graph, np.mean(rewards_graph) * np.ones(episode))
    _ = plt.xlabel("Number of Episodes")
    _ = plt.ylabel("Total Reward per Episode")
    plt.show()
    
    

