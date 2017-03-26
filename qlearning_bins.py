'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
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
        

    goal_average_steps = 1995
    max_number_of_steps = 2000
    last_time_steps = np.ndarray(0)
    n_bins = 5

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    feature1_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature2_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature3_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature4_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature5_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature6_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature7_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    feature8_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.1, gamma=0.90, epsilon=0.1)


    episodes_graph = range(1, 1001)
    rewards_graph = np.empty(1000)
    
    for i_episode in xrange(1000):
        observation = env.reset()

        feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = observation            
        state = build_state([to_bin(feature1, feature1_bins),to_bin(feature2, feature2_bins),
                             to_bin(feature3, feature3_bins),to_bin(feature4, feature4_bins),
			     to_bin(feature5, feature5_bins),to_bin(feature6, feature6_bins),
			     to_bin(feature7, feature7_bins),to_bin(feature8, feature8_bins)])


        total_reward = 0


        for t in xrange(max_number_of_steps):	    	


            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = observation            
            nextState = build_state([to_bin(feature1, feature1_bins),to_bin(feature2, feature2_bins),
                             to_bin(feature3, feature3_bins),to_bin(feature4, feature4_bins),
			     to_bin(feature5, feature5_bins),to_bin(feature6, feature6_bins),
			     to_bin(feature7, feature7_bins),to_bin(feature8, feature8_bins)])

               
            qlearn.learn(state, action, reward, nextState)
            state = nextState
            total_reward += reward
            

            if done:
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                print("Episode finished after {} timesteps".format(t+1))
                break

        rewards_graph[i_episode] = total_reward    
        print("Episode {:d} reward score: {:0.2f}".format(i_episode, total_reward))

    l = last_time_steps.tolist()
    l.sort()
    print("Mean reward: {:0.2f}".format(np.mean(rewards_graph)))
    print("Median reward: {:0.2f}".format(np.median(rewards_graph)))
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    _ = plt.plot(episodes_graph,rewards_graph)
    _ = plt.xlabel("Number of Episodes")
    _ = plt.ylabel("Total Reward per Episode")
    plt.show()
    
    env.monitor.close()

