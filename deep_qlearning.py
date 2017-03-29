import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop, Adam
from keras import backend as K

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = 0.995
        self.e_min = 0.05
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                a = np.argmax(self.model.predict(next_state)[0])
                t = self.target_model.predict(next_state)[0]             
                target[action] = reward + self.gamma * t[a]
            X[i], Y[i] = state, target
        self.model.fit(X, Y, nb_epoch=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')

    np.random.seed(1)
    random.seed(1)
    env.seed(1)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episode = 2500
    episodes_graph = range(episode)
    rewards_graph = np.empty(episode)
    mean_graph = {}
    for ep in range(episode):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0
        for i in range(1000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            reward = reward
            total_reward += reward  
            if done:
                print("episode: {}/{}, epsilon: {:.2}"
                        .format(ep, episode, agent.epsilon))
                break
        if ep % 20 == 0:
            agent.update_target_model()
        agent.replay(32)
        if ep % 50 == 0:
            mean_graph[ep] = np.mean(total_reward)   
        rewards_graph[ep] = total_reward    
        print("Episode {:d} reward score: {:0.2f}".format(ep, total_reward))

    print("Median reward: {:0.2f}".format(np.median(rewards_graph)))
    print("Mean reward: {:0.2f}".format(np.mean(rewards_graph)))
    print("Standard deviation of rewards: {:0.2f}".format(np.std(rewards_graph)))
    print("Max total reward: {:0.2f}".format(np.max(rewards_graph)))
    print("Min total reward: {:0.2f}".format(np.min(rewards_graph)))
    _ = plt.plot(episodes_graph,rewards_graph)
    _ = plt.plot(sorted(mean_graph.keys()), [value for (key, value) in sorted(mean_graph.items())]) 
    _ = plt.plot(episodes_graph, np.mean(rewards_graph) * np.ones(episode))
    _ = plt.xlabel("Number of Episodes")
    _ = plt.ylabel("Total Reward per Episode")
    plt.show()
