import itertools as iter
import numpy as np

import gym
env = gym.make('LunarLander-v2')

# get info about the action space
print(env.action_space.n)

# ger info about the environment space
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        #env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


