import gym
env = gym.make('LunarLanderContinuous-v2')
env.reset()
for _ in range(100): # testing changes to number of epochs per episode
    env.render()
    env.step(env.action_space.sample())  # take a random action


