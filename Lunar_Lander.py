import gym
env = gym.make('LunarLander-v2')
env.reset()
for i in range(1000):
    env.render()
    env.step(env.action_space.sample())


