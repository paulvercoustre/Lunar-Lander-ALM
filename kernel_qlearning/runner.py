from setup import Experiment
from agents import RandomAgent, QlearningAgent
import gym

seed = 2
env = gym.make("LunarLander-v2")
env.seed(seed)

# instantiate an agent object
RandAgent = RandomAgent(env.observation_space.shape[0], env.action_space.n, seed=seed)

QAgent = QlearningAgent(env.observation_space.shape[0], env.action_space.n, seed=seed)

nb_episodes = 1000

# instantiate the setup
Trial = Experiment(env, QAgent, nb_episodes)

Trial.run()
