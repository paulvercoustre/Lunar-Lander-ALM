import time
import numpy as np
import matplotlib.pyplot as plt


class Experiment(object):

    def __init__(self, env, agent, nb_episodes):

        """Instantiate the environment, the agent and the number of episodes"""

        self.env = env
        self.agent = agent
        self.nb_episodes = nb_episodes
        self.epsilon = 1  # exploration/exploitation ratio
        self.reward_history = np.empty(nb_episodes)

    def run(self, test=False):

        """Run the agent on the environment for the given number of episodes"""

        # get info about the action space
        print(self.env.action_space)

        # get info about the environment space
        print(self.env.observation_space)
        print(self.env.observation_space.high)
        print(self.env.observation_space.low)

        for i_episode in range(self.nb_episodes):
            tic = time.clock()

            new_observation = self.env.reset()
            done = False
            t = 0
            total_reward = 0

            if self.epsilon > 0.1:  # epsilon decay
                self.epsilon *= 0.99

            while not done:
                # self.env.render()
                action = self.agent.act(new_observation, self.epsilon, testmode=test)
                observation = new_observation
                new_observation, reward, done, info = self.env.step(action)
                # print(reward)
                total_reward += reward
                t += 1

                if not test:
                    self.agent.update(observation, action, reward, new_observation, done)

                if done:
                    print("Episode {} finished after {} timesteps with "
                          "total reward {}".format(i_episode, t + 1, total_reward))

            self.reward_history[i_episode] = total_reward
            toc = time.clock()
            print("Episode {} done in {} seconds".format(i_episode, toc - tic))

        print("Mean reward: {:0.2f}".format(np.mean(self.reward_history)))
        print("Median reward: {:0.2f}".format(np.median(self.reward_history)))
        print("Standard deviation of rewards: {:0.2f}".format(np.std(self.reward_history)))
        print("Max total reward: {:0.2f}".format(np.max(self.reward_history)))
        print("Min total reward: {:0.2f}".format(np.min(self.reward_history)))

        episode_reward = plt.plot(range(1, self.nb_episodes + 1), self.reward_history)
        mean = plt.plot(range(1, self.nb_episodes + 1), np.mean(self.reward_history) * np.ones(self.nb_episodes))
        # _ = plt.legend([episode_reward, mean], ["Episode reward", "Mean reward"])
        _ = plt.xlabel("Number of episodes")
        _ = plt.ylabel("Total reward per episode")

        plt.savefig("Qlearning_Agent_Kernel_1bin")



