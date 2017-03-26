

class Experiment(object):

    def __init__(self, env, agent, nb_episodes):

        """Instantiate the environment, the agent and the number of episodes"""

        self.env = env
        self.agent = agent
        self.nb_episodes = nb_episodes

    def run(self, test=False):

        """Run the agent on the environment for the given number of episodes"""

        # get info about the action space
        print(self.env.action_space)

        # get info about the environment space
        print(self.env.observation_space)
        print(self.env.observation_space.high)
        print(self.env.observation_space.low)

        for i_episode in range(self.nb_episodes):

            new_observation = self.env.reset()
            done = False
            t = 0

            while not done:
                self.env.render()
                action = self.agent.act(new_observation, testmode=test)
                observation = new_observation
                new_observation, reward, done, info = self.env.step(action)
                print(reward)
                t += 1

                if not test:
                    self.agent.update(observation, action, reward, new_observation, done)

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))

            print("Episode done")



