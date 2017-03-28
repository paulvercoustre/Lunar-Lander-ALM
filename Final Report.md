# Lunar-Lander-AML
## Final Project for Advanced Machine Learning - Centrale Paris

Amaury Fouret b00702216, Paul Vercoustre b00698926, Andrea Drake Sveen b00689974

_March 27th, 2017_

### Introduction


### Analysis

**Random Agent**
![random agent](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/Random_Agent.png)

#### Q learning

We tried different alphas between 0.001 and 0.5, and found that 0.1 worked best.
Below you can see different ouputs of the model. After running different variations we saw that we got the best results when 
using 3 bins so we expermineted with 3 different gammas, 0.90, 0.95 and 0.99 and saw that 0.99 worked best. We therefore ran
20,000 episodes with parameters; alpha=0.1, bin=3 and gamma=0.99: 

![gamma0.99, 3 bin, 20k](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins3_20kepisode.png)

- Mean reward: 66.70
- Median reward: 16.92
- Standard deviation of rewards: 138.48
- Max total reward: 272.51
- Min total reward: -507.18

**3 bins**

**Gamma = 0.99**

Mean reward: 24.46
Median reward: -25.39
Standard deviation of rewards: 155.90
Max total reward: 272.51
Min total reward: -507.18

2,000 episodes

![gamma0.99, 3 bin, 2k](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/Qlearning_3bins_Gamma099_2k_episode.png)


1,000 episodes

![gamma0.99, 3 bin, 1k](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins3_1000ep_99gamma.png)

2,000 episodes

![gamma0.99, 3 bin, 2k](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins3_2k_ep_99gamma.png)

**Gamma = 0.95**

![gamma0.95, 3 bin](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins3_2kep_95gamma.png)

**Gamma = 0.90**

![gamma0.90, 3 bin](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins3_2kep_9gamma.png)

**5 bins**

gamma = 0.99

1000 episodes

![5 bin, 1k](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins5_1000episodes.png)

2000 episodes

![5 bin 2k](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_5bins_2k_episodes_99gamma.png)

7 bins
Gamma = 0.99

![7bin](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_bins7_2k_ep_99gamma.png)

9 bins
Gamma = 0.99

![9bin](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/qlearning_9bins_2k_ep_99gamma.png)


**Q Learning Agent 1 bin**

![1 bin](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/Qlearning_Agent_Kernel_1bin.png?)

**Deep Q Learning**
We use simple DQN and double DQN with MSE and huber_loss as loss function
one hidden layer with ReLU as activation unit and 64 neurons

![dqn](https://github.com/paulvercoustre/Lunar-Lander-ALM/blob/master/img/dqn.png)
