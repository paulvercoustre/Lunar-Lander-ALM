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


def get_phi(s1,s2,s3,s4,s5,s6,s7,s8,state):
    aux = [range(0,len(s1)),range(0,len(s2)),range(0,len(s3)),range(0,len(s4)),
           range(0,len(s5)),range(0,len(s6)),range(0,len(s7)),range(0,len(s8))]
    combo = list(itertools.product(*aux)) 
    
    idx = 0
    phi_aux = np.zeros(len(s1)*len(s2)*len(s3)*len(s4)*len(s5)*len(s6)*len(s7)*len(s8))

    for idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8 in combo:
        phi_aux[idx] = np.exp(-(state[0]-s1[idx1])**2) * np.exp(-(state[1]-s2[idx2])**2) * np.exp(-(state[2]-s2[idx3])**2)* np.exp(-(state[3]-s2[idx4])**2) * np.exp(-(state[4]-s2[idx5])**2)* np.exp(-(state[5]-s2[idx6])**2)* np.exp(-(state[6]-s2[idx7])**2) * np.exp(-(state[7]-s2[idx8])**2)
        idx += 1

    return phi_aux


class QLearn():
    def __init__(self):
        self.p = 100 #nr of possible coord for x (poz)
        self.k = 100 #nr of possible coord for v (speed)
        self.l = 100
        self.o = 100
        self.n = 100
        self.m = 100
        self.r = 100
        self.t = 100
        
        self.epsilon = 0.1 #for eps greedy req for ql

        self.wght = np.ones((3,(self.p+1)*(self.k+1)*(self.l+1)*(self.o+1)*(self.n+1)*(self.m+1)*(self.r+1)*(self.t+1))) / ((self.p+1)*(self.k+1)*(self.l+1)*(self.o+1)*(self.n+1)*(self.m+1)*(self.r+1)*(self.t+1)) #weight vector for the Q function
        self.alpha = 0.2 #for weights update
        self.gamma = 0.9 #for Q fct, [r + gamma*max Q(s',a')] - Q(s,a)

        self.cur_act = 0 #initialize with 0, do nothing
        self.cur_Q = np.array((0,0,0)) #initialize Q for all actions with 0
        #self.phi = np.zeros((self.p+1)*(self.k+1)) #current phi function
        self.cur_state = np.array((0,0,0,0,0,0,0,0))#np.array((self.mountain_car.x, self.mountain_car.vx)) #initial state

        self.s_i = np.array([-100+j*100/self.p for j in range(0,self.p+1)]) #get s_i index
        self.s_j = np.array([-100+j*100/self.k for j in range(0,self.k+1)]) #get s_j index
        self.s_u = np.array([-100+j*100/self.l for j in range(0,self.l+1)])
        self.s_a = np.array([-100+j*100/self.o for j in range(0,self.o+1)])
        self.s_e = np.array([-100+j*100/self.n for j in range(0,self.n+1)])
        self.s_y = np.array([-100+j*100/self.m for j in range(0,self.m+1)])
        self.s_x = np.array([-100+j*100/self.r for j in range(0,self.r+1)])
        self.s_z = np.array([-100+j*100/self.t for j in range(0,self.t+1)])

        self.phi = np.zeros(len(self.s_i)*len(self.s_j)*len(self.s_u)
                            *len(self.s_a)*len(self.s_e)*len(self.s_y)*len(self.s_x)*len(self.s_z))#get_phi(self.s_i, self.s_j, self.cur_state) #get phi functions


    def act(self, c_state):

        rand=np.random.uniform()

        if rand < self.epsilon:
            self.cur_act = np.random.randint(range(env.action_space.n)) #only 4 possible actions
        else:
            self.cur_state = c_state
            self.phi = get_phi(self.s_i, self.s_j,self.s_u,self.s_a,self.s_e,self.s_y,self.s_x,self.s_z, self.cur_state)
            #print np.max(self.phi)
            self.cur_Q = np.dot(self.wght,self.phi) #get Q estimate for each current action
            self.cur_act = np.argmax(self.cur_Q) - 1 #get max action

        return self.cur_act

    def update(self,next_state, reward):
        phi_fwd = get_phi(self.s_i, self.s_j,self.s_u,self.s_a,self.s_e,self.s_y,self.s_x,self.s_z,next_state) #get phi function for next state//next step phi function
        Q_fwd = np.dot(self.wght,phi_fwd) #get all possible Q values next step

        self.wght[(self.cur_act+1),:] += (self.alpha * (reward + self.gamma * np.max(Q_fwd) - self.cur_Q[(self.cur_act+1)])) * self.phi #update weights

        self.cur_state = next_state #update current state

        self.phi = phi_fwd #update phi function

        #print self.cur_Q #Q_fwd#np.min(self.alpha * (reward + self.gamma * np.max(Q_fwd) - self.cur_Q[(self.cur_act+1)]) * self.phi)

# test class, you do not need to modify this class

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
        

    goal_average_steps = 1995
    max_number_of_steps = 2000
    last_time_steps = np.ndarray(0)

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)


    # The Q-learn algorithm
    qlearn = QLearn()
                    


    episodes_graph = range(1, 1001)
    rewards_graph = np.empty(1000)
    
    for i_episode in xrange(1000):
        observation = env.reset()
        total_reward = 0


        for t in xrange(max_number_of_steps):	    	


            # Pick an action based on the current state
            action = qlearn.act(c_state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)



            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     qlearn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break
               
            qlearn.update(cur_state, action, next_state, reward)
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

