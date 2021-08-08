#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 13:32:23 2021

@author: bumba
"""

import gym
import numpy as np
import math
import matplotlib.pyplot as plt

#environment 
env = gym.make('CartPole-v0')

no_actions = env.action_space.n
no_states = env.observation_space.shape[0]

upper_bounds = [
        env.observation_space.high[0], 
        0.5, 
        env.observation_space.high[2], 
        math.radians(50)
        ]
lower_bounds = [
        env.observation_space.low[0], 
        -0.5, 
        env.observation_space.low[2], 
        -math.radians(50)]

no_buckets = (1, 1, 6, 12) # define the number of buckets for each state value 

Q_value = np.zeros(no_buckets + (no_actions,)) #define Q value
print(Q_value.shape)

#hyperparameters
min_explore_rate = 0.01
min_learning_rate = 0.1

max_episodes = 1000
max_time_steps = 250
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0

# choosing an action using epsilon greedy policy
def epsilon_greedy_policy(state,explore_rate):
    if np.random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_value[state])
    return action

# choosing an action using greedy policy
def greedy_policy(state):
    return np.argmax(Q_value[state])

def select_explore_rate(x):
    return max(min_explore_rate, min(1, 1.0 - math.log10((x+1)/25)))

def select_learning_rate(x):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((x+1)/25)))

def discretize(state):
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((no_buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(no_buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)


# TRAINING 
rewards = [] 

for no_episode in range(max_episodes):
    #no_episode = 5
    explore_rate = select_explore_rate(no_episode)
    learning_rate = select_learning_rate(no_episode)
    observation = env.reset()
    current_state = discretize(observation)
    episode_rewards = 0
    for time_step in range(max_time_steps):
        env.render()
        action = epsilon_greedy_policy(current_state, explore_rate)
        new_state, reward, complete, _ = env.step(action)
        new_state = discretize(new_state)
        Q_value[current_state][action] += learning_rate * (reward + discount* np.max(Q_value[new_state]) - Q_value[current_state][action])
        current_state = new_state
        episode_rewards += reward
        
        # print('Episode number : %d' % no_episode)
        # print('Time step : %d' % time_step)
        # print('Selection action : %d' % action)
        # print('Current state : %s' % str(new_state))
        # print('Reward obtained : %f' % reward)
        # #print('Best Q value : %f' % best_q_value)
        # print('Learning rate : %f' % learning_rate)
        # print('Explore rate : %f' % explore_rate)
        # print('Streak number : %d' % no_streaks)
        if complete:
            print('Episode:{}/{} finished with a total reward of: {}'.format(no_episode,max_episodes, episode_rewards))
            break
    #     if complete:
    #        #print('Episode %d finished after %f time steps' % (no_episode, time_step))
    #        print('Episode:{}/{} finished with a total reward of: {}'.format(no_episode,max_episodes, episode_rewards))

    #        if time_step >= solved_time:
    #            no_streaks += 1
    #        else:
    #            no_streaks = 0
    #        break 
    # if no_streaks > streak_to_end:
    #     break
    rewards.append(episode_rewards)
    
# PLOT RESULTS
x = range(max_episodes)
plt.plot(x, rewards)
plt.xlabel('episode')
plt.ylabel('Training cumulative reward')
#plt.savefig('Q_learning_CART.png', dpi=300)
plt.show()    
    
    
# TEST PHASE
current_state = env.reset()
current_state = discretize(current_state)
episode_rewards = 0

for t in range(max_time_steps):
    env.render()
    action = greedy_policy(current_state)
    new_state, reward, complete, _ = env.step(action)
    new_state = discretize(new_state)
    #update_q(current_state, action, reward, new_state, alpha)
    Q_value[current_state][action] += learning_rate * (reward + discount* np.max(Q_value[new_state]) - Q_value[current_state][action])
    current_state = new_state
    episode_rewards += reward

    # at the end of the episode
    if complete:
        print('Test episode finished with a total reward of: {}'.format(episode_rewards))
        break
 
env.close()    
