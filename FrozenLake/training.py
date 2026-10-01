import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

def get_action(obs):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return int(np.argmax(q_values[obs]))

def update(obs, action, next_obs, terminated, reward):
    if terminated:
        future_value = 0
    else:
        future_value = np.max(q_values[next_obs])
        
    predicted = reward + discount_factor * future_value
    error = predicted - q_values[obs][action]
    q_values[obs][action] = q_values[obs][action] + (learning_rate * error)


env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode=None)

q_values = np.zeros((env.observation_space.n, env.action_space.n))

learning_rate = 0.01
start_epsilon = 1.0
epsilon = start_epsilon
n_episodes = 60000
rewards_per_episode = np.zeros(n_episodes)
discount_factor = 0.95
epsilon_decay = start_epsilon / (n_episodes/2)

for i in range(n_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action = get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        update(obs, action, next_obs, terminated, reward)
        done = terminated or truncated
        obs = next_obs
    
    epsilon = max(0, epsilon - epsilon_decay)

    if (epsilon == 0):
        learning_rate = 0.0001
    if reward == 1:
        rewards_per_episode[i] = 1

env.close()

sum_rewards = np.zeros(n_episodes)
for t in range(n_episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake8x8.png')

f = open('frozen_lake8x8.pkl', 'wb')
pickle.dump(q_values, f)
f.close()

env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='human')
for i in range(10):
    input("Press a button to continue: ")
    done = False
    obs, info = env.reset()
    while not done:
        action = get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        obs = next_obs