import gymnasium as gym
import numpy as np
#from stable_baselines3 import PPO
import torch
import matplotlib.pyplot as plt

from torch import nn 

env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)

n_episodes = 35000

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=n_episodes)

env = gym.make("CartPole-v1", render_mode='human')
env = gym.wrappers.RecordEpisodeStatistics(env)

rewards = []
for i in range(1000):
    total_reward = 0
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs
    rewards.append(total_reward)
    
plt.plot(rewards)
plt.ylabel("Reward")
plt.xlabel("Test Data")
plt.title("Reward per Test Run")
plt.show()
