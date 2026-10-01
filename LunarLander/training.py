import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque

BATCH_SIZE      = 64
GAMMA           = 0.99       # discount factor
LR              = 5e-4       # learning rate
MEMORY_SIZE     = 500_000    # replay buffer capacity
EPS_START       = 1.0        # starting exploration rate
EPS_END         = 0.01       # minimum exploration rate
EPS_DECAY       = 10_000     # controls how fast epsilon decays (in steps)
TARGET_UPDATE   = 800        # sync target network every N episodes
MAX_EPISODES    = 800

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    def forward(self, x):
        return self.net(x)
'''
class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.value_stream     = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.shared(x)
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        # Combine: subtract mean advantage for stability
        return V + A - A.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, maximum_capacity):
        self.buffer = deque(maxlen=maximum_capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.steps_done = 0
        
        self.policy_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
    
    def select_action(self, state, isRandom):
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if not isRandom:
            eps = 0

        if random.random() < eps:
            return random.randrange(self.action_size)    # explore
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                return self.policy_net(state_t).argmax(dim=1).item()
    
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 
    
        states, action, reward, next_states, done = self.memory.sample(BATCH_SIZE)
        
        current_q = self.policy_net(states).gather(1, action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = reward + GAMMA * max_next_q * (1 - done)
            
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()


env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=True, wind_power=10.0, turbulence_power=1.5)
agent = Agent(state_size=8, action_size=4)
reward_history = []
best_avg = -float('inf')

for episode in range(MAX_EPISODES+1):
    state, info = env.reset()
    total_reward = 0
    done = False    
    
    while not done:
        action = agent.select_action(state, True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.memory.push(state, action, reward, next_state, done)
        
        if agent.steps_done % 4 == 0:
            agent.train_step()
        if agent.steps_done % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        state = next_state
        total_reward += reward
        
    reward_history.append(total_reward)

    
    if episode % 50 == 0:
        avg = np.mean(reward_history[-100:])
        print(f"Episode {episode:>4d} | Avg Reward (last 100): {avg:.0f}")
        if avg > best_avg:
            best_avg = avg
            torch.save(agent.policy_net.state_dict(), "best_model.pth")
        
env.close()


### Testing
env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=True, wind_power=10.0, turbulence_power=1.5, render_mode='human')
for i in range(10):
    state, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        
        action = agent.select_action(state, False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        #agent.memory.push(state, action, reward, next_state, done)
        #agent.train_step()
        
        state = next_state
        total_reward += reward
    print(f"Reward: {total_reward}")

env.close()