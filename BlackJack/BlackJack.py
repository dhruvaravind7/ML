from collections import defaultdict
import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 discount_factor: float = 0.95):
      self.env = env
      self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
      self.lr = learning_rate
      self.discount_factor = discount_factor  # How much we care about future rewards
      self.epsilon = initial_epsilon
      self.epsilon_decay = epsilon_decay
      self.final_epsilon = final_epsilon
      self.training_error = []

    def get_action(self, obs: tuple[int, int, int]) -> int:
      if np.random.random() < self.epsilon:
        return(self.env.action_space.sample())
      else:
        return int(np.argmax(self.q_values[obs]))
    
    def update(self, obs: tuple[int, int, int],
               action: int,
               reward: float,
               next_obs: tuple[int, int, int],
               terminated: bool):
      
      if (terminated):
        future_value = 0
      else:
        future_value = np.max(self.q_values[next_obs])
      predicted = reward + self.discount_factor * future_value
      
      error = predicted - self.q_values[obs][action]
      self.q_values[obs][action] = self.q_values[obs][action] + (self.lr * error)
      self.training_error.append(error)
  
    def decay_epsilon(self):
      self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

      