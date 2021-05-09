import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5) # int(1e5)  # replay buffer size
BATCH_SIZE = 256 #128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 5e-4 #1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0 # 0        # L2 weight decay
NOISE_DECAY = 1.0 # 1.0
MU = 0.0
THETA = 0.15
SIGMA = 0.2 # 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed_0, random_seed_1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed_0)

        # Actor Network (w/ Target Network)
        self.actor_local_0 = Actor(state_size, action_size, random_seed_0).to(device)
        self.actor_target_0 = Actor(state_size, action_size, random_seed_0).to(device)
        self.actor_optimizer_0 = optim.Adam(self.actor_local_0.parameters(), lr=LR_ACTOR)

        self.actor_local_1 = Actor(state_size, action_size, random_seed_1).to(device)
        self.actor_target_1 = Actor(state_size, action_size, random_seed_1).to(device)
        self.actor_optimizer_1 = optim.Adam(self.actor_local_1.parameters(), lr=LR_ACTOR)       
        
        # Critic Network (w/ Target Network)
        self.critic_local_0 = Critic(state_size*2, action_size*2, random_seed_0).to(device)
        self.critic_target_0 = Critic(state_size*2, action_size*2, random_seed_0).to(device)
        self.critic_optimizer_0 = optim.Adam(self.critic_local_0.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.critic_local_1 = Critic(state_size*2, action_size*2, random_seed_1).to(device)
        self.critic_target_1 = Critic(state_size*2, action_size*2, random_seed_1).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed_0, mu=MU, theta=THETA, sigma=SIGMA)

        # Replay memory
        self.memory_0 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed_0)
        self.memory_1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed_1)
    
    def step(self, states, actions, rewards, next_states, dones, train=True, iterations=10):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory_0.add(states[0], states[1], actions[0], rewards[0], next_states[0], next_states[1], dones[0])
        self.memory_1.add(states[1], states[0], actions[1], rewards[1], next_states[1], next_states[0], dones[1])

        # Learn, if enough samples are available in memory
        if len(self.memory_0) > BATCH_SIZE and train == True:
            for _ in range(iterations):
                experiences = self.memory_0.sample()
                self.learn(experiences, GAMMA, 0)
                
        if len(self.memory_1) > BATCH_SIZE and train == True:
            for _ in range(iterations):
                experiences = self.memory_1.sample()
                self.learn(experiences, GAMMA, 1)               

    def act(self, state, time, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local_0.eval()
        with torch.no_grad():
            action_0 = self.actor_local_0(state[0]).cpu().data.numpy()
        self.actor_local_0.train()
        if add_noise:
            action_0 += self.noise.sample() * (NOISE_DECAY ** time)
            action_0 = np.clip(action_0, -1, 1)
    
        self.actor_local_1.eval()
        with torch.no_grad():
            action_1 = self.actor_local_1(state[1]).cpu().data.numpy()
        self.actor_local_1.train()
        if add_noise:
            action_1 += self.noise.sample() * (NOISE_DECAY ** time)
            action_1 = np.clip(action_1, -1, 1)    
    
        #print(action_0)
        #return np.hstack((action_0, action_1))
        return torch.tensor([action_0, action_1])
 

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, states_cross, actions, rewards, next_states, next_states_cross, dones = experiences
        agent_a = str(agent)
        agent_b = str((agent+1)%2)
 
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #actions_next = self.actor_target_0(next_states)
        next_actions = getattr(self,'actor_target_' + agent_a)(next_states)
        next_actions_cross = getattr(self, 'actor_target_' + agent_b)(next_states_cross)
        #print(next_states)
        #print(actions_next.shape)
        next_critic_states = torch.cat([next_states, next_states_cross], dim=1)
        next_critic_actions = torch.cat([next_actions, next_actions_cross], dim=1)
        Q_targets_next = getattr(self, 'critic_target_' + agent_a)(next_critic_states, next_critic_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        actions_cross = getattr(self, 'actor_target_' + agent_b)(states_cross)
        critic_states = torch.cat([states, states_cross], dim=1)
        critic_actions = torch.cat([actions, actions_cross], dim=1)
        #print(critic_states.shape)
        #print(actions_cross.shape)
        #print(actions.shape)
        Q_expected = getattr(self, 'critic_local_' + agent_a)(critic_states, critic_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        getattr(self, 'critic_optimizer_' + agent_a ).zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(getattr(self, 'critic_local_' + agent_a).parameters(), 1)
        getattr(self, 'critic_optimizer_' + agent_a).step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = getattr(self, 'actor_local_' + agent_a)(states)
        actions_pred_cross = getattr(self, 'actor_local_' + agent_b)(states_cross)
        actor_actions = torch.cat([actions_pred, actions_pred_cross], dim=1)
        actor_loss = -getattr(self, 'critic_local_' + agent_a)(critic_states, actor_actions).mean()
        # Minimize the loss
        getattr(self, 'actor_optimizer_' + agent_a).zero_grad()
        actor_loss.backward()
        getattr(self, 'actor_optimizer_' + agent_a).step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(getattr(self, 'critic_local_' + agent_a), getattr(self,'critic_target_' + agent_a), TAU)
        self.soft_update(getattr(self, 'actor_local_' + agent_a), getattr(self, 'actor_target_' + agent_a), TAU)                     

     
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    # sigma=0.2
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.standard_normal() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "state_cross", "action", "reward", "next_state", "next_state_cross", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, state_cross, action, reward, next_state, next_state_cross, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_cross, action, reward, next_state, next_state_cross, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states_cross = torch.from_numpy(np.vstack([e.state_cross for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states_cross = torch.from_numpy(np.vstack([e.next_state_cross for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, states_cross, actions, rewards, next_states, next_states_cross, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)