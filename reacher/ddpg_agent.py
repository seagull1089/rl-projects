import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0  # 0.000001  # L2 weight decay
LEARN_EVERY = 5
EPSILON = 1.0
EPSILON_DECAY = 0.000001
TRAIN_STEPS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, params):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            params: Dictionary of String to values. 
            - BATCH_SIZE
            - BUFFER_SIZE
            - TRAIN_STEPS
            - LEARN_EVERY 
            - GAMMA
            - LR_ACTOR 
            - LR_CRITIC
            - WEIGHT_DECAY
            - EPSILON
            - EPSILON_DECAY
            - TAU

        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        self.batch_size = params.get('BATCH_SIZE', BATCH_SIZE)
        self.buffer_size = params.get('BUFFER_SIZE', BUFFER_SIZE)
        self.train_steps = params.get('TRAIN_STEPS', TRAIN_STEPS)
        self.learn_every = params.get('LEARN_EVERY', LEARN_EVERY)
        self.gamma = params.get('GAMMA', GAMMA)
        self.epsilon = params.get('EPSILON', EPSILON)
        self.epsilon_decay = params.get('EPSILON_DECAY', EPSILON_DECAY)
        self.tau = params.get('TAU', TAU)
        self.lr_actor = params.get('LR_ACTOR', LR_ACTOR)
        self.lr_critic = params.get('LR_CRITIC', LR_CRITIC)
        self.weight_decay = params.get('WEIGHT_DECAY', WEIGHT_DECAY)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.lr_actor,
            weight_decay=self.weight_decay)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.lr_critic,
            weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size,
            self.buffer_size,
            self.batch_size,
            random_seed)

    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use 
        random sample from buffer to learn.
        """
        # Save experience / reward
        for (state, action, reward, next_state, done) in \
                zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if timestep > 0 and len(self.memory) > self.batch_size and \
                timestep % self.learn_every == 0:
            # print(f"training at {timestep} for 10 batches.")
            for _ in range(self.train_steps):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            # action += self.epsilon * self.noise.sample()
            action += self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
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
        states, actions, rewards, next_states, dones = experiences
        # update the critic
        self.update_critic(states, actions, rewards, next_states, dones)
        # update the actor
        self.update_actor(states)
        # update the target networks
        self.update_target_networks()
        # ensure that the epsilon doesn't go below a minimum value
        # (from the reviewer's feedback)
        self.epsilon = max(0.0001, self.epsilon-self.epsilon_decay)
        self.noise.reset()

    def update_critic(self, states, actions, rewards,
                      next_states, dones):
        """
            Update the critic 
        Args:
            states ([type]): [description]
            actions ([type]): [description]
            rewards ([type]): [description]
            next_states ([type]): [description]
            dones ([type]): [description]
            gamma ([type]): [description]
        """
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

    def update_actor(self, states):
        """
        Compute actor losses and update weights
        """
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_networks(self):
        """
         soft updates to the target models.
        """
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # using normal distribution instead of uniform distribution
        # (from reviewer's feedback)
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.size)
        # len(x) -> size
        # np.array([random.random() for i in range(len(x))])
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
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward",
                                     "next_state", "done"])
        random.seed(seed)
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def convert_to_tensor(self, numpy_array):
        """ wrapper method to convert a numpy array to
         tensor and to move it to GPU/CPU"""
        return torch.from_numpy(numpy_array).float().to(device)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.convert_to_tensor(
            np.vstack([e.state for e in experiences if e is not None]))
        actions = self.convert_to_tensor(
            np.vstack([e.action for e in experiences if e is not None]))
        rewards = self.convert_to_tensor(
            np.vstack([e.reward for e in experiences if e is not None]))
        next_states = self.convert_to_tensor(np.vstack(
            [e.next_state for e in experiences if e is not None]))
        dones = self.convert_to_tensor(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
