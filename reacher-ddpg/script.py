#!/usr/bin/env python3
from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
from ddpg_agent import Agent
from tqdm import tqdm

env = UnityEnvironment(file_name='./Reacher_Linux_NoVis_v1/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(
    states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size, action_size, random_seed=0)
env_info = env.reset(train_mode=True)[brain_name]
env_info.vector_observations.shape


def ddpg(n_episodes=10, max_t=1000, print_every=20):
    scores_deque = deque(maxlen=100)

    for i_episode in tqdm(range(1, n_episodes+1)):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            # actions = np.clip(actions, -1,1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            # print(f"{actions} -- {rewards}")
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, t)
            states = next_states
            scores += rewards
            # if rewards[0] > 0.0:
            #    print(rewards)
            if any(dones):
                break

        mean_score = np.mean(scores)
        scores_deque.append(mean_score)

        if i_episode % print_every == 0:
            print('\rEpisode {}\tScore: {:.2f}\t Average Score: {:.2f}'.format(
                i_episode, mean_score, np.mean(scores_deque)))

        if len(scores_deque) == 100 and np.mean(scores_deque) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')
            break

    return scores


print("starting the loop")
scores = ddpg(1000)
env.close()
