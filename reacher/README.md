[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

This the second project in the Udacity's Deep Reinforcement Learning Nanodegree.

## Introduction

Goal is solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

### Project Details/ Details of the environment: 

- In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
- The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Single vs Multiple Agents in the environment. 
There are two environments available to solve: 

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

For this project, we chose the environment with 20 agents. 

### Solved environment details
- In this (solved) project, the agent was able to reach an average score (over last 100 episodes) of 30 in the first 100 episodes itself. 
- The solution was implemented using a modifield implementation of  DDPG used in the Pendulam task (more details in the Report.md).


## Setup Instructions

### Environment setup Instructions

- make sure you have anaconda installed - https://docs.anaconda.com/anaconda/install/.
- then run this command to create continuous-control environment: ```conda env create -f - environment.yml```
- activate the environment: ```conda activate continuous-control```
- install the jupyter kernel: ```ipython kernel install --name "continuos-control" --user```


### Download the Unity Environment
 Download the environment from one of the links below.  You need only select the environment that matches your operating system:

- **_Version 1: One (1) Agent_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


### To Run the code
- Launch jupyter notebook from the continuous-control folder: jupyter lab
- Open ```Continuous_Control.ipynb``` file and set continuous-control as your kernel.
- Set the Unity Environment path according to your OS (Cell number 2 in the notebook). 
- Run through each of the code cells. (shift + enter)