# Reinforcement learning using continuous states and interactive feedback

Source code repository for the research project CIP2017030 of Universidad Central de Chile, as partial support.

Here you can find five different RL agents for two different environments implemented from [Gym](https://github.com/openai/gym/), an OpenAI toolkit.

All the agents use the interactive feedback approach, where an external trainer offers advice given a certain probability, to improve the time required for learning.

### CartPole-v1 env

For this environments has been developing three agents

* The first one, *cartpole-discreto*, use the BOXES method for discretization of the state space, storing the Q values in memory.
* The others two, uses the DQN and DDQN methods for Q values approximation.

### Arcade Learning Environments

For the ALE environments, the *Space Invaders* is used.

* One has been implemented with DQN for the RAM variants of the *Space Invaders*.
* Another has been implemented with DQN but with a CNN for the Q values approximation.

*If you need more information, email me angel4ayala at gmail.com*
