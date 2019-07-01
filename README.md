# Reinforcement learning using continuous states and interactive feedback

<<<<<<< HEAD
Source code for the work "Reinforcement learning using continuous states and interactive feedback" part of my thesis "Continuous Algorithms in Reinforcement Learning with Mobile Agents" (*Desarrollo de algoritmos continuos e aprendizaje por refuerzo interactivo en agentes móviles*) with acknowledge partial support by Universidad Central de Chile under the research project CIP2017030.

Here you can find five different RL agents for two different environments implemented from [Gym](https://github.com/openai/gym/), an OpenAI toolkit.

---

### Updates
-   Improvements of the CartpoleDQN agent 06/2019.
-   Some codes reorganization!.

---

### CartPole-v1 enviroment

For this environments has been developing three agents

*   The first one, ~~*cartpole-discreto*~~ *discrete_qlearning* agent, use the BOXES method for discretization of the state space, storing the Q values in memory.
*   The others two, *continuous_qlearning* agents, uses the DQN and DDQN methods for Q values approximation.

In the continuous observation space, the agent must learn an approximation function to achieve a better generalization. The agent **ContinuousQlearning** was designed to handle continuous observation space.

---
=======
Source code repository for the research project CIP2017030 of Universidad Central de Chile, as partial support.

Here you can find five different RL agents for two different environments implemented from [Gym](https://github.com/openai/gym/), an OpenAI toolkit.

All the agents use the interactive feedback approach, where an external trainer offers advice given a certain probability, to improve the time required for learning.

### CartPole-v1 env

For this environments has been developing three agents

* The first one, *cartpole-discreto*, use the BOXES method for discretization of the state space, storing the Q values in memory.
* The others two, uses the DQN and DDQN methods for Q values approximation.
>>>>>>> a540d3f0a341b026db3c1f37b10371cd9f6d7878

### Arcade Learning Environments

For the ALE environments, the *Space Invaders* is used.

<<<<<<< HEAD
*   One has been implemented with DQN for the RAM variants of the *Space Invaders*.
*   Another has been implemented with DQN but with a CNN for the Q values approximation.

*This environments is still in research how optimize with the state-of-art*
=======
* One has been implemented with DQN for the RAM variants of the *Space Invaders*.
* Another has been implemented with DQN but with a CNN for the Q values approximation.
>>>>>>> a540d3f0a341b026db3c1f37b10371cd9f6d7878

*If you need more information, email me angel4ayala at gmail.com*
