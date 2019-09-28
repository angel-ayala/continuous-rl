# Reinforcement learning using continuous states and interactive feedback

Angel Ayala, Claudio Henríquez, Francisco Cruz

Universidad Central de Chile

Santiago, Chile

Research in intelligent systems field has led to different learningmethods for machines to acquire knowledge, among them, reinforcement learning (RL). 
Given the problem of the time required to learn how to develop a problem, using RL this work tackles the interactive reinforcement learning (IRL) approach as a way of solution for the training of agents. 
Furthermore, this work also addresses the problem of continuous representations along with the interactive approach. 
In this regards, we have performed experiments with simulated environments using different representations in the state vector in order to show the efficiency of this approach under a certain probability of interaction. 
The obtained results in the simulated environments show a faster learning convergence when using continuous states and interactive feedback in comparison to discrete and autonomous reinforcement learning respectively.

The authors gratefully acknowledge partial support by Universidad Central de Chile under the research project CIP2017030.

DOI: [https://doi.org/10.1145/3309772.3309801](https://doi.org/10.1145/3309772.3309801)

Paper: [Go to file](http://franciscocruz.cl/publications/Ayala_APPIS_2019_Proceedings.pdf)

Here you can find five different RL agents for two different environments implemented from [Gym](https://github.com/openai/gym/), an OpenAI toolkit.

All the agents use the interactive feedback approach, where an external trainer offers advice given a certain probability, to improve the time required for learning.

## Experimental Setup

### CartPole-v1 env

For this environments has been developing three agents

* The first one, *cartpole-discreto*, use the BOXES method for discretization of the state space, storing the Q values in memory.
* The others two, uses the DQN and DDQN methods for Q values approximation.

### Arcade Learning Environments

For the ALE environments, the *Space Invaders* is used.

* One has been implemented with DQN for the RAM variants of the *Space Invaders*.
* Another has been implemented with DQN but with a CNN for the Q values approximation.

*If you need more information, email me angel4ayala at gmail.com*

## Results
<table border="0">
 <tr>
    <td align="center"><b style="font-size:30px">Discrete Results</b></td>
    <td align="center"><b style="font-size:30px">Continuous Results</b></td>
 </tr>
 <tr>
    <td>
      <img src="https://github.com/angel-ayala/continuous-rl/blob/v1/cartpole-discreto/rewards.png?raw=true" height=100% width=100%>
      <br>
      Results of training of 50 agents for the environment CartPole-v1 discretization state vector with BOXES,and feedback probability of 0.3.
   </td>
   <td>
    <img src="https://github.com/angel-ayala/continuous-rl/blob/v1/cartpole-dqn/rewards.png?raw=true" height=100% width=100%>
    <br>
    Results of training of 50 agents for environment CartPole-v1 with continuous representation and feedback probability of 0.3.
   </td>
 </tr>
</table>

            
