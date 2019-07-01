# -*- coding: utf-8 -*-
"""

@author: Angel

"""
#Libraries Declaration
import gym
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from agentes.discrete_qlearning import DiscreteQlearning, GymObservationDiscretize

from helpers.DataFiles import DataFiles
from helpers.Agent import select_advisor

resultsFolder = 'results/discrete_qlearning/tests/'
# Create target Directory if don't exist
if not os.path.exists(resultsFolder):
    os.mkdir(resultsFolder)
    print("Directory " , resultsFolder ,  " Created ")
else:
    input("Output folder already exist. Press Enter to overwrite...")
files = DataFiles()

def plotRewards(filename):
    dataRL = np.genfromtxt(resultsFolder + filename + 'RL.csv', delimiter=',')
    dataIRL = np.genfromtxt(resultsFolder + filename + 'IRL.csv', delimiter=',')
    meansRL = np.mean(dataRL, axis=0)
    meansIRL = np.mean(dataIRL, axis=0)
    print('meansRL', np.average(meansRL), np.max(meansRL), np.min(meansRL), dataRL.shape)
    print('meansIRL', np.average(meansIRL), np.max(meansIRL), np.min(meansIRL), dataIRL.shape)

    convolveSet = 0
#    convolveRL = np.convolve(meansRL, np.ones(convolveSet)/convolveSet)
#    convolveIRL = np.convolve(meansIRL, np.ones(convolveSet)/convolveSet)

    plt.rcParams['font.size'] = 16
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.figure('Collected reward')
    plt.suptitle('Collected reward')

    plt.plot(meansIRL, label = 'Average reward IRL', linestyle = '--', color =  'r')
    plt.plot(meansRL, label = 'Average reward RL', linestyle = '--', color = 'y' )

#    plt.plot(convolveIRL, linestyle = '-', color =  '0.2')
#    plt.plot(convolveRL, linestyle = '-', color = '0.5' )

    plt.legend(loc=4,prop={'size':12})
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid()

    my_axis = plt.gca()
    #my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
#    my_axis.set_xlim(convolveSet, len(meansRL))
    my_axis.set_xlim(convolveSet, 350)

    plt.show()

#end of plotRewards method

def trainAgent(tries, episodes, scenario, teacherAgent=None, feedback=0):
    if teacherAgent == None:
#        filenameSteps = resultsFolder + 'stepsRL.csv'
        filenameRewards = resultsFolder + 'rewardsRL.csv'
        filenameEpsilons = resultsFolder + 'epsilonsRL.csv'
        filenameAlphas = resultsFolder + 'alphasRL.csv'
    else:
#        filenameSteps = resultsFolder + 'stepsIRL.csv'
        filenameRewards = resultsFolder + 'rewardsIRL.csv'
        filenameEpsilons = resultsFolder + 'epsilonsIRL.csv'
        filenameAlphas = resultsFolder + 'alphasIRL.csv'

    files.createFile(filenameEpsilons)
    files.createFile(filenameRewards)
    files.createFile(filenameAlphas)

    for i in range(tries):
        print('Training agent number: ' + str(i+1))

        # agente
        agente = DiscreteQlearning(entorno, epsilon=0.9, alpha=0.5)
        [rewards, epsilons, alphas] = agente.entrenar(episodes, teacherAgent, feedback)
        recompensaPromedio = float(sum(rewards) / float(len(rewards)))

        suffix = '_i' + str(i) + '_r' + str(recompensaPromedio)
        if(teacherAgent is None):
            agentPath = resultsFolder+'/agenteRL'+suffix+'.npy'
        else:
            agentPath = resultsFolder+'/agenteIRL'+suffix+'.npy'

        agente.guardar(agentPath)

        # files.addToFile(filenameSteps, steps)
        files.addFloatToFile(filenameRewards, rewards)
        files.addFloatToFile(filenameEpsilons, epsilons)
        files.addFloatToFile(filenameAlphas, alphas)
    #endfor

    return agente
#end trainAgent method

if __name__ == "__main__":
    print("Interactive RL for Cartpole is running ... ")
    tries = 50
    episodes = 500
    feedbackProbability = 0.3

    #entorno
    cartpole = gym.make("CartPole-v1")

    #discretizar
    limites = list(zip(cartpole.observation_space.low, cartpole.observation_space.high))
    limites[1] = [-0.5, 0.5]
    limites[3] = [-math.radians(50), math.radians(50)]
    rangos = (1, 1, 6, 3)
    entorno = GymObservationDiscretize(cartpole, limites, rangos)

    #Training with autonomous RL
    print('RL is now training the teacher agent with autonomous RL')
    agent = trainAgent(tries, episodes, entorno)
    # agent = DiscreteQlearning(entorno, epsilon=0.9, alpha=0.5)
    # choose advisor
    teacherAgent, number, teacherPath = select_advisor(agent, resultsFolder, entorno)
    print('Using agent:', number, teacherPath)

    # Training with interactive RL
    print('IRL is now training the learner agent with interactive RL')
    learnerAgent = trainAgent(tries, episodes, entorno, teacherAgent, feedbackProbability)

    plotRewards("rewards")
#    plotRewards("alphas")
#    plotRewards("epsilons")
    # plotRewards("steps")

    print("Fin")

# end of main method
