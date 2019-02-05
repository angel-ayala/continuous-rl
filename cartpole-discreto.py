# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:43:35 2015

@author: cruz

"""
#Libraries Declaration
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from agentes.CartpoleDiscreto import CartpoleDiscreto, ObservationDiscretize

from helpers.DataFiles import DataFiles

resultsFolder = 'results/cartpole_discreto/irl_feed03/'
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
        agente = CartpoleDiscreto(entorno, epsilon=0.9, alpha=0.5)
        # agent = Agent(scenario)
        [rewards, epsilons, alphas] = agente.entrenar(episodes, teacherAgent, feedback)

        # files.addToFile(filenameSteps, steps)
        files.addFloatToFile(filenameRewards, rewards)
        files.addFloatToFile(filenameEpsilons, epsilons)
        files.addFloatToFile(filenameAlphas, alphas)
    #endfor

    return agente
#end trainAgent method

if __name__ == "__main__":
    print("Interactive RL for Cartpole is running ... ")
    tries = 1
    episodes = 350
    feedbackProbability = 0.3

    #entorno
    # scenario = Scenario()
    cartpole = gym.make("CartPole-v1")

    #discretizar
#    limites = list(zip(cartpole.observation_space.low, cartpole.observation_space.high))
#    limites[1] = [-0.5, 0.5]
#    limites[3] = [-math.radians(50), math.radians(50)]
#    rangos = (1, 1, 6, 3)
#    print(limites)
##
#    entorno = ObservationDiscretize(cartpole, limites, rangos)
#
#    #Training with autonomous RL
#    print('RL is now training the teacher agent with autonomous RL')
#    teacherAgent = trainAgent(tries, episodes, entorno)
#    teacherAgent.guardar(resultsFolder+'/agenteRL.npy')
#    # teacherAgent = CartpoleDiscreto(entorno)
#    # teacherAgent.cargar(resultsFolder+'/agente.npy')
#
    #Training with interactive RL
#    print('IRL is now training the learner agent with interactive RL')
#    learnerAgent = trainAgent(tries, episodes, entorno, teacherAgent, feedbackProbability)
#    learnerAgent.guardar(resultsFolder + '/agenteIRL.npy')

    plotRewards("rewards")
#    plotRewards("alphas")
#    plotRewards("epsilons")
    # plotRewards("steps")

    print("Fin")

# end of main method
