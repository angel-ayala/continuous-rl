#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:06:37 2018

@author: angel4ayala@gmail.com
"""
#Libraries Declaration
import gym
import numpy as np
import matplotlib.pyplot as plt
from agentes.CartpoleDDQN import CartpoleDDQN

from helpers.DataFiles import DataFiles

resultsFolder = 'results/cartpole_ddqn/default/'
files = DataFiles()

def plotRewards(filename):
    dataRL = np.genfromtxt(resultsFolder + filename + 'RL.csv', delimiter=',')
    dataIRL = np.genfromtxt(resultsFolder + filename + 'IRL.csv', delimiter=',')
    meansRL = np.mean(dataRL, axis=0)
    meansIRL = np.mean(dataIRL, axis=0)
    print('meansRL', np.average(meansRL))
    print('meansIRL', np.average(meansIRL))

    convolveSet = 50
    convolveRL = np.convolve(meansRL, np.ones(convolveSet)/convolveSet)
    convolveIRL = np.convolve(meansIRL, np.ones(convolveSet)/convolveSet)

    plt.rcParams['font.size'] = 16
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.figure('Collected reward')
    plt.suptitle('Collected reward')

    plt.plot(meansIRL, label = 'Average reward IRL', linestyle = '--', color =  'r')
    plt.plot(meansRL, label = 'Average reward RL', linestyle = '--', color = 'y' )

    plt.plot(convolveIRL, linestyle = '-', color =  '0.2')
    plt.plot(convolveRL, linestyle = '-', color = '0.5' )

    plt.legend(loc=4,prop={'size':12})
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid()

    my_axis = plt.gca()
    #my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
    my_axis.set_xlim(convolveSet, len(meansRL))

    plt.show()

#end of plotRewards method

def trainAgent(tries, episodes, scenario, teacherAgent=None, feedback=0):
    if teacherAgent == None:
        filenameSteps = resultsFolder + 'stepsRL.csv'
        filenameRewards = resultsFolder + 'rewardsRL.csv'
        filenameEpsilons = resultsFolder + 'epsilonsRL.csv'
        filenameAlphas = resultsFolder + 'alphasRL.csv'
    else:
        filenameSteps = resultsFolder + 'stepsIRL.csv'
        filenameRewards = resultsFolder + 'rewardsIRL.csv'
        filenameEpsilons = resultsFolder + 'epsilonsIRL.csv'
        filenameAlphas = resultsFolder + 'alphasIRL.csv'

    files.createFile(filenameEpsilons)
    files.createFile(filenameRewards)
    files.createFile(filenameAlphas)

    for i in range(tries):
        print('Training agent number: ' + str(i+1))

        # agente
        agente = CartpoleDDQN(entorno)
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
    
    tries = 20
    episodes = 600
    feedbackProbability = 0.3

    #entorno
    # scenario = Scenario()
    entorno = gym.make("CartPole-v1")

    #Training with autonomous RL
    print('RL is now training the teacher agent with autonomous RL')
#    teacherAgent = trainAgent(tries, episodes, entorno)
#    teacherAgent.guardar(resultsFolder+'/agenteRL.h5')
#    teacherAgent = CartpoleDDQN(entorno)
#    teacherAgent.cargar(resultsFolder+'/agenteRL.h5')

    #Training with interactive RL
    print('IRL is now training the learner agent with interactive RL')
#    learnerAgent = trainAgent(tries, episodes, entorno, teacherAgent, feedbackProbability)
#    learnerAgent.guardar(resultsFolder + '/agenteIRL.h5')

    plotRewards("rewards")
    plotRewards("epsilons")
    # plotRewards("steps")

    print("Fin")

# end of main method
