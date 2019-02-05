#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:56:01 2018

@author: Angel
"""
import math
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

class CartpoleDDQN:

    def __init__(self, entorno, alpha = 0.001, epsilon = 1.0, gamma = 0.99):
        self.entorno = entorno
        self.nEstados = entorno.observation_space.shape[0]
        self.nAcciones = entorno.action_space.n
        # print('nEstados', self.nEstados)
        # print('nAcciones', self.nAcciones)

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # print(self.Q)
#        self.MIN_ALPHA = 0.1
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.999
        
        self.streaks = 0 # max50
        self.maxStreaks = 50

        # Interactive
        self.feedbackAmount = 0

        # Continuos Space
        self.memoria = deque(maxlen=2000) # memory replay
        self.batchSize = 64
        
        # create main model and target model
        self.valueFunction = self.build_model()
        self.valueTarget = self.build_model()
        # initialize target model
        self.actualizarValueTarget()
    # end __init__

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.nEstados, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.nAcciones, activation='linear',
                        kernel_initializer='he_uniform'))
        # model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model
    # end build_model
    
    def actualizarValueTarget(self):
        self.valueTarget.set_weights(self.valueFunction.get_weights())
    # end actualizarValueTarget

    def recordar(self, estado, accion, reward, estadoSiguiente, fin):
        self.memoria.append((estado, accion, reward, estadoSiguiente, fin))
        self.update_explore_rate()
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
    # end recordar

    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            # return np.argmax(self.Q[estado])
            return np.argmax(self.valueFunction.predict(estado)[0])
    # end seleccionarAccion

    def accionPorFeedback(self, estado, teacherAgent, feedbackProbability):
        if (np.random.rand() < feedbackProbability):
            #get advice
            action = np.argmax(teacherAgent.valueFunction.predict(estado)[0])
            self.feedbackAmount += 1
        else:
            action = self.seleccionarAccion(estado)
        #endIf
        return action
    #end of accionPorFeedback

    def actualizarPolitica(self): # entrenar modelo
        
        if len(self.memoria) < 1000:#self.train_start:
            return
        miniBatch = random.sample(self.memoria, self.batchSize)
#        print('miniBatch', miniBatch)

        update_input = np.zeros((self.batchSize, self.nEstados))
        update_target = np.zeros((self.batchSize, self.nEstados))
        action, reward, done = [], [], []

        for i in range(self.batchSize):
            update_input[i] = miniBatch[i][0]
            action.append(miniBatch[i][1])
            reward.append(miniBatch[i][2])
            update_target[i] = miniBatch[i][3]
            done.append(miniBatch[i][4])

        target = self.valueFunction.predict(update_input)
        target_next = self.valueFunction.predict(update_target)
        target_val = self.valueTarget.predict(update_target)

        for i in range(self.batchSize):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (
                    target_val[i][a])

        # and do the model fit!
        self.valueFunction.fit(update_input, target, batch_size=self.batchSize,
                       epochs=1, verbose=0)
    # end actualizarPolitica

#    def update_explore_rate(self, t):
    def update_explore_rate(self):
#        if len(self.memoria) < 2000:
#            return True
        if self.epsilon > self.epsilonMin:
#            self.epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t+1)/25)))
#            self.epsilon = min(self.epsilon, 1.0 - math.log10((t+1)/25))
#        if(self.epsilon > self.MIN_EPSILON):
            self.epsilon *= self.epsilonDecay
#        self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.epsilon_decay)

    # end update_explore_rate

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []
#        resetCuenta = False

        for e in range(episodios):
            # estadoAnterior = estadoActual = self.entorno.reset()
            estadoAnterior = estadoActual = np.reshape(self.entorno.reset(), [1, self.nEstados])
            recompensa = 0
            fin = False

            while not fin:
#                self.entorno.render()
                accion = self.accionPorFeedback(estadoActual, teacherAgent, feedbackProbability)
                estadoActual, reward, fin, info = self.entorno.step(accion)
                estadoActual = np.reshape(estadoActual, [1, self.nEstados])
                recompensa += reward
#                print('r', reward)
                
                
#                print('recompensa', recompensa)
                # if an action make the episode end, then gives penalty of -100
                reward = reward if not fin or recompensa == 500 else -100
#                print('reward', reward)
#                self.streaks =  self.streaks + 1 if recompensa == 500 else 0
#                reward += 1 if self.streaks >= 5 else 0
#                if (self.streaks > 0 and fin and recompensa < 500):
#                    reward = 0

                # memory replay
                self.recordar(estadoAnterior, accion, reward, estadoActual, fin)
#                self.update_explore_rate(e)
                # actualizar valor Q entrenar modelo
#                if (self.streaks < self.maxStreaks):
                if recompensa < 495 and not fin:
                    self.actualizarPolitica()

                estadoAnterior = estadoActual
#                resetCuenta = False

            if (recompensa == 500):
                self.streaks += 1
            else:
#                resetCuenta = True
                self.streaks = 0
                
#            self.update_explore_rate(e)
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)
            print('Fin ep {0}\tmem: {1}\trew: {2}\tstr: {3}\teps: {4:0.5f}'.format(e, len(self.memoria), recompensa, self.streaks, self.epsilon))
            # every episode update the target model to be same with model
            self.actualizarValueTarget()
#            print('lMemoria: {}, reward: {}, e: {}'.format(len(self.memoria), recompensa, e))
#            print('lMemoria', len(self.memoria))

#            self.update_explore_rate(e)
#             self.update_learning_rate(e)

        return recompensas, epsilones, alphas
    # end entrenar

    def guardar(self, filename):
        # np.save(filename, self.Q)
        self.valueFunction.save_weights(filename)
        self.valueTarget.save_weights(filename)
    # end guardar

    def cargar(self, filename):
        self.valueFunction.load_weights(filename)
        self.valueTarget.load_weights(filename)
        # self.Q = np.load(filename)
    # end cargar
