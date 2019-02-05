#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Angel
"""
import random
import math
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

class ArcadeDQN:

    def __init__(self, entorno, alpha = 0.0002, epsilon = 1.0, gamma = 0.99):
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
        self.epsilonMin = 0.1
#        self.epsilonDecay = 0.999
        self.framesDecay = 1e6
        self.epsilonDecay = float((epsilon - self.epsilonMin) / self.framesDecay)
        
#        self.streaks = 0 # max50
#        self.maxStreaks = 50

        # Interactive
        self.feedbackAmount = 0

        # Continuo
        self.memoria = deque(maxlen=1000000) # memory replay
        self.batchSize = 32
        self.minimoMemoria = 50000
        
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
        model.add(Dense(512, input_dim=self.nEstados, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu',
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
    # end recordar
    
    #policy Epsilon-Greedy
    def epsilonGreedyPolicy(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            # return np.argmax(self.Q[estado])
            return np.argmax(self.valueFunction.predict(estado)[0])
    # end epsilonGreedyPolicy

    def seleccionarAccion(self, estado):
        accion = self.epsilonGreedyPolicy(estado)
        return accion
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

#    def actualizarPolitica(self): # entrenar modelo
#        
#        if len(self.memoria) < self.minimoMemoria:
#            return
#        miniBatch = random.sample(self.memoria, self.batchSize)
##        print('miniBatch', miniBatch)
#
#        update_input = np.zeros((self.batchSize, self.nEstados))
#        update_target = np.zeros((self.batchSize, self.nEstados))
#        action, reward, done = [], [], []
#
#        for i in range(self.batchSize):
#            update_input[i] = miniBatch[i][0]
#            action.append(miniBatch[i][1])
#            reward.append(miniBatch[i][2])
#            update_target[i] = miniBatch[i][3]
#            done.append(miniBatch[i][4])
#
#        target = self.valueFunction.predict(update_input)
#        target_val = self.valueTarget.predict(update_target)
#
#        for i in range(self.batchSize):
#            # Q Learning: get maximum Q value at s' from target model
#            if done[i]:
#                target[i][action[i]] = reward[i]
#            else:
#                target[i][action[i]] = reward[i] + self.gamma * (
#                    np.amax(target_val[i]))
#
#        # and do the model fit!
#        self.valueFunction.fit(update_input, target, batch_size=self.batchSize,
#                       epochs=1, verbose=0)
#    # end actualizarPolitica
    
    def actualizarPolitica(self): # entrenar modelo
        
        if len(self.memoria) < self.minimoMemoria:
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
        eps = self.epsilon - self.epsilonDecay if (self.epsilon > self.epsilonMin) else self.epsilon
        self.epsilon = max(self.epsilonMin, eps)
    # end update_explore_rate
    
    def transform_reward(self, reward):
        return np.sign(reward)

    def test(self, episodios):
        recompensas = []
        frame = 0

        for e in range(episodios):
            # estadoAnterior = estadoActual = self.entorno.reset()
            estado = np.reshape(self.entorno.reset(), [1, self.nEstados])
            recompensa = 0
            fin = False

            while not fin:
                if frame == 0 or (frame % 4) == 0:
                    accion = self.seleccionarAccion(estado)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                estado_sig = np.reshape(estado_sig, [1, self.nEstados])
                recompensa += reward

                estado = estado_sig
            recompensas.append(recompensa)

        return recompensas
    #end test

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0, carpeta='results/', idx=0):
        recompensas = []
        epsilones = []
        alphas = []
#        resetCuenta = False
        frame = 0

        for e in range(episodios):
            # estadoAnterior = estadoActual = self.entorno.reset()
            estado = np.reshape(self.entorno.reset(), [1, self.nEstados])
            recompensa = 0
#            vidas = 3
            fin = False

            while not fin:
#                self.entorno.render()
                if frame == 0 or (frame % 4) == 0:
                    accion = self.accionPorFeedback(estado, teacherAgent, feedbackProbability)
                    
                estado_sig, reward, fin, info = self.entorno.step(accion)
                estado_sig = np.reshape(estado_sig, [1, self.nEstados])
                recompensa += reward
#                print('r', reward)
                reward = self.transform_reward(reward)
                
                # memory replay
                self.recordar(estado, accion, reward, estado_sig, fin)
                
                if frame > 0 and (frame % 1e5) == 0:
                    if(teacherAgent is None):
                        suffix = 'RL' + str(idx)
                    else:
                        suffix = 'IRL' + str(idx)
                    self.guardar(carpeta + suffix + 'agente_r' +str(sum(recompensas)/len(recompensas)))

                # actualizar valor Q entrenar modelo
                self.actualizarPolitica()

                estado = estado_sig
#                resetCuenta = False
                frame += 1
                self.update_explore_rate()
            
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)
            print('Fin ep {0}\tfrm: {1}\tmem: {2}\trew: {3}\teps: {4:0.5f}'.format(e, frame, len(self.memoria), recompensa, self.epsilon))
            # every episode update the target model to be same with model
            self.actualizarValueTarget()
#            print('lMemoria: {}, reward: {}, e: {}'.format(len(self.memoria), recompensa, e))
#            print('lMemoria', len(self.memoria))

#             self.update_learning_rate(e)

        return recompensas, epsilones, alphas
    # end entrenar

    def guardar(self, filename):
#         np.save(filename, self.Q)
        self.valueFunction.save_weights(filename)
    # end guardar

    def cargar(self, filename):
        self.valueFunction.load_weights(filename)
#         self.Q = np.load(filename)
    # end cargar
