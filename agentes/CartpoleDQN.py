#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Angel
"""
import math
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Sequential

class CartpoleDQN:

    def __init__(self, entorno, alpha = 0.001, epsilon = 1, gamma = 0.99):
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
        self.epsilonMin = 0.005
        self.epsilonDecay = 0.999
        self.tau = 1
        self.tauMin = 0.01
        
        self.streaks = 0 # max50
        self.maxStreaks = 50

        # Interactive
        self.feedbackAmount = 0

        # Continuo
        self.memoria = deque(maxlen=2000) # memory replay
        self.batchSize = 32
        self.minimoMemoria = 1000
        
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
#        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        #y SGD ?
#        model.compile(loss='mse', optimizer=SGD(lr=self.alpha))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.alpha, rho=0.95))
#        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.alpha))
#        model.compile(loss='cross_entropy)        
#        model.compile(loss='categorical_crossentropy',
#                      optimizer=Adam(lr=0.1))
        return model
    # end build_model
    
    def actualizarValueTarget(self):
        self.valueTarget.set_weights(self.valueFunction.get_weights())
    # end actualizarValueTarget

    def recordar(self, estado, accion, reward, estadoSiguiente, fin):
        self.memoria.append((estado, accion, reward, estadoSiguiente, fin))
#        self.update_explore_rate()
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
    # end recordar
    
    def epsilonGreedyPolicy(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            # return np.argmax(self.Q[estado])
            return np.argmax(self.valueFunction.predict(estado)[0])
    # end epsilonGreedyPolicy
        
#    def softMaxPolicy(self, estado):        
#        policy = self.valueFunction.predict(estado, batch_size=1).flatten()
#        print('policy', policy)
#        return np.random.choice(self.nAcciones, 20, p=policy)[0]
    # end softMaxPolicy
    
    def softMaxPolicy(self, estado):
        tau = self.tau
        probabilities = np.zeros(self.nAcciones)
        
        values = self.valueTarget.predict(estado).flatten()
#        values = np.sign(values)
#        print('values b', values)
#        values = values / 70 if abs(values.max()) > 70 else values
#        print('values a', values)
        
        denom = sum([math.exp(val/tau) for val in values])
#        print('denom', denom)
        for i in range(self.nAcciones):
            nom = math.exp(values[i] / tau)
#            denom = sum(math.exp(val/tau) for val in values)
            p = float(nom / denom)# if denom > 0 else 0
#            print('p', i, p)
#            probabilities[i] = float(nom / denom) if denom > 0 else 0
            probabilities[i] = p
        action = np.random.choice(range(self.nAcciones), p=probabilities)
        return action

    def seleccionarAccion(self, estado):
#        accion = self.epsilonGreedyPolicy(estado)
        accion = self.softMaxPolicy(estado)
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

    def actualizarPolitica(self): # entrenar modelo
        
        if len(self.memoria) < self.minimoMemoria:#self.train_start:
            return
        miniBatch = random.sample(self.memoria, self.batchSize)
#        print('miniBatch', miniBatch)

        update_input = np.zeros((self.batchSize, self.nEstados))
        update_target = np.zeros((self.batchSize, self.nEstados))
        action, reward, done = [], [], []

        for i in range(self.batchSize):
            update_input[i] = miniBatch[i][0] # estado
            action.append(miniBatch[i][1]) # accion
            reward.append(miniBatch[i][2]) # recompensa
            update_target[i] = miniBatch[i][3] # estado siguiente
            done.append(miniBatch[i][4]) # fin

        target = self.valueFunction.predict(update_input)
        target_val = self.valueTarget.predict(update_target)

        for i in range(self.batchSize):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (
                    np.amax(target_val[i]))

#        print('target', target)
        # and do the model fit!
        self.valueFunction.fit(update_input, target, batch_size=self.batchSize,
                       epochs=1, verbose=0)
    # end actualizarPolitica
    
    def update_explore_rate(self, t):
#    def update_explore_rate(self):
        self.tau = max(self.tauMin, min(self.tau, 1.0 - math.log10((t+1)/20)))
#        if len(self.memoria) < 2000:
#            return True
#        if self.tau > self.epsilonMin:
#            self.tau *= self.epsilonDecay
#        if self.epsilon > self.epsilonMin:
#        self.epsilon = max(self.epsilonMin, min(self.epsilon, 1.0 - math.log10((t+1)/15)))
#            self.epsilon = min(self.epsilon, 1.0 - math.log10((t+1)/25))
#        if(self.epsilon > self.MIN_EPSILON):
#            self.epsilon *= self.epsilonDecay
#        self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.epsilon_decay)

    # end update_explore_rate
    
    def test(self, episodios):
        recompensas = []

        for e in range(episodios):
            # estadoAnterior = estadoActual = self.entorno.reset()
            estado = np.reshape(self.entorno.reset(), [1, self.nEstados])
            recompensa = 0
            fin = False

            while not fin:
                accion = self.seleccionarAccion(estado)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                estado_sig = np.reshape(estado_sig, [1, self.nEstados])
                recompensa += reward

                estado = estado_sig
            recompensas.append(recompensa)

        return recompensas
    #end test

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []
        time_steps = 0
#        resetCuenta = False
        
        # conjunto inicial de memoria random
#        fin = True        
#        while (len(self.memoria) < self.minimoMemoria):
#            if fin:
#                estado = np.reshape(self.entorno.reset(), [1, self.nEstados])
#                recompensa = 0
#            accion = self.entorno.action_space.sample()
#            estadoSiguiente, reward, fin, info = self.entorno.step(accion)
#            estadoSiguiente = np.reshape(estadoSiguiente, [1, self.nEstados])
#            recompensa += reward
#            
#            # if an action make the episode end, then gives penalty of -100
#            reward = 0 if recompensa < 485 else reward
#            reward = reward if not fin or recompensa >= 485 else -100
#            reward += 1 if recompensa >= 485 and not fin else 0
#            
#            
#            self.recordar(estado, accion, reward, estadoSiguiente, fin)
#            
#            estado = estadoSiguiente
#        
#        self.actualizarPolitica()
#        self.actualizarValueTarget() 

        for e in range(episodios):
            # estadoAnterior = estadoActual = self.entorno.reset()
            estadoAnterior = estadoActual = np.reshape(self.entorno.reset(), [1, self.nEstados])
            recompensa = 0
            fin = False

            while not fin:
                time_steps += 1
                update = time_steps%250 == 0
#                print('update', update, 't_step', time_steps,  time_steps%25)
#                self.entorno.render()
                accion = self.accionPorFeedback(estadoActual, teacherAgent, feedbackProbability)
                estadoActual, reward, fin, info = self.entorno.step(accion)
                estadoActual = np.reshape(estadoActual, [1, self.nEstados])
                recompensa += reward
#                print('r', reward)
                
                
#                print('recompensa', recompensa)
                # if an action make the episode end, then gives penalty of -100
#                reward = 0 if recompensa < 485 else reward
#                reward = 0 if recompensa < 200 else reward
                reward = reward if not fin or recompensa >= 485 else -100
                
#                print('reward', reward)
#                self.streaks =  self.streaks + 1 if recompensa == 500 else 0
#                reward += 1 if recompensa >= 450 else 0
#                if (self.streaks > 0 and fin and recompensa < 500):
#                    reward = 0
                # boost reward
#                reward += 1 if recompensa >= 100 and not fin else 0
#                reward += 1 if recompensa >= 200 and not fin else 0
#                reward += 1 if recompensa >= 300 and not fin else 0
#                reward += 1 if recompensa >= 400 and not fin else 0
#                reward += 3 if recompensa >= 495 and not fin else 0
#                reward += 1 if recompensa >= 485 and not fin else 0
#                if recompensa > 250 or len(self.memoria) < 2000:
                reward = np.sign(reward)
                # memory replay
                self.recordar(estadoAnterior, accion, reward, estadoActual, fin)
#                self.update_explore_rate(e)
                # actualizar valor Q entrenar modelo
#                if (self.streaks < self.maxStreaks):
#                if (recompensa < 495 or self.streaks%50 == 0) and not fin:
                if (recompensa < 485) and not fin:# or update):
                    self.actualizarPolitica()
                    
                if update:
                    self.actualizarValueTarget()

                estadoAnterior = estadoActual
#                resetCuenta = False

            if (recompensa >= 475):
                self.streaks += 1
            else:
#                resetCuenta = True
                self.streaks = 0
                
#            if update:
            self.update_explore_rate(e)
#            self.update_explore_rate()
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)
#            print('Fin ep {0}\tmem: {1}\trew: {2}\tstr: {3}\teps: {4:0.5f}'.format(e, len(self.memoria), recompensa, self.streaks, self.epsilon))
            print('Fin ep {0}\tmem: {1}\trew: {2}\tstr: {3}\ttau: {4:0.5f}'.format(e, len(self.memoria), recompensa, self.streaks, self.tau))
            # every episode update the target model to be same with model
#            self.actualizarValueTarget()
            if (self.streaks == 50):
                break
#            print('lMemoria: {}, reward: {}, e: {}'.format(len(self.memoria), recompensa, e))
#            print('lMemoria', len(self.memoria))

#            self.update_explore_rate(e)
#             self.update_learning_rate(e)

        return recompensas, epsilones, alphas
    # end entrenar

    def guardar(self, filename):
        # np.save(filename, self.Q)
        self.valueFunction.save_weights(filename)
    # end guardar

    def cargar(self, filename):
        self.valueFunction.load_weights(filename)
        # self.Q = np.load(filename)
    # end cargar
