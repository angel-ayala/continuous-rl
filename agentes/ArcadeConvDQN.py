#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Angel
"""
import random
import math
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD, RMSprop

import cv2

class ArcadeConvDQN:

    def __init__(self, entorno, alpha = 0.00025, epsilon = 1, gamma = 0.99):
        self.entorno = entorno
        self.nEstados = entorno.observation_space.shape[0]
#        self.nEstados = (84, 84, 1) # final shape conv
#        self.nEstados = map(int,[84,84,1])
        self.nAcciones = entorno.action_space.n
        print('nEstados', self.nEstados)
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
        
        self.tau = 1
        self.tauMin = 0.1
        self.tauDecay = 0.999
        
        self.streaks = 0 # max50
        self.maxStreaks = 50

        # Interactive
        self.feedbackAmount = 0

        # Continuo
        self.memoria = deque(maxlen=1000000) # memory replay
        self.batchSize = 32
        self.minimoMemoria = 50000
        
        # create main model and target model
        self.valueFunction = self.build_model(84, 84, 1, self.nAcciones)
        self.valueTarget = self.build_model(84, 84, 1, self.nAcciones)
        # initialize target model
        self.actualizarValueTarget()
    # end __init__

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
#    def build_model(self):
#        model = Sequential()
#        model.add(Dense(24, input_dim=self.nEstados, activation='relu',
#                        kernel_initializer='he_uniform'))
#        model.add(Dense(24, activation='relu',
#                        kernel_initializer='he_uniform'))
#        model.add(Dense(self.nAcciones, activation='linear',
#                        kernel_initializer='he_uniform'))
#        # model.summary()
#        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
#        return model
#    # end build_model
    
    def build_model(self, width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
#        inputShape = (height, width)
     
    	# if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
#            inputShape = (height, width)
            
    	# first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (8, 8), padding="same",	input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(4, 4)))
        
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (4, 4), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # third set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
     
    	# softmax classifier
        model.add(Dense(classes))
#        model.add(Activation("softmax"))
        
        model.compile(loss='mse', optimizer=RMSprop(lr=self.alpha, rho=0.95, epsilon=0.01))
     	
        # return the constructed network architecture
        return model
    
    def actualizarValueTarget(self):
        self.valueTarget.set_weights(self.valueFunction.get_weights())
    # end actualizarValueTarget

    def recordar(self, estado, accion, reward, estadoSiguiente, fin):
        self.memoria.append((estado, accion, reward, estadoSiguiente, fin))
#        self.update_explore_rate()
#        if self.epsilon > self.epsilon_min:
#            self.epsilon *= self.epsilon_decay
    # end recordar
    
     #policy Epsilon-Greedy
    def epsilonGreedyPolicy(self, estado):
        valoresQ = self.valueFunction.predict(estado, batch_size=1)[0]
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
#            return self.entorno.action_space.sample()
#            return np.random.choice(self.nAcciones, 1, p=valoresQ)
            return np.random.choice(self.nAcciones)
        #explotacion
        else: # mejor valor Q
            # return np.argmax(self.Q[estado])
            return np.argmax(valoresQ)
    # end epsilonGreedyPolicy
    
    def softMaxPolicy(self, estado):
        tau = self.tau
        probabilities = np.zeros(self.nAcciones)
        
        values = self.valueTarget.predict(estado).flatten()
        
#        values = values / 70 if abs(values.max()) > 70 else values
#        print('values a', values)
        
        denom = sum([math.exp(val/tau) for val in values])
        for i in range(self.nAcciones):
            nom = math.exp(values[i] / tau)
            p = float(nom / denom)
            probabilities[i] = p
        action = np.random.choice(range(self.nAcciones), p=probabilities)
        return action
    # end softMaxPolicy

    def seleccionarAccion(self, estado):
        accion = self.epsilonGreedyPolicy(estado)
#        accion = self.softMaxPolicy(estado)
        return accion
#        policy = self.actor.predict(estado, batch_size=1).flatten()
#        return np.random.choice(self.nAcciones, 1, p=policy)[0]
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
#        miniBatch = random.sample(self.memoria, self.batchSize)
#        miniBatch = [self.memoria.get() for _ in range(self.batchSize)]
        miniBatch = list()
        for i in range(self.batchSize):
            miniBatch.append(self.memoria[i])
#        miniBatch = self.memoria[:self.batchSize]
#        print(len(miniBatch))
#        print('miniBatch', miniBatch)

#        update_input = np.zeros((self.batchSize, self.nEstados))
        update_input = np.zeros((self.batchSize, 84, 84, 1))
#        update_target = np.zeros((self.batchSize, self.nEstados))
        update_target = np.zeros((self.batchSize, 84, 84, 1))
        action, reward, done = [], [], []

        for i in range(self.batchSize):
            update_input[i] = miniBatch[i][0]
            action.append(miniBatch[i][1])
            reward.append(miniBatch[i][2])
            update_target[i] = miniBatch[i][3]
            done.append(miniBatch[i][4])

        target = self.valueFunction.predict(update_input)
        target_val = self.valueTarget.predict(update_target)

        for i in range(self.batchSize):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = 0#reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.valueFunction.fit(update_input, target, batch_size=4,
                       epochs=1, verbose=0)
    # end actualizarPolitica

#    def update_explore_rate(self, t):
    def update_explore_rate(self):
#        if len(self.memoria) < 2000:
#            return True
#        if self.epsilon > self.epsilonMin:
#            self.epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t+1)/25)))
#        if(self.epsilon > self.MIN_EPSILON):
#            self.epsilon *= self.epsilonDecay
#        self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.epsilon_decay)
#        if(self.tau > self.t1auMin):
#            self.tau *= self.tauDecay
        eps = self.epsilon - self.epsilonDecay if (self.epsilon > self.epsilonMin) else self.epsilon
        self.epsilon = max(self.epsilonMin, eps)

    # end update_explore_rate
    
#    def to_grayscale(self, img):
#        return np.mean(img, axis=2).astype(np.uint8)

#    def downsample(self, img):
#        return img[::2, ::2]
    
#    def preprocess(self, img):
#        return self.to_grayscale(self.downsample(img))
    
    def preprocesarEstado(self, estado):
        # preprocesamiento
#        return self.preprocess(estado)
        # blanco y negro
        estadoActual = cv2.cvtColor(estado, cv2.COLOR_RGB2GRAY)
#        print('estadoActual', estadoActual.shape, estadoActual)
        # resize
        estadoActual = cv2.resize(estadoActual, (84, 110))
#        print('estadoActual', estadoActual.shape, estadoActual)
        # crop
        estadoActual = estadoActual[6:90, :]        
#        print('estadoActual', estadoActual.shape, estadoActual)
        
        return estadoActual
    
    def transform_reward(self, reward):
        return np.sign(reward)

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []
        frames = 0
        
        # conjunto inicial de memoria random
        fin = True        
        while (len(self.memoria) < self.minimoMemoria):
            if fin:
                estado = np.reshape(self.preprocesarEstado(self.entorno.reset()), [1, 84, 84, 1])
                recompensa = 0
            accion = self.entorno.action_space.sample()
            estadoSiguiente, reward, fin, info = self.entorno.step(accion)
            estadoSiguiente = self.preprocesarEstado(estadoSiguiente)
            recompensa += reward
            
            # if an action make the episode end, then gives penalty of -100
#            reward = 0 if recompensa < 485 else reward
#            reward = reward if not fin or recompensa >= 485 else -100
#            reward += 1 if recompensa >= 485 and not fin else 0
            reward = self.transform_reward(reward)            
            
            # reshape para red
            estadoSiguiente = np.reshape(estadoSiguiente, [1, 84, 84, 1])
            self.recordar(estado, accion, reward, estadoSiguiente, fin)
            
            estado = estadoSiguiente

        for e in range(episodios):
#            estadoAnterior = estadoActual = self.preprocesarEstado(self.entorno.reset())
            estadoAnterior = estadoActual = np.reshape(self.preprocesarEstado(self.entorno.reset()), [1, 84, 84, 1])
            recompensa = 0
#            vidas = 3
            fin = False
            

            while not fin:
                frames += 1
#                print('eps', self.epsilon)    
#                self.entorno.render()
                accion = self.accionPorFeedback(estadoActual, teacherAgent, feedbackProbability)
                estadoActual, reward, fin, info = self.entorno.step(accion)
                
                estadoActual = self.preprocesarEstado(estadoActual)
                recompensa += reward
#                print('r', reward)
                reward = self.transform_reward(reward)
                
                # penalty si no avanza comiendo
#                reward = -1 if not fin and reward == 0 else reward
                
                # penalty si pierde vida
#                if (vidas != info['ale.lives']):
#                    reward = -50
#                    vidas = info['ale.lives']
                # penalty si finaliza el episodio
#                reward = reward if not fin else -100
                    
                
                # reshape para red
                estadoActual = np.reshape(estadoActual, [1, 84, 84, 1])

                # memory replay
                self.recordar(estadoAnterior, accion, reward, estadoActual, fin)

                # actualizar valor Q entrenar modelo
#                if (self.streaks < self.maxStreaks):
                self.actualizarPolitica()
                
                if(frames % 2e3 == 0):
                    self.actualizarValueTarget()

                estadoAnterior = estadoActual
#                resetCuenta = False
                self.update_explore_rate()

#            if (recompensa == 500):
#                self.streaks += 1
#            else:
#                resetCuenta = True
#                self.streaks = 0
            
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)
#            print('Fin ep {0}\tmem: {1}\trew: {2}\tstr: {3}\ttau: {4:0.5f}'.format(e, len(self.memoria), recompensa, self.streaks, self.tau))
            print('Fin ep {0}\tmem: {1}\trew: {2}\tstr: {3}\teps: {4:0.5f}'.format(e, len(self.memoria), recompensa, self.streaks, self.epsilon))
            
#            self.actualizarPolitica()
            # every episode update the target model to be same with model
#            self.actualizarValueTarget()
#            print('lMemoria: {}, reward: {}, e: {}'.format(len(self.memoria), recompensa, e))
#            print('lMemoria', len(self.memoria))

#            self.update_explore_rate()
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
