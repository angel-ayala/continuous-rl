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
from keras.optimizers import Adam, RMSprop, Nadam
from keras.models import Sequential

class ContinuousQlearning:

    def __init__(self, entorno, alpha = 0.001, epsilon = 1, gamma = 0.99, input_shape=None):
        self.entorno = entorno
        if input_shape is None:
            self.input_shape = entorno.observation_space.shape
        else:
            self.input_shape = input_shape
        self.nAcciones = entorno.action_space.n
        # Cartpole Limits
        self.envLimits = entorno.observation_space.high
        self.envLimits[0] = 2.4
        self.envLimits[2] = math.radians(50)

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # print(self.Q)
        self.epsilonMin = 0.001

        self.streaks = 0 # max50
        self.maxStreaks = 20
        self.expected_reward = 495

        # Interactive
        self.feedbackAmount = 0

        # Continuo
        self.memoria = deque(maxlen=2000) # memory replay
        self.batchSize = 32
        self.minimoMemoria = 1000

        # create main model and target model
        self.valueFunction = self.build_model()
        self.valueTarget = self.build_model()

        self.render = False
        # initialize target model
        self.actualizarValueTarget()
    # end __init__

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.input_shape, kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Dense(24, kernel_initializer='glorot_uniform', activation='relu'))
        model.add(Dense(self.nAcciones, kernel_initializer='he_uniform', activation='linear'))
        # model.summary()
        opt = Nadam(lr=self.alpha)
        model.compile(loss="mse", optimizer=opt)
        return model
    # end build_model

    def actualizarValueTarget(self):
        self.valueTarget.set_weights(self.valueFunction.get_weights())
    # end actualizarValueTarget

    def recordar(self, estado, accion, reward, estadoSiguiente, fin):
        self.memoria.append((estado, accion, reward, estadoSiguiente, fin))
    # end recordar

    def epsilonGreedyPolicy(self, estado, epsilon = 0.001):
        #exploracion
        if np.random.rand() <= epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.valueFunction.predict(estado)[0])
    # end epsilonGreedyPolicy

    def seleccionarAccion(self, estado):
        accion = self.epsilonGreedyPolicy(estado, self.epsilon)
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

    def get_mini_batches(self):
        if len(self.memoria) < self.minimoMemoria:
            return
        batch = random.sample(self.memoria, self.batchSize)
        batch_shape = (self.batchSize, ) + self.input_shape

        states_mb = np.zeros(batch_shape)
        next_states_mb = np.zeros(batch_shape)
        actions_mb, rewards_mb, dones_mb = [], [], []

        for i in range(self.batchSize):
            states_mb[i] = batch[i][0][0]
            actions_mb.append(batch[i][1])
            rewards_mb.append(batch[i][2])
            next_states_mb[i] = batch[i][3][0]
            dones_mb.append(batch[i][4])

        return states_mb, next_states_mb, actions_mb, rewards_mb, dones_mb

    def actualizarPolitica(self): # entrenar modelo
        update_input, update_target, action, reward, done = self.get_mini_batches()

        target = self.valueFunction.predict(update_input)
        target_val = self.valueTarget.predict(update_target)

        for i in range(self.batchSize):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * np.amax(target_val[i])

        # and do the model fit!
        self.valueFunction.fit(update_input, target, batch_size=self.batchSize,
                         epochs=1, verbose=0)
    # end actualizarPolitica

    def update_explore_rate(self, t):
        self.epsilon = max(self.epsilonMin, min(self.epsilon, 1.0 - math.log10((t+1)/20)))
    # end update_explore_rate

    def state_preprocess(self, state):
        # for nn work
        state = np.reshape(state, list((1,) + self.input_shape))
        return state

    def rewardFunction(self, state):
        state = state[0]
        reward = -1 * abs(state[0])
        # reward = -1
        if (-self.envLimits[2] < state[2] or state[2] < self.envLimits[2]) or (
            -self.envLimits[0] < state[0] or state[0] < self.envLimits[0] ):
            diff = ( ( ( state[2] / self.envLimits[2] ) **2 ) + ( ( state[0] / self.envLimits[0] ) **2 ) ) / 2
            reward = 1 - diff
        return reward
    #end rewardFunction

    # conjunto inicial de memoria random
    def initMemoria(self):
        print("Inicializando Memoria...")
        fin = True
        recompensa = 0
        while (len(self.memoria) < self.minimoMemoria):
            if fin:
               estado = self.state_preprocess(self.entorno.reset())

            accion = self.entorno.action_space.sample()
            estado_sig, reward, fin, info = self.entorno.step(accion)
            estado_sig = self.state_preprocess(estado_sig)

            reward = self.rewardFunction(estado_sig)
            recompensa += reward

            self.recordar(estado, accion, reward, estado_sig, fin)
            estado = estado_sig
        self.actualizarPolitica()
        self.actualizarValueTarget()
        print("OK")
    #end initMemory

    def test(self, episodios, render=False):
        recompensas = []

        for e in range(episodios):
            estado = self.state_preprocess(self.entorno.reset())
            recompensa = 0
            fin = False

            while not fin:
                if render:
                    self.entorno.render()
                accion = self.epsilonGreedyPolicy(estado, epsilon=self.epsilonMin)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                estado_sig = self.state_preprocess(estado_sig)
                reward = self.rewardFunction(estado_sig)
                recompensa += reward

                estado = estado_sig
            recompensas.append(recompensa)

        return recompensas
    #end test

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []
        steps_episode = []
        previous_avg = 0
        must_train = True
        self.initMemoria()

        for e in range(episodios):
            estado = self.state_preprocess(self.entorno.reset())
            ep_steps = 0
            recompensa = 0
            fin = False

            while not fin:
                if self.render:
                    self.entorno.render()
                accion = self.accionPorFeedback(estado, teacherAgent, feedbackProbability)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                estado_sig = self.state_preprocess(estado_sig)
                reward = self.rewardFunction(estado_sig)
                recompensa += reward

                # memory replay
                self.recordar(estado, accion, reward, estado_sig, fin)
                ep_steps += 1

                if must_train:
                    self.actualizarPolitica()

                estado = estado_sig

            if (recompensa >= self.expected_reward):
                self.streaks += 1
            else:
                self.streaks = 0

            promedio = np.mean(recompensas)
            if must_train:
                # target value function for TD
                self.actualizarValueTarget()

            # evaluation
            must_train = self.test(1)[0] < self.expected_reward or promedio < previous_avg

            self.update_explore_rate(e)
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)
            steps_episode.append(ep_steps)

            print('Fin ep {0}\tmem: {1}\trew: {2:0.3f}\tavg: {5:0.3f}\tstr: {3}\teps: {4:0.5f}\te-steps: {6}'.format(e, len(self.memoria), recompensa, self.streaks, self.epsilon, promedio, ep_steps))

            previous_avg = promedio

        print("Ep. Result: avg={0}\tstd={1}\tmin={2}\tmax={3}".format(np.mean(recompensas), np.std(recompensas), np.min(recompensas), np.max(recompensas)))

        metrics = {
            'rewards': recompensas,
            'steps': steps_episode,
            'epsilons': epsilones,
            'alphas': alphas
        }

        return metrics
    # end entrenar

    def guardar(self, filename):
        self.valueFunction.save_weights(filename)
    # end guardar

    def cargar(self, filename):
        self.valueFunction.load_weights(filename)
        self.actualizarValueTarget()
    # end cargar

class ContinuousDoubleQlearning(ContinuousQlearning):

    def actualizarPolitica(self): # entrenar modelo
        update_input, update_target, action, reward, done = self.get_mini_batches()

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
                target[i][action[i]] = reward[i] + self.gamma * target_val[i][a]

        # and do the model fit!
        self.valueFunction.fit(update_input, target, batch_size=self.batchSize,
                         epochs=1, verbose=0)
    # end actualizarPolitica
