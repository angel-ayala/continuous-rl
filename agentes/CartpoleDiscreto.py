#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Angel
"""
import gym
import math
import numpy as np

class CartpoleDiscreto:

    def __init__(self, entorno, alpha = 0.5, epsilon = 0.9, gamma = 1):#0.99):
        self.entorno = entorno
        self.nEstados = entorno.observation_space.shape[0]
        self.nAcciones = entorno.action_space.n
        # print('nEstados', self.nEstados)
        # print('nAcciones', self.nAcciones)

        # policy params
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        # self.Q = np.zeros(entorno.unwrapped.rangos + (entorno.action_space.n, ))
        self.Q = np.random.uniform(-1, 1, entorno.unwrapped.rangos + (entorno.action_space.n, ))
        # print(self.Q)
        self.MIN_ALPHA = 0.1
        self.MIN_EPSILON = 0.01

        # Interactive
        self.feedbackAmount = 0
    # end __init__

    #policy Epsilon-Greedy
    def seleccionarAccion(self, estado):
        #exploracion
        if np.random.rand() <= self.epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado])
    # end seleccionarAccion

    def accionPorFeedback(self, estado, teacherAgent, feedbackProbability):
        if (np.random.rand() < feedbackProbability):
            #get advice
            action = np.argmax(teacherAgent.Q[estado])
            self.feedbackAmount += 1
        else:
            action = self.seleccionarAccion(estado)
        #endIf
        return action
    #end of accionPorFeedback

    def actualizarPolitica(self, estadoAnterior, estadoActual, accion, reward):
        # sarsa
        #next_q = self.Q[estadoActual + (accionActual,)]
        #self.Q[estadoAnterior + (accionAnterior, )] += self.alpha * (reward + self.gamma *
        #                   next_q - self.Q[estadoAnterior + (accionAnterior, )])
        # q learning
        best_q = np.amax(self.Q[estadoActual])
        self.Q[estadoAnterior + (accion, )] += self.alpha * (reward + self.gamma *
                           best_q - self.Q[estadoAnterior + (accion, )])
    # end actualizarPolitica

    def update_explore_rate(self, t):
        # print('e', self.epsilon)
        # self.epsilon = max(self.MIN_EPSILON, self.epsilon * 0.999)
        # return True
        # self.epsilon = max(self.MIN_EPSILON, min(1, 1.0 - math.log10((t+1)/25)))
        self.epsilon = max(self.MIN_EPSILON, min(self.epsilon, 1.0 - math.log10((t+1)/25)))
        # self.epsilon = max(self.MIN_EPSILON, self.epsilon * 0.95)
    # end update_explore_rate

    def update_learning_rate(self, t):
        # self.alpha = max(self.MIN_ALPHA, min(0.5, 1.0 - math.log10((t+1)/25)))
        self.alpha = max(self.MIN_ALPHA, min(self.alpha, 1.0 - math.log10((t+1)/25)))
        # self.alpha = max(self.MIN_ALPHA, self.alpha * 0.95)
    # end update_learning_rate

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []

        for e in range(episodios):
            estadoAnterior = estadoActual = self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                # self.entorno.render()
                accion = self.accionPorFeedback(estadoActual, teacherAgent, feedbackProbability)
                estadoActual, reward, fin, info = self.entorno.step(accion)
                recompensa += reward

                #actualizar valor Q
                self.actualizarPolitica(estadoAnterior, estadoActual, accion, reward)
                estadoAnterior = estadoActual

            # print("fin t=", t+1, 'r=', recompensa, 's=', estadoActual)
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)

            self.update_explore_rate(e)
            self.update_learning_rate(e)

        return recompensas, epsilones, alphas
    # end entrenar

    def guardar(self, filename):
        np.save(filename, self.Q)
    # end guardar

    def cargar(self, filename):
        self.Q = np.load(filename)
    # end cargar

# wrapper para el espacio de observacion de continuo a discreto
class ObservationDiscretize(gym.ObservationWrapper):

    def __init__(self, env, states_boundaries, states_fold):
        super(ObservationDiscretize, self).__init__(env)
        self.sb = states_boundaries
        self.sf = states_fold
        self.unwrapped.rangos = states_fold

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        obs = self.observation(observation)
        self.env.state = obs
        return obs

    def observation(self, obs):
        bucket_indice = []
        for i in range(len(obs)):
            if obs[i] <= self.sb[i][0]:
                bucket_index = 0
            elif obs[i] >= self.sb[i][1]:
                bucket_index = self.sf[i] - 1
            else:
                bound_width = self.sb[i][1] - self.sb[i][0]
                offset = (self.sf[i]-1) *  self.sb[i][0] / bound_width
                scaling = (self.sf[i]-1) / bound_width
                bucket_index = int(round(scaling * obs[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)
