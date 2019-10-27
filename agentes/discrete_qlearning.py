#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Angel
"""
import gym
import math
import numpy as np

class DiscreteQlearning:

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
        self.Q = np.random.uniform(-1, 1, entorno.unwrapped.rangos + (entorno.action_space.n, ))
        self.expected_reward = 480
        # print(self.Q)
        self.MIN_ALPHA = 0.1
        self.MIN_EPSILON = 0.01

        # Interactive
        self.feedbackAmount = 0
    # end __init__


    #policy Epsilon-Greedy
    def epsilonGreedyPolicy(self, estado, epsilon = 0.1):
        #exploracion
        if np.random.rand() <= epsilon: #aleatorio
            return self.entorno.action_space.sample()
        #explotacion
        else: # mejor valor Q
            return np.argmax(self.Q[estado])
    # end epsilonGreedyPolicy

    def seleccionarAccion(self, estado):
        accion = self.epsilonGreedyPolicy(estado, self.epsilon)
        return accion
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

    def actualizarPolitica(self, estado, estado_sig, accion, reward):
        # q learning
        best_q = np.amax(self.Q[estado_sig])
        td_target = reward + self.gamma * best_q
        td_error = td_target - self.Q[estado + (accion, )]
        self.Q[estado + (accion, )] += self.alpha * td_error
    # end actualizarPolitica

    def update_explore_rate(self, t):
        self.epsilon = max(self.MIN_EPSILON, min(self.epsilon, 1.0 - math.log10((t+1)/25)))
    # end update_explore_rate

    def update_learning_rate(self, t):
        self.alpha = max(self.MIN_ALPHA, min(self.alpha, 1.0 - math.log10((t+1)/25)))
    # end update_learning_rate

    def test(self, episodios):
        recompensas = []

        for e in range(episodios):
            estado = self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                # self.entorno.render()
                accion = self.epsilonGreedyPolicy(estado, epsilon=self.MIN_EPSILON)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                recompensa += reward

                estado = estado_sig
            recompensas.append(recompensa)

        return recompensas
    #end test

    def entrenar(self, episodios, teacherAgent=None, feedbackProbability=0):
        recompensas = []
        epsilones = []
        alphas = []
        must_train = True
        previous_avg = 0

        for e in range(episodios):
            estado = self.entorno.reset()
            recompensa = 0
            fin = False

            while not fin:
                # self.entorno.render()
                accion = self.accionPorFeedback(estado, teacherAgent, feedbackProbability)
                estado_sig, reward, fin, info = self.entorno.step(accion)
                recompensa += reward

                #actualizar valor Q
                if must_train:
                    self.actualizarPolitica(estado, estado_sig, accion, reward)
                estado = estado_sig

            # print("fin t=", t+1, 'r=', recompensa, 's=', estadoActual)
            recompensas.append(recompensa)
            epsilones.append(self.epsilon)
            alphas.append(self.alpha)

            promedio = np.mean(recompensas)
            must_train = self.test(1)[0] < self.expected_reward or promedio < previous_avg

            self.update_explore_rate(e)
            self.update_learning_rate(e)
            previous_avg = promedio

        return recompensas, epsilones, alphas
    # end entrenar

    def guardar(self, filename):
        np.save(filename, self.Q)
    # end guardar

    def cargar(self, filename):
        self.Q = np.load(filename)
    # end cargar

# wrapper para el espacio de observacion de continuo a discreto
class GymObservationDiscretize(gym.ObservationWrapper):

    def __init__(self, env, states_boundaries, states_fold):
        super(GymObservationDiscretize, self).__init__(env)
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
