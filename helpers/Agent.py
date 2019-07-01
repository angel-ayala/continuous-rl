#Libraries Declaration
import gym
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from helpers.DataFiles import DataFiles

# sort helping
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def select_advisor(agente, agents_folder, entorno):
    log_file = open(agents_folder + '/log.txt', 'w')
    epocas = 500
    # rewards read
    dataRL = np.genfromtxt(agents_folder + '/rewardsRL.csv', delimiter=',')
    meansRL = np.mean(dataRL, axis=1)
    stdRL = np.std(dataRL, axis=1)
    # order by less of standar deviation
    sorted_std = np.argsort(stdRL)
    # agentes guardados
    agentsWeights = glob.glob(agents_folder + '/agenteRL*')
    agentsWeights.sort(key=natural_keys)
    for i in sorted_std:
        print('Checking', agentsWeights[i])
        agente.cargar(agentsWeights[i])
        rewards = agente.test(epocas)
        print(np.mean(rewards))
        if np.mean(rewards) == 500:
            print('passed!')
            log_file.write('Using agent: ' + str(i) + ' ' + agentsWeights[i])
            log_file.close()
            return agente, i, agentsWeights[i]
    log_file.write('None Found')
    log_file.close()
    return None

def generate_tests(agente, agents_folder, entorno, epocas=500, is_irl=False):
    #data save
    files = DataFiles()
    if is_irl:
        agentsWeights = glob.glob(agents_folder + '/agenteIRL*')
        filenameRewardsTest = agents_folder + '/rewardsTestIRL.csv'
    else:
        agentsWeights = glob.glob(agents_folder + '/agenteRL*')
        filenameRewardsTest = agents_folder + '/rewardsTestRL.csv'

    cantidad = len(agentsWeights)
    files.createFile(filenameRewardsTest)
    agentsWeights.sort(key=natural_keys)

    for i in range(cantidad):
        print('Running', i, agentsWeights[i])
        agente.cargar(agentsWeights[i])
        # test
        rewards = agente.test(epocas)
        files.addFloatToFile(filenameRewardsTest, rewards)

    print('done')
