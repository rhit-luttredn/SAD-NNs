# Backpropamine: differentiable neuromdulated plasticity.
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License file in this repository for the specific language governing 
# permissions and limitations under the License.


#This code implements the "Grid Maze" task. See Section 4.2 in Miconi et al.
#ICLR 2019 ( https://openreview.net/pdf?id=r1lrAiA5Ym ), or Section 4.5 in
#Miconi et al. ICML 2018 ( https://arxiv.org/abs/1804.02464 ).

import argparse
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import platform
##import makemaze

import numpy as np
#import matplotlib.pyplot as plt
import glob

# Our setup
import torch
from torch.functional import F
from statsmodels.tsa.stattools import adfuller 
# from scipy.stats import linregress

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
print(device)

import copy
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from sad_nns.uncertainty import *
from neurops import *

import random
import pandas as pd


np.set_printoptions(precision=4)
NBDA = 1  # Number of different DA output neurons. At present, the code assumes NBDA=1 and will NOT WORK if you change this.


np.set_printoptions(precision=4)


ADDINPUT = 4 # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs

NBACTIONS = 4  # Up, Down, Left, Right

RFSIZE = 3 # Receptive Field

TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDINPUT + NBACTIONS



def train(paramdict):
    #params = dict(click.get_current_context().params)
    t = pd.DataFrame(columns=['epochs', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10'])
    vdf = pd.DataFrame()
    lossvses = []
    results = pd.DataFrame()
    #TOTALNBINPUTS =  RFSIZE * RFSIZE + ADDINPUT + NBNONRESTACTIONS
    print("Starting training...")
    params = {}
    #params.update(defaultParams)
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
    suffix = "btchFixmod_"+"".join([str(x)+"_" if pair[0] is not 'nbsteps' and pair[0] is not 'rngseed' and pair[0] is not 'save_every' and pair[0] is not 'test_every' and pair[0] is not 'pe' else '' for pair in sorted(zip(params.keys(), params.values()), key=lambda x:x[0] ) for x in pair])[:-1] + "_rngseed_" + str(params['rngseed'])   # Turning the parameters into a nice suffix for filenames

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    # params['rngseed'] = random.randint(0,10000)
    print(params['rngseed'])
    np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
    #print(click.get_current_context().params)

    print("Initializing network")
    # net = Network(params) # TODO:
    net = ModSequential(
        ModLinear(TOTALNBINPUTS, 16),
        ModLinear(16, 16),
        ModLinear(16, 16),
        ModLinear(16, 5,nonlinearity=""),
        track_activations=False,
        track_auxiliary_gradients=True,
        input_shape=(TOTALNBINPUTS)
    ).to(device)
    print ("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("Size (numel) of all optimized elements:", allsizes)
    print ("Total size (numel) of all optimized elements:", sum(allsizes))

    #total_loss = 0.0
    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0*params['lr'], eps=1e-4, weight_decay=params['l2'])

    BATCHSIZE = params['bs']

    LABSIZE = params['msize'] 
    lab = np.ones((LABSIZE, LABSIZE))
    CTR = LABSIZE // 2 


    # Grid maze
    lab[1:LABSIZE-1, 1:LABSIZE-1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help loclization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    lab[CTR,CTR] = 0 



    all_losses = []
    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    all_losses_v = []
    lossbetweensaves = 0
    nowtime = time.time()
    meanrewards = np.zeros((LABSIZE, LABSIZE))
    meanrewardstmp = np.zeros((LABSIZE, LABSIZE, params['eplen']))


    pos = 0
    # DONE
    # hidden = net.initialZeroState()
    # hebb = net.initialZeroHebb()
    # pw = net.initialZeroPlasticWeights()

    print("Total number of parameters:", sum([x.numel() for x in net.parameters()]))


    print("Starting episodes!")

    initial_scores = [] # for north
    for numiter in range(params['nbiter']):

        PRINTTRACE = 0
        # if (numiter+1) % (params['pe']) == 0:
        #     PRINTTRACE = 1


        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        # We always start the episode from the center (when hitting reward, we may teleport either to center or to a random location depending on params['rsp'])
        posr = {}; posc = {}
        rposr = {}; rposc = {}
        for nb in range(BATCHSIZE):
            # Note: it doesn't matter if the reward is on the center (see below). All we need is not to put it on a wall or pillar (lab=1)
            myrposr = 0; myrposc = 0
            while lab[myrposr, myrposc] == 1 or (myrposr == CTR and myrposc == CTR):
                myrposr = np.random.randint(1, LABSIZE - 1)
                myrposc = np.random.randint(1, LABSIZE - 1)
            rposr[nb] = myrposr; rposc[nb] = myrposc
            # print("Reward pos:", rposr, rposc)
            # Agent always starts an episode from the center
            posc[nb] = CTR
            posr[nb] = CTR

        optimizer.zero_grad()
        loss = 0
        lossv = 0

        # DONE
        # hidden = net.initialZeroState()
        # hebb = net.initialZeroHebb()
        # et = net.initialZeroHebb() # Eligibility Trace is identical to Hebbian Trace in shape
        # pw = net.initialZeroPlasticWeights()
        numactionchosen = 0


        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        for numstep in range(params['eplen']):

            inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32') 
        
            labg = lab.copy()
            for nb in range(BATCHSIZE):
                inputs[nb, 0:RFSIZE * RFSIZE] = labg[posr[nb] - RFSIZE//2:posr[nb] + RFSIZE//2 +1, posc[nb] - RFSIZE //2:posc[nb] + RFSIZE//2 +1].flatten() * 1.0
                
                # Previous chosen action
                inputs[nb, RFSIZE * RFSIZE +1] = 1.0 # Bias neuron
                inputs[nb, RFSIZE * RFSIZE +2] = numstep / params['eplen']
                inputs[nb, RFSIZE * RFSIZE +3] = 1.0 * reward[nb]
                inputs[nb, RFSIZE * RFSIZE + ADDINPUT + numactionschosen[nb]] = 1
            
            inputsC = torch.from_numpy(inputs).cuda()
            
            ##### Running the network DONE
            # y, v, hidden, hebb, et, pw = net(Variable(inputsC, requires_grad=False), hidden, hebb, et, pw)  # y  should output raw scores, not probas
            y = net(Variable(inputsC, requires_grad=False))[:,0:4]
            v = net(Variable(inputsC, requires_grad=False))[:,4:5]
            # print(inputsC)

            y = F.softmax(y, dim=1)     # Now y is conveted to "proba-like" quantities
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()  
            logprobs.append(distrib.log_prob(actionschosen))
            numactionschosen = actionschosen.data.cpu().numpy()    # Turn to scalar
            reward = np.zeros(BATCHSIZE, dtype='float32')


            for nb in range(BATCHSIZE):
                myreward = 0
                numactionchosen = numactionschosen[nb]

                tgtposc = posc[nb]
                tgtposr = posr[nb]
                if numactionchosen == 0:  # Up
                    tgtposr -= 1
                elif numactionchosen == 1:  # Down
                    tgtposr += 1
                elif numactionchosen == 2:  # Left
                    tgtposc -= 1
                elif numactionchosen == 3:  # Right
                    tgtposc += 1
                else:
                    raise ValueError("Wrong Action")
                
                reward[nb] = 0.0  # The reward for this step
                if lab[tgtposr][tgtposc] == 1:
                    reward[nb] -= params['wp']
                else:
                    #dist += 1
                    posc[nb] = tgtposc
                    posr[nb] = tgtposr

                # Did we hit the reward location ? Increase reward and teleport!
                # Note that it doesn't matter if we teleport onto the reward, since reward hitting is only evaluated after the (obligatory) move
                if rposr[nb] == posr[nb] and rposc[nb] == posc[nb]:
                    reward[nb] += params['rew']
                    posr[nb]= np.random.randint(1, LABSIZE - 1)
                    posc[nb] = np.random.randint(1, LABSIZE - 1)
                    while lab[posr[nb], posc[nb]] == 1 or (rposr[nb] == posr[nb] and rposc[nb] == posc[nb]):
                        posr[nb] = np.random.randint(1, LABSIZE - 1)
                        posc[nb] = np.random.randint(1, LABSIZE - 1)

            rewards.append(reward)
            vs.append(v)
            sumreward += reward


            # This is the "entropy bonus" of A2C, except that since our version
            # of PyTorch doesn't have an entropy() function, we implement it as
            # a penalty on the sum of squares instead. The effect is the same:
            # we want to penalize concentration of probabilities, i.e.
            # encourage diversity of actions.
            loss += ( params['bent'] * y.pow(2).sum() / BATCHSIZE )  


            if PRINTTRACE:
                print("Step ", numstep, " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS], " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ", numactionschosen[0],
                        " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())), " -Reward (this step, 1st in batch): ", reward[0])



        # Episode is done, now let's do the actual computations


        R = Variable(torch.zeros(BATCHSIZE).cuda(), requires_grad=False)
        gammaR = params['gr']
        for numstepb in reversed(range(params['eplen'])) :
            R = gammaR * R + Variable(torch.from_numpy(rewards[numstepb]).cuda(), requires_grad=False)
            ctrR = R - vs[numstepb][0]
            lossv += ctrR.pow(2).sum() / BATCHSIZE
            loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE  # Need to check if detach() is OK
            #pdb.set_trace()

        loss += params['blossv'] * lossv
        loss /= params['eplen']

        # if PRINTTRACE:
        if True: #params['algo'] == 'A3C':
            #print("lossv: ", lossv.data.cpu().numpy()[0])
            # print("lossv: ", float(lossv))
            lossvses.append(float(lossv))
        bs = sumreward
        bs = np.insert(bs, 0, numiter, axis=0)
        t = pd.concat([t, pd.DataFrame([bs], columns=t.columns)], ignore_index=True)
        # print ("Total reward for this episode (all in batch):", sumreward, "Dist:", dist)

        loss.backward()
        optimizer.step()

        all_grad_norms.append(torch.nn.utils.clip_grad_norm(net.parameters(), params['gc']))
        if numiter > 100:  # Burn-in period for meanrewards
            optimizer.step()
            #pdb.set_trace()

        # TODO: finished training, start growing
        to_add = 0
        if numiter%20==19: # check every 20 epoch
            print("----------------------GROW--------------------------")
            window = all_losses_objective[-20:]

            # using slope of trend line
            # x_axis = [i for i in range(len(window))]
            # lg_result = linregress(window, x_axis)
            # print('slope of trend line: %f' % lg_result.slope)
            # if abs(lg_result.slope) < 0.05:
            #     print("GROW")
            #     to_add = 1

            # using ADF
            adf_result = adfuller(window)

            # print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % adf_result[1])
            if adf_result[1] < 0.05:
                # reject the null hypothesis that it is non stationary. 
                # Therefore the data is stationary, we should grow
                print("GROW")
                to_add = 1

        for i in range(len(net)-1):
            # print("The size of activation of layer {}: {}".format(i, modded_model_grow.activations[str(i)].shape))
            # print("The size of my activation of layer {}: {}".format(i, activation[str(i)].shape))
            #score = orthogonality_gap(modded_model_grow.activations[str(i)])
            # max_rank = net[i].width()
            # score = NORTH_score(net.activations[str(i)], batchsize=batch_size)
            # score = NORTH_score(net[i].weight, batchsize=1, threshold=0)
            # # score = NORTH_score_alter(activation[str(i)], threshold=epsilon)
            # # score = NORTH_score(modded_model_grow[i].weight, batchsize=batch_size)
            # if numiter == 0:
            #     initial_scores.append(score)
            # initScore = 0.97 * initial_scores[i]
            # to_add = max(0, int(net[i].weight.size()[0] * (score - initScore)))
            # if numiter%20==0:
            #     to_add = 1
            # else:
            #     to_add = 0
            # if to_add != 0:
            #     print("Layer {} score: {}/{}, neurons to add: {}".format(i, score, max_rank, to_add))

            # "iterative_orthogonalization", "kaiming_uniform", and "autoinit" are the three options in function north_select, "autoinit" not working
            if to_add > 0:
                net.grow(i, to_add, fanin_weights="kaiming_uniform", optimizer=optimizer)


        lossnum = float(loss)
        lossbetweensaves += lossnum
        all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward.mean())


        if (numiter+1) % params['pe'] == 0:

            print(numiter, "====")
            print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last", params['pe'], "eps.): ", np.sum(all_total_rewards[-params['pe']:])/ params['pe'])
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            if params['type'] == 'plastic' or params['type'] == 'lstmplastic':
                print("ETA: ", net.eta.data.cpu().numpy(), "alpha[0,1]: ", net.alpha.data.cpu().numpy()[0,1], "w[0,1]: ", net.w.data.cpu().numpy()[0,1] )
            elif params['type'] == 'modul':
                print("etaet: ", float(net.etaet), " mean-abs pw: ", torch.mean(torch.abs(pw.data)))
            elif params['type'] == 'rnn':
                print("w[0,1]: ", net.w.data.cpu().numpy()[0,1] )

        if (numiter+1) % params['save_every'] == 0:
            # print("Saving files...")
            losslast100 = np.mean(all_losses_objective[-100:])
            # print("Average loss over the last 100 episodes:", losslast100)
            # print("Saving local files...")
            with open('grad_'+suffix+'.txt', 'w') as thefile:
                for item in all_grad_norms[::10]:
                        thefile.write("%s\n" % item)
            with open('loss_'+suffix+'.txt', 'w') as thefile:
                for item in all_total_rewards[::10]:
                        thefile.write("%s\n" % item)
            torch.save(net.state_dict(), 'torchmodel_'+suffix+'.dat')
            with open('params_'+suffix+'.dat', 'wb') as fo:
                pickle.dump(params, fo)
            # print("Done!")
            # Uber-only stuff:
            if os.path.isdir('/mnt/share/tmiconi'):
                print("Transferring to NFS storage...")
                for fn in ['params_'+suffix+'.dat', 'loss_'+suffix+'.txt', 'torchmodel_'+suffix+'.dat']:
                    result = os.system(
                        'cp {} {}'.format(fn, '/mnt/share/tmiconi/modulmaze/'+fn))
                print("Done!")
    results['loss'] = all_losses_objective
    for j in range(len(net)):
        print("Layer {} weight matrix after growth {}".format(j, net[j].weight.size()))
    vdf['lossv'] = lossvses
    # TODO
    name_tag = "3000g"
    t.to_csv("reward" + name_tag + ".csv")
    results.to_csv("loss" + name_tag + ".csv")
    vdf.to_csv("lossv" + name_tag + ".csv")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    #parser.add_argument("--clamp", type=float, help="maximum (absolute value) gradient for clamping", default=1000000.0)
    #parser.add_argument("--wp", type=float, help="wall penalty (reward decrement for hitting a wall)", default=0.1)
    parser.add_argument("--rew", type=float, help="reward value (reward increment for taking correct action after correct stimulus)", default=1.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.05)
    #parser.add_argument("--pen", type=float, help="penalty value (reward decrement for taking any non-rest action)", default=.2)
    #parser.add_argument("--exprew", type=float, help="reward value (reward increment for hitting reward location)", default=.0)
    parser.add_argument("--bent", type=float, help="coefficient for the entropy reward (really Simpson index concentration measure)", default=0.03)
    #parser.add_argument("--probarev", type=float, help="probability of reversal (random change) in desired stimulus-response, per time step", default=0.0)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    #parser.add_argument("--lsize", type=int, help="size of the labyrinth; must be odd", default=7)
    #parser.add_argument("--rp", type=int, help="whether the reward should be on the periphery", default=0)
    #parser.add_argument("--squash", type=int, help="squash reward through signed sqrt (1 or 0)", default=0)
    #parser.add_argument("--nbarms", type=int, help="number of arms", default=2)
    #parser.add_argument("--nbseq", type=int, help="number of sequences between reinitializations of hidden/Hebbian state and position", default=3)
    #parser.add_argument("--activ", help="activ function ('tanh' or 'selu')", default='tanh')
    #parser.add_argument("--algo", help="meta-learning algorithm (A3C or REI)", default='A3C')
    #parser.add_argument("--rule", help="learning rule ('hebb' or 'oja')", default='hebb')
    parser.add_argument("--type", help="network type ('lstm' or 'rnn' or 'plastic')", default='modul')
    parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=9)
    parser.add_argument("--da", help="transformation function of DA signal (tanh or sig or lin)", default='tanh')
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--gc", type=float, help="gradient norm clipping", default=1000.0)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    #parser.add_argument("--nu", type=float, help="REINFORCE baseline time constant", default=.1)
    #parser.add_argument("--samestep", type=int, help="compare stimulus and response in the same step (1) or from successive steps (0) ?", default=0)
    #parser.add_argument("--nbin", type=int, help="number of possible inputs stimulis", default=4)
    #parser.add_argument("--modhalf", type=int, help="which half of the recurrent netowkr receives modulation (1 or 2)", default=1)
    #parser.add_argument("--nbac", type=int, help="number of possible non-rest actions", default=4)
    parser.add_argument("--rsp", type=int, help="does the agent start each episode from random position (1) or center (0) ?", default=1)
    parser.add_argument("--addpw", type=int, help="are plastic weights purely additive (1) or forgetting (0) ?", default=1)
    #parser.add_argument("--clp", type=int, help="inputs clamped (1), fully clamped (2) or through linear layer (0) ?", default=0)
    #parser.add_argument("--md", type=int, help="maximum delay for reward reception", default=0)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=100)
    #parser.add_argument("--exptime", type=int, help="exploration (no reward) time (must be < eplen)", default=0)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--bs", type=int, help="batch size", default=1)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=3e-6)
    #parser.add_argument("--steplr", type=int, help="duration of each step in the learning rate annealing schedule", default=100000000)
    #parser.add_argument("--gamma", type=float, help="learning rate annealing factor", default=0.3)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=1000)
    parser.add_argument("--pe", type=int, help="number of cycles between successive printing of information", default=100)
    #parser.add_argument("--", type=int, help="", default=1e-4)
    args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
    #train()
    train(argdict)

