###################################################
#    This file is part of py-smc2.
#    http://code.google.com/p/py-smc2/
#
#    py-smc2 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    py-smc2 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py-smc2.  If not, see <http://www.gnu.org/licenses/>.
###################################################

#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from numpy import random, power, sqrt, exp, zeros, \
        ones, mean, average, prod, log, sum, repeat, \
        array, float32, int32, zeros_like, newaxis, \
        transpose, dot, floor, maximum, minimum
from scipy.stats import norm, truncnorm, gamma, nbinom
import scipy.weave as weave
import os
import math
from src.statespacemodel import SSM

################################################################
#data from http://www.esapubs.org/archive/ecol/E093/025/suppl-1.htm

# logistic diffusion Model:  
# state: N_t2 = N_t1 exp(r(t2-t1) + sigma RW(t2-t1))
# observ: 
# params[0]: tau
# params[1]: sigma
# params[2]: r

################################################################
#the first year is dummy
Time_start = 1973
Time = [ #[1973, \
1973.497, 1973.75, 1974.163, 1974.413, 1974.665, 1975.002, 1975.245, 1975.497, 1975.75, 1976.078, 1976.33, 1976.582, 1976.917, \
1977.245, 1977.497, 1977.665, 1978.002, 1978.33, 1978.582, 1978.832, 1979.078, 1979.582, 1979.832, 1980.163, 1980.497, 1980.75, \
1980.917, 1981.163, 1981.497, 1981.665, 1981.917, 1982.163, 1982.413, 1982.665, 1982.917, 1983.163, 1983.413, 1983.665, 1983.917, \
1984.163, 1984.413] #shift to zero at initial time, note that the prior of state should be very non-informative

#NOTE: this file is about true data generating model, and some helpful functions used in smc2 class  


#IMPORTANT: whether or not use H score for inference, i.e. to update the theta particle weights 
HScoreInference = 0
#for that purpose, I updated codes that involve
#logLike = log(mean(self.xweights)) 
#transitionAndWeight(...)
#essentially I added HScoreInference as a flag, and transitionAndWeight() an additional output 

#IMPORTANT: whether or not the observations are continuously valued (differ a lot in the computing of H score)
continuousObs = 0

### these functions take untransformed parameters as arguments
#### See src/models.py for explanations about the model functions.
def firstStateGenerator(parameters, size):
    #return random.normal(size = size, loc = 0, scale = 1)[:, newaxis] #new axis is for adding a new dimension 
    #transpose(random.multivariate_normal(repeat(0, nbparameters), proposalcovmatrix, size = currentvalue.shape[1]))
    #first_state = zeros((size, M))
    first_state = exp(random.multivariate_normal(repeat(0, 1), [[10^2]], size = size)) #dim: size x 1
    return first_state  
def observationGenerator(states, parameters): 
    #input states is 2D: numStates x dimObs, parameters is 1D
    obs = zeros((states.shape[0], 2))
    for i in range(states.shape[0]):
        #obs[i,:] = transpose(random.multivariate_normal(repeat(states[i,0],1), [[parameters[0] * states[i,0] ** 2]], size = 2))
        p = 1 / (1 + parameters[0] * states[i,0])
        p = minimum(p, 1-1e-7) 
        p = maximum(p, 1e-7)
        n = maximum(1, floor( states[i,0] * p / (1-p) ) ).astype(int32)
        obs[i,:] = nbinom.rvs(n, p, 2)
    return obs
    # p = 1 / (1 + parameters[0] * states)
    # n = floor( states * p / (1-p) )
    # return nbinom.rvs(n, p, size = states.shape[0])

    #the following funcion is a key: it sample the last states, and output new states & weights (filtering density)
    #under log score inference, the average weights is the unbiased loglik 
    #under H score inference, we use the updated weights and F/L identity to compute the Hscore of one obser 
    #evaluated at each theta particle 
def transitionAndWeight(states, y, parameters, t): #here, the last argument t now becomes useful 
    #the duration from last state to the current 
    if t == 0:
        time = Time[0]-Time_start #the duration from last state to the current 
    else:
        time = Time[t]-Time[t-1]
    # Negative bin is too slow in the c code below, so approximate it by Gaussisan 
    # code = \
    # """
    # int temp1;
    # int temp2;
    # double p;
    # int n;
    # for (int j = 0; j < Ntheta; j++)
    # {
    #     for (int k = 0; k < Nx; k++)
    #     {
    #         states(k, 0, j) =  states(k, 0, j) * exp( parameters(2, j) * (%(time)s) + parameters(1, j) * noise(k,j) ); 
    #         p = 1 / (1 + parameters(0, j) * states(k, 0, j));
    #         n = (int) states(k, 0, j) * p / (1-p); 
    #         temp1 = 1;
    #         for (int i = 1; i <= y(0); i++)
    #         {
    #             temp1 = temp1 * (1-p) / i;
    #         }
    #         for (int i = 1; i <= n+y(0)-1; i++)
    #         {
    #             temp1 = temp1 * i; 
    #         }
    #         for (int i = 1; i <= n-1; i++)
    #         {
    #             temp1 = temp1 * p / i;
    #         }
    #         temp1 = temp1 * p;

    #         temp2 = 1;
    #         for (int i = 1; i <= y(1); i++)
    #         {
    #             temp2 = temp2 * (1-p) / i;
    #         }
    #         for (int i = 1; i <= n+y(1)-1; i++)
    #         {
    #             temp2 = temp2 * i; 
    #         }
    #         for (int i = 1; i <= n-1; i++)
    #         {
    #             temp2 = temp2 * p / i;
    #         }
    #         temp2 = temp2 * p;

    #         weights(k, j) = temp1 * temp2; 
    #     }
    # } 
    # double HScoreInferenceFlag = (double) %(HScoreInference)s;
    # if ( HScoreInferenceFlag > 0 )
    # {
    #     double tbd = 0;
    # }
    # """ % {"time": time, "HScoreInference": HScoreInference} #the inline code first sample states and udpate weights, then normalize the weights, 
    
    # use Gaussian approx. 
    # code = \
    # """
    # double R;
    # double loglik1;
    # double loglik2;
    # for (int j = 0; j < Ntheta; j++)
    # {
    #     for (int k = 0; k < Nx; k++)
    #     {
    #         states(k, 0, j) =  states(k, 0, j) * exp( parameters(2, j) * (%(time)s) + parameters(1, j) * noise(k,j) ); 
    #         R = states(k, 0, j) + parameters(0, j) * states(k, 0, j) * states(k, 0, j); 
    #         loglik1 = -0.9189385 - 0.5 * log(R) - 0.5 / R * ((double) y(0) - states(k, 0, j)) * ((double) y(0) - states(k, 0, j));
    #         loglik2 = -0.9189385 - 0.5 * log(R) - 0.5 / R * ((double) y(1) - states(k, 0, j)) * ((double) y(1) - states(k, 0, j));
    #         weights(k, j) = loglik1 + loglik2; 
    #     }
    # } 
    # double HScoreInferenceFlag = (double) %(HScoreInference)s;
    # if ( HScoreInferenceFlag > 0 )
    # {
    #     ;
    # }
    # """ % {"time": time, "HScoreInference": HScoreInference}

    # the code below uses negative bin likelihood 
    code = \
    """
    for (int j = 0; j < Ntheta; j++)
    {
        for (int k = 0; k < Nx; k++)
        {
            states(k, 0, j) =  states(k, 0, j) * exp( parameters(2, j) * (%(time)s) + parameters(1, j) * noise(k,j) ); 
        }
    } 
    double HScoreInferenceFlag = (double) %(HScoreInference)s;
    if ( HScoreInferenceFlag > 0 )
    {
        ;
    }
    """ % {"time": time, "HScoreInference": HScoreInference}

    dimY = y.shape[0]
    if dimY == 2:
        y0 = int(y[0])
        y1 = int(y[1])
    # y = array([y]) #input to the c++ code must be array type 
    Nx = states.shape[0]
    Dx = states.shape[1] #state dimension, which is M here
    Dparam = parameters.shape[0] #dimension of the postulated model/parameter 
    Ntheta = states.shape[2]
    weights = zeros((Nx, Ntheta))
    weightsNormalized = zeros((Nx, Ntheta)) #normalized and exponentialed weights 
    grad = zeros(Ntheta)
    lap = zeros(Ntheta)
    simpleHScore = zeros(Ntheta)
    noise = random.multivariate_normal(repeat(0,1), [[time]], (Nx, Ntheta)) #dim: Nx x Ntheta x M
    weave.inline(code,['Nx', 'Ntheta', 'Dx', 'Dparam', 'states', 'y', 'parameters', 'noise', 'weights', \
            'simpleHScore', 'grad', 'lap', 'weightsNormalized', 'time'], \
            type_converters=weave.converters.blitz, libraries = ["m"])
    if dimY == 2: #make sure we are in inference, instead of synthetic-data generating
        p = zeros((Nx, Ntheta))
        n = zeros((Nx, Ntheta))
        for k in range(Nx):
            p[k,:] = 1 / (1 + parameters[0,:] * states[k,0,:])
            p[k,:] = minimum(p[k,:], 1-1e-7) 
            p[k,:] = maximum(p[k,:], 1e-7) #to prevent overflow  
            states[k,0,:] = minimum(states[k,0,:], 1e7) #to prevent overflow  
            n[k,:] = maximum(1, floor(states[k,0,:] * p[k,:] / (1-p[k,:])) ).astype(int32)
            weights[...] = nbinom.logpmf(y0, n, p) + nbinom.logpmf(y1, n, p) 
    return {"states": states , "weights": weights, "simpleHScore": simpleHScore} #the xweights are logged
    #the following function is to compute composite H score using current state, obser, theta particle, and weights
    #IMPORTANT note: weave.inline() should include the name of the variables you want to return 
# def computeHscore(states, y, parameters, xweights, thetaweights, t):
#     #xweights is of size Nx x Ntheta (unnormalized); thetaweights is also unnormalized 
#     time = Time[t]-Time[0]
#     code = \
#     """ 
#     double temp;
#     double ave_grad = 0;
#     double R;
#     for (int i = 0; i < Ntheta; i++)
#     {
#         for (int j = 0; j < Nx; j++)
#         {
#             R = states(j, 0, i) + parameters(0, i) * states(j, 0, i) * states(j, 0, i);
#             temp = - 1 / R * ((double) y(0) - states(j, 0, i));
#             grad(i) = grad(i)  + temp * xweights(j, i);
#             lap(i) = lap(i) + ( - 1 / R + temp * temp ) * xweights(j, i);
#         }
#         grad(i) = grad(i) / xNormConst(i);
#         lap(i) = -grad(i) * grad(i) + lap(i) / xNormConst(i);            
#         hScore_theta(i) = lap(i) + (grad(i) * grad(i)) / 2;
#     }
#     for (int i = 0; i < Ntheta; i++)
#     {
#         simpleHScore(0) = simpleHScore(0) + hScore_theta(i) * thetaweights(i);
#         ave_grad = ave_grad + grad(i) * thetaweights(i);
#     }        
#     compositeHScore(0) = simpleHScore(0);
#     for (int i = 0; i < Ntheta; i++)
#     {
#         compositeHScore(0) = compositeHScore(0) + 0.5 * (grad(i) - ave_grad) * (grad(i) - ave_grad) * thetaweights(i);
#     }
#     """ % {"time": time}
#     y = array([y])
#     Nx = states.shape[0]
#     Ntheta = states.shape[2]
#     Dparam = parameters.shape[0] #dimension of the postulated model/parameter
#     thetaweights = thetaweights / sum(thetaweights)  #first transform to normalized ones      
#     xNormConst = sum(xweights, axis=0)
#     grad = zeros(Ntheta)  
#     lap = zeros(Ntheta)
#     hScore_theta = zeros(Ntheta) #store simple H score for each theta particle 
#     compositeHScore = zeros(1)
#     simpleHScore = zeros(1)
#     weave.inline(code,['states', 'y', 'parameters', 'thetaweights', 'xweights', 'xNormConst', 'hScore_theta', \
#             'Nx', 'Ntheta', 'Dparam', 'grad', 'lap', 'compositeHScore', 'simpleHScore'], \
#             type_converters=weave.converters.blitz, libraries = ["m"])
#     return {"simpleHScore" : simpleHScore , "compositeHScore" : compositeHScore}
#call the above: computeHscore(self.xparticles, self.observations[t], self.thetaparticles, xweights_safe, thetaweights)
def computeHscore(states, y, parameters_last, xweights_last, thetaweights_last, t):
    #xweights is of size Nx x Ntheta (unnormalized); thetaweights is also unnormalized 
    #the ouput compositeHScore is for model comparision, while simpleHScore is for H-based Bayes 
    #the composite score is to evaluate the t-time obs. at the predictive distribution made at time t-1
    y0 = int(y[0])
    y1 = int(y[1])
    Nx = states.shape[0]
    Ntheta = states.shape[2]
    #compute the weighting matrix 
    W = zeros((Nx, Ntheta))
    thetaweights_last = thetaweights_last / sum(thetaweights_last)  #first transform to normalized ones
    xNormConst = sum(xweights_last, axis=0)
    for k in range(Ntheta):
        W[:,k] =  thetaweights_last[k] * xweights_last[:,k] / xNormConst[k]   

    #compute the conditional density given a last theta-particle and a last x-particle
    p = zeros((Nx, Ntheta))
    n = zeros((Nx, Ntheta)) 
    for k in range(Nx):
        p[k,:] = 1 / (1 + parameters_last[0,:] * states[k,0,:])
        p[k,:] = minimum(p[k,:], 1-1e-7) 
        p[k,:] = maximum(p[k,:], 1e-7) #to prevent overflow  
        states[k,0,:] = minimum(states[k,0,:], 1e7) #to prevent overflow  
#        try:
        n[k,:] = maximum(1, floor(states[k,0,:] * p[k,:] / (1-p[k,:])) ).astype(int32)
        # except RuntimeError:
        #     print states[k,0,:], '\n', p[k,:]
 #       n[k,:] = minimum(n[k,:], 1e7) #to prevent overflow  
    # conDensi = zeros((Nx, Ntheta))
    
    score_y0 = average(a = nbinom.pmf(y0, n, p) , weights = W)
    score_y0_p = average(a = nbinom.pmf(y0+1, n, p) , weights = W) #plus 1 
    score_y1 = average(a = nbinom.pmf(y1, n, p) , weights = W)
    score_y1_p = average(a = nbinom.pmf(y1+1, n, p) , weights = W)
    # print '\n', score_y0_p, score_y0
    if y0 == 0:
        score0 = score_y0_p/score_y0 - 1 + 0.5 * pow(score_y0_p/score_y0 - 1, 2)
    else:
        score_y0_m = average(a = nbinom.pmf(y0-1, n, p) , weights = W) #minus 1
        score0 = score_y0_p/score_y0 - score_y0/score_y0_m + 0.5 * pow(score_y0_p/score_y0 - 1, 2)
    if y1 == 0:
        score1 = score_y1_p/score_y1 - 1 + 0.5 * pow(score_y1_p/score_y1 - 1, 2)
    else:
        score_y1_m = average(a = nbinom.pmf(y1-1, n, p) , weights = W) #minus 1
        score1 = score_y1_p/score_y1 - score_y1/score_y1_m + 0.5 * pow(score_y1_p/score_y1 - 1, 2)

    compositeHScore = score0 + score1
    simpleHScore = zeros(1)

    return {"simpleHScore" : simpleHScore , "compositeHScore" : compositeHScore}

modelx = SSM(name = "logistic diffusion model x", xdimension = 1, ydimension = 2, HScoreInference = HScoreInference, continuousObs = continuousObs)
modelx.setFirstStateGenerator(firstStateGenerator)
modelx.setObservationGenerator(observationGenerator)
modelx.setTransitionAndWeight(transitionAndWeight)
modelx.setComputeHscore(computeHscore)
# Values used to generate the synthetic dataset when needed:
# (untransformed parameters)
modelx.setParameters(array([0.05, 0.1, 1])) #put true params here to generate synthetic data, ow. COMMENT and choose a datafile in userfile.py 
modelx.addStateFiltering()
modelx.addStatePrediction()
modelx.addObsPrediction()


