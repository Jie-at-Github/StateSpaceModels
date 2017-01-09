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
        array, float32, int32, zeros_like, newaxis, transpose, dot
from scipy.stats import norm, truncnorm, gamma
import scipy.weave as weave
import os
import math
from src.statespacemodel import SSM

################################################################
# Linear Gaussian Model: known autoregressive states with unknown linear observations
# M dimensional X = A * X + N(0, Sigma)
# scalar Y = b1 X1 + ... + bL (L<=M) + N(0, R) 
# X_0 ~ N(0, Sigma)
# sigma = I_M
# R = 0.5 #var, not sd
# A = I
# parameters[0, :] = b1
# ...
# parameters[L-1, :] = bL
################################################################
M = 4
#L = 3 #this can be L=1 (underfit) 3,4 (overfit), change the theta file only
#NOTE: this file is about true data generating model, and some helpful functions used in smc2 class  
rho = 0.0 #the identifiability is not guaranteed unless setting b1,b2>0, 
            #consider for example extreme casese rho = 0,1  
SIGMA = [[1, rho, rho, rho], [rho, 1, rho, rho], [rho, rho, 1, rho], [rho, rho, rho, 1]]
R = 0.5
#B = [3, 5]
Bfull = [1,2,0,0] #[3, 5, 0, 0] #use only to generate synthetic data 

#IMPORTANT: whether or not use H score for inference, i.e. to update the theta particle weights 
HScoreInference = 0
#for that purpose, I updated codes that involve
#logLike = log(mean(self.xweights)) 
#transitionAndWeight(...)
#essentially I added HScoreInference as a flag, and transitionAndWeight() an additional output 

#whether or not the observations are continuously valued (differ a lot in the computing of H score)
continuousObs = 1


### these functions take untransformed parameters as arguments
#### See src/models.py for explanations about the model functions.
def firstStateGenerator(parameters, size):
    #return random.normal(size = size, loc = 0, scale = 1)[:, newaxis] #new axis is for adding a new dimension 
    #transpose(random.multivariate_normal(repeat(0, nbparameters), proposalcovmatrix, size = currentvalue.shape[1]))
    #first_state = zeros((size, M))
    first_state = random.multivariate_normal(repeat(0, M), SIGMA, size = size) #dim: size x M
    return first_state  
def observationGenerator(states, parameters):
    return random.normal(size = states.shape[0], loc = dot(states, Bfull), scale = R)[:, newaxis]
    #the following funcion is a key: it sample the last states, and output new states & weights (filtering density)
    #under log score inference, the average weights is the unbiased loglik 
    #under H score inference, we use the updated weights and F/L identity to compute the Hscore of one obser 
    #evaluated at each theta particle 
def transitionAndWeight(states, y, parameters, t): #here, the last argument t seems useless throughout the software
    code = \
    """
    double tempmeasure1 = -0.9189385 - 0.5 * log(%(R)s);
    double tempmeasure2 = -0.5 / (%(R)s);
    double tempMean;
    for (int j = 0; j < Ntheta; j++)
    {
        for (int k = 0; k < Nx; k++)
        {
            for (int d = 0; d < Dx; d++)
            {
                states(k, d, j) =  states(k, d, j) + noise(k, j, d); 
            }
            tempMean = 0;
            for(int b = 0; b < Dparam; b++)
            {
               tempMean = tempMean + parameters(b, j) * states(k, b, j);
            }
            weights(k, j) = tempmeasure1 + tempmeasure2 * 
                ((double) y(0) - tempMean) * ((double) y(0) - tempMean);
        }
    } 
    double HScoreInferenceFlag = (double) %(HScoreInference)s;
    if ( HScoreInferenceFlag > 0 )
    {
        for (int j = 0; j < Ntheta; j++)
        {
            double tempMax = weights(0, j);
            double tempSum = 0;
            for (int k = 0; k < Nx; k++)
            {
                if (tempMax < weights(k, j))
                {
                    tempMax = weights(k, j);
                }
                tempSum = tempSum + weights(k, j);
            } 
            tempSum = 0;
            for (int k = 0; k < Nx; k++)
            {
                weightsNormalized(k, j) = exp( weights(k, j) - tempMax );
                tempSum = tempSum + exp( weights(k, j) - tempMax );
            }
            for (int k = 0; k < Nx; k++)
            {
                weightsNormalized(k, j) = weightsNormalized(k, j) / tempSum;
            }
            for (int k = 0; k < Nx; k++)
            {
                tempMean = - (double) y(0) / (%(R)s);
                for(int b = 0; b < Dparam; b++)
                {
                    tempMean = tempMean + parameters(b, j) * states(k, b, j) / (%(R)s);
                }
                grad(j) = grad(j) + weightsNormalized(k, j) * tempMean;
                lap(j) = lap(j) + weightsNormalized(k, j) * ( -1 / (%(R)s) + tempMean * tempMean );
            }
            simpleHScore(j) = lap(j) + (grad(j) * grad(j)) / 2;
        }
    }
    """ % {"R": R, "HScoreInference": HScoreInference} #the inline code first sample states and udpate weights, then normalize the weights, 
    #finally compute simple HScore for each theta 
    y = array([y]) #input to the c++ code must be array type 
    Nx = states.shape[0]
    Dx = states.shape[1] #state dimension, which is M here
    Dparam = parameters.shape[0] #dimension of the postulated model/parameter 
    Ntheta = states.shape[2]
    weights = zeros((Nx, Ntheta))
    weightsNormalized = zeros((Nx, Ntheta)) #normalized and exponentialed weights 
    grad = zeros(Ntheta)
    lap = zeros(Ntheta)
    simpleHScore = zeros(Ntheta)
    noise = random.multivariate_normal(repeat(0, M), SIGMA, (Nx, Ntheta)) #dim: Nx x Ntheta x M
    weave.inline(code,['Nx', 'Ntheta', 'Dx', 'Dparam', 'states', 'y', 'parameters', 'noise', 'weights', \
            'simpleHScore', 'grad', 'lap', 'weightsNormalized'], \
            type_converters=weave.converters.blitz, libraries = ["m"])
    return {"states": states , "weights": weights, "simpleHScore": simpleHScore} #the xweights are logged
    #the following function is to compute composite H score using current state, obser, theta particle, and weights
    #IMPORTANT note: weave.inline() should include the name of the variables you want to return 
def computeHscore(states, y, parameters, xweights, thetaweights, t):
    #xweights is of size Nx x Ntheta (unnormalized); thetaweights is also unnormalized 
    code = \
    """ 
    float tempMean;
    float ave_grad = 0;
    for (int i = 0; i < Ntheta; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            tempMean = - (double) y(0) / (%(R)s);
            for (int d = 0; d < Dparam; d++)
            {
                tempMean = tempMean + states(j, d, i) * parameters(d, i) / (%(R)s);
            }
            grad(i) = grad(i) + tempMean * xweights(j, i);
            lap(i) = lap(i) + ( -1 / (%(R)s) + tempMean * tempMean ) * xweights(j, i);
        }
        grad(i) = grad(i) / xNormConst(i);
        lap(i) = -grad(i) * grad(i) + lap(i) / xNormConst(i);            
        hScore_theta(i) = lap(i) + (grad(i) * grad(i)) / 2;
    }
    for (int i = 0; i < Ntheta; i++)
    {
        simpleHScore(0) = simpleHScore(0) + hScore_theta(i) * thetaweights(i);
        ave_grad = ave_grad + grad(i) * thetaweights(i);
    }        
    compositeHScore(0) = simpleHScore(0);
    for (int i = 0; i < Ntheta; i++)
    {
        compositeHScore(0) = compositeHScore(0) + 0.5 * (grad(i) - ave_grad) * (grad(i) - ave_grad) * thetaweights(i);
    }
    """ % {"R": R}
    y = array([y])
    Nx = states.shape[0]
    Ntheta = states.shape[2]
    Dparam = parameters.shape[0] #dimension of the postulated model/parameter
    thetaweights = thetaweights / sum(thetaweights)  #first transform to normalized ones      
    xNormConst = sum(xweights, axis=0)
    grad = zeros(Ntheta)  
    lap = zeros(Ntheta)
    hScore_theta = zeros(Ntheta) #store simple H score for each theta particle 
    compositeHScore = zeros(1)
    simpleHScore = zeros(1)
    weave.inline(code,['states', 'y', 'parameters', 'thetaweights', 'xweights', 'xNormConst', 'hScore_theta', \
            'Nx', 'Ntheta', 'Dparam', 'grad', 'lap', 'compositeHScore', 'simpleHScore'], \
            type_converters=weave.converters.blitz, libraries = ["m"])
    return {"simpleHScore" : simpleHScore , "compositeHScore" : compositeHScore}
#call the above: computeHscore(self.xparticles, self.observations[t], self.thetaparticles, xweights_safe, thetaweights)

modelx = SSM(name = "Nested LGSSM model x", xdimension = M, ydimension = 1, HScoreInference = HScoreInference, continuousObs = continuousObs)
modelx.setFirstStateGenerator(firstStateGenerator)
modelx.setObservationGenerator(observationGenerator)
modelx.setTransitionAndWeight(transitionAndWeight)
modelx.setComputeHscore(computeHscore)
# Values used to generate the synthetic dataset when needed:
# (untransformed parameters)
modelx.setParameters(array([3, 5])) #(array([3, 5])) #true params 
modelx.addStateFiltering()
modelx.addStatePrediction()
modelx.addObsPrediction()


