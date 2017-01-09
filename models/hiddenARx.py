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
        array, float32, int32, zeros_like, newaxis
from scipy.stats import norm, truncnorm, gamma
import scipy.weave as weave
import os
import math
from src.statespacemodel import SSM

################################################################
# Linear Gaussian Model: hidden autoregressive process
# X = rho * X + sigma Epsilon
# Y = X + tau Eta
# X_0 ~ N(0, 1)
# parameters[0, :] = rho
# sigma = 0.5
# tau = 0.5
################################################################

SIGMA = 0.5
TAU = 1
#SIGMA2 = SIGMA * SIGMA
TAU2 = TAU * TAU
#rho = 0.9

### key parameters 
#whether or not use H score for inference, i.e. to update the theta particle weights 
HScoreInference = 0
#whether or not the observations are continuously valued (differ a lot in the computing of H score)
continuousObs = 1


### these functions take untransformed parameters as arguments
#### See src/models.py for explanations about the model functions.
def firstStateGenerator(parameters, size):
    return random.normal(size = size, loc = 0, scale = 1)[:, newaxis]
def observationGenerator(states, parameters):
    return random.normal(size = states.shape[0], loc = states[:, 0], scale = TAU)[:, newaxis]
def transitionAndWeight(states, y, parameters, t):
    code = \
    """
    float tempmeasure1;
    float tempmeasure2;
    float temptransition;
    for (int j = 0; j < Ntheta; j++)
    {
        tempmeasure1 = -0.9189385 - 0.5 * log(%(TAU2)s);
        tempmeasure2 = -0.5 / (%(TAU2)s);
        temptransition = %(SIGMA)s;
        for(int k = 0; k < Nx; k++)
        {
            states(k, 0, j) = parameters(0, j) * states(k, 0, j) + temptransition * noise(k, j);
            weights(k, j) = tempmeasure1 + 
            tempmeasure2 * ((double) y(0) - states(k, 0, j)) * ((double) y(0) - states(k, 0, j));
        }
    }
    """ % {"TAU2": TAU2, "SIGMA": SIGMA}
    y = array([y])
    Nx = states.shape[0]
    Ntheta = states.shape[2]
    weights = zeros((Nx, Ntheta))
    noise = random.normal(size = (Nx, Ntheta), loc = 0, scale = 1)
    weave.inline(code,['Nx', 'Ntheta', 'states', 'y', 'parameters', 'noise', 'weights'], \
            type_converters=weave.converters.blitz, libraries = ["m"])
    return {"states": states , "weights": weights}

def computeHscore(states, y, parameters, xweights, thetaweights, t):
    #xweights is of size Nx x Ntheta (unnormalized); thetaweights is also unnormalized 
    code = \
    """ 
    double temp;
    double ave_grad = 0;
    double R;
    for (int i = 0; i < Ntheta; i++)
    {
        for (int j = 0; j < Nx; j++)
        {
            temp = - 1 / (%(TAU2)s) * ((double) y(0) - states(j, 0, i));
            grad(i) = grad(i)  + temp * xweights(j, i);
            lap(i) = lap(i) + ( - 1 / (%(TAU2)s) + temp * temp ) * xweights(j, i);
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
    """ % {"TAU2": TAU2}
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


modelx = SSM(name = "Simplest model x", xdimension = 1, ydimension = 1, HScoreInference = HScoreInference, continuousObs = continuousObs)
modelx.setComputeHscore(computeHscore)
modelx.setFirstStateGenerator(firstStateGenerator)
modelx.setObservationGenerator(observationGenerator)
modelx.setTransitionAndWeight(transitionAndWeight)
# Values used to generate the synthetic dataset when needed:
# (untransformed parameters)
modelx.setParameters(array([0.9]))
modelx.addStateFiltering()
modelx.addStatePrediction()
modelx.addObsPrediction()


