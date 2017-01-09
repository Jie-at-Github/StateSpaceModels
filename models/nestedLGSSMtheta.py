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

#! /usr/bin/env pythonproposal
# -*- coding: utf-8 -*-

from numpy import random, power, sqrt, exp, zeros_like, zeros, \
        ones, mean, average, prod, log, array, repeat
from scipy.stats import norm, truncnorm, gamma
from src.parametermodel import ParameterModel

#### most functions here take transformed parameters, but rprior returns
#### untransformed parameters

hyperparameters = { \
        "b_mean": 0, "b_sd": sqrt(100), \
        "b1_rate": 2.0, \
        "low": -10.0, "high": 100.0}  
        
M = 4
#L can be changed L=1 (underfit) 3,4 (overfit),
L = 2 #when change L, consider also change modeltheta.setRprior({}) #which is used in smc2plots
        
def logdprior(parameters, hyperparameters):
    """ Takes transformed parameters.  When the parameter is transformed, 
    a jacobian appears in the formula.
    """
    # the following is the log density of Gaussian
    b_all = 0
    for ell in range(L):
        #if ell==0:
        #b_all += parameters[ell] - hyperparameters["b1_rate"] * exp(parameters[ell]) + log(hyperparameters["b1_rate"]) #expo prior 
        #else:
        #b_all += - 0.5 / (hyperparameters["b_sd"]**2) * (parameters[ell] - hyperparameters["b_mean"])**2 #gaussian prior
        b_all += parameters[ell] - log(hyperparameters["high"]) #uniform prior 
    return b_all

def rprior(size, hyperparameters):
    """ returns untransformed parameters """ 
    parameters = zeros((L, size))
    for ell in range(L):
        ##same as random.normal
        #if ell==0:
        parameters[ell, :] = random.exponential(scale = 1.0 / hyperparameters["b1_rate"], size = size)
        #else:
        #    parameters[ell, :] = norm.rvs(size = size, loc = hyperparameters["b_mean"], scale = hyperparameters["b_sd"]) 
    return parameters

## Prior hyperparameters
modeltheta = ParameterModel(name = "Nested LGSSM model theta", dimension = L)
modeltheta.setHyperparameters(hyperparameters)
modeltheta.setPriorlogdensity(logdprior)
modeltheta.setPriorgenerator(rprior)
modeltheta.setParameterNames(["expression(B)"])
#transType = ["log"]
transType = []
for m in range(L):
    transType.append("log")
modeltheta.setTransformation(transType) #(["none", "none"])
#modeltheta.setRprior({}) #seems useless
#modeltheta.setRprior(["priorfunction <- function(x) dexp(x, rate = %.5f)" % hyperparameters["b1_rate"],\
#    "priorfunction <- function(x) dnorm(x, sd = %.5f)" % hyperparameters["b_sd"]])
#modeltheta.setRprior(["priorfunction <- function(x) dexp(x, rate = %.5f)" % hyperparameters["b1_rate"],\
#    "priorfunction <- function(x) dexp(x, rate = %.5f)" % hyperparameters["b1_rate"]])
set_argu = []
for m in range(L):
    set_argu.append("priorfunction <- function(x) dexp(x, rate = %.5f)" % hyperparameters["b1_rate"])
modeltheta.setRprior(set_argu)

