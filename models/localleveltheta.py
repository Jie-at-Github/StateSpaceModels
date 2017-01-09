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

from numpy import random, power, sqrt, exp, zeros_like, zeros, \
        ones, mean, average, prod, log
from scipy.stats import norm, truncnorm, gamma
from src.parametermodel import ParameterModel
import scipy.special as sps

#### See src/models.py for explanations about the model functions.
#### most functions here take transformed parameters, except rprior

## Prior hyperparameters
hyperparameters = { \
        "sigma_shape": 0.03, \
        "sigma_scale": 1, \
        "tau_shape": 0.03, \
        "tau_scale": 1}

def invgamma_logpdf(logx, shape, scale):
    return (-shape - 1) * logx - scale / exp(logx) \
            + shape * log(scale) - log(sps.gamma(scale)) 
    #I added the second line which depends on the hyperparameter, and which is necessary for computing the score

def logdprior(parameters, hyperparameters): #this function computes p(transformed_parameter) (measure on the trans-space)
    """ Takes transformed parameters.  When the parameter is transformed, 
    a jacobian appears in the formula.
    """
    sigma_part = parameters[0] + invgamma_logpdf(parameters[0], hyperparameters["sigma_shape"], hyperparameters["sigma_scale"])
    tau_part = parameters[1] + invgamma_logpdf(parameters[1], hyperparameters["tau_shape"], hyperparameters["tau_scale"])
    return sigma_part + tau_part

def rprior(size, hyperparameters):
    """ returns untransformed parameters """
    sigma = 1 / gamma.rvs(hyperparameters["sigma_shape"], scale = hyperparameters["sigma_scale"], size = size)
    tau = 1 / gamma.rvs(hyperparameters["tau_shape"], scale = hyperparameters["tau_scale"], size = size)
    parameters = zeros((2, size))
    parameters[0, :] = sigma
    parameters[1, :] = tau
    return parameters

modeltheta = ParameterModel(name = "Linear Gaussian model theta", dimension = 2)
modeltheta.setHyperparameters(hyperparameters)
modeltheta.setPriorlogdensity(logdprior)
modeltheta.setPriorgenerator(rprior)
modeltheta.setParameterNames(["expression(sigma^2)", "expression(tau^2)"])
modeltheta.setTransformation(["log", "log"])
InverseGammaTemplate = """
priorfunction <- function(x){
    shape <- %.5f 
    scale <- %.5f
    return(scale**shape / gamma(shape) * x**(- shape - 1) * exp(-scale / x))
}
"""
modeltheta.setRprior([InverseGammaTemplate % (hyperparameters["sigma_shape"], hyperparameters["sigma_scale"]), \
InverseGammaTemplate % (hyperparameters["tau_shape"], hyperparameters["tau_scale"])])


