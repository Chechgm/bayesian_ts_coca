##################################################################################
# This script contains relevant functions for the evaluation of the models using
# Pareto Smoothed Importance Sampling (PSIS), Leave Future Out (LFO)
# Cross Validation (CV).
#
# More information on PSIS-LFO-CV can be found in the paper:
#   Approximate leave-future-out cross-validation for Bayesian time series models
#   https://arxiv.org/pdf/1902.06281.pdf
#
# We use part of the code provided by the authors
##################################################################################

import numpy as np
import pickle 
import pystan
from numpy import genfromtxt
import time
import os
import argparse


def psislw(lw, Reff=1.0, overwrite_lw=False):
    """Pareto smoothed importance sampling (PSIS).
    Parameters
    ----------
    lw : ndarray
        Array of size n x m containing m sets of n log weights. It is also
        possible to provide one dimensional array of length n.
    Reff : scalar, optional
        relative MCMC efficiency ``N_eff / N``
    overwrite_lw : bool, optional
        If True, the input array `lw` is smoothed in-place, assuming the array
        is F-contiguous. By default, a new array is allocated.
    Returns
    -------
    lw_out : ndarray
        smoothed log weights
    kss : ndarray
        Pareto tail indices
    """
    if lw.ndim == 2:
        n, m = lw.shape
    elif lw.ndim == 1:
        n = len(lw)
        m = 1
    else:
        raise ValueError("Argument `lw` must be 1 or 2 dimensional.")
    if n <= 1:
        raise ValueError("More than one log-weight needed.")

    if overwrite_lw and lw.flags.f_contiguous:
        # in-place operation
        lw_out = lw
    else:
        # allocate new array for output
        lw_out = np.copy(lw, order='F')

    # allocate output array for kss
    kss = np.empty(m)

    # precalculate constants
    cutoff_ind = - int(np.ceil(min(0.2 * n, 3 * np.sqrt(n / Reff)))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)
    logn = np.log(n)
    k_min = 1/3

    # loop over sets of log weights
    for i, x in enumerate(lw_out.T if lw_out.ndim == 2 else lw_out[None, :]):
        # improve numerical accuracy
        x -= np.max(x)
        # sort the array
        x_sort_ind = np.argsort(x)
        # divide log weights into body and right tail
        xcutoff = max(
            x[x_sort_ind[cutoff_ind]],
            cutoffmin
        )
        expxcutoff = np.exp(xcutoff)
        tailinds, = np.where(x > xcutoff)
        x2 = x[tailinds]
        n2 = len(x2)
        if n2 <= 4:
            # not enough tail samples for gpdfitnew
            k = np.inf
        else:
            # order of tail samples
            x2si = np.argsort(x2)
            # fit generalized Pareto distribution to the right tail samples
            np.exp(x2, out=x2)
            x2 -= expxcutoff
            k, sigma = gpdfitnew(x2, sort=x2si)
        if k >= k_min and not np.isinf(k):
            # no smoothing if short tail or GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, n2)
            sti /= n2
            qq = gpinv(sti, k, sigma)
            qq += expxcutoff
            np.log(qq, out=qq)
            # place the smoothed tail into the output array
            x[tailinds[x2si]] = qq
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
        # renormalize weights
        x -= sumlogs(x)
        # store tail index k
        kss[i] = k

    # If the provided input array is one dimensional, return kss as scalar.
    if lw_out.ndim == 1:
        kss = kss[0]

    return lw_out, kss


def gpdfitnew(x, sort=True, sort_in_place=False, return_quadrature=False):
    """Estimate the paramaters for the Generalized Pareto Distribution (GPD)
    Returns empirical Bayes estimate for the parameters of the two-parameter
    generalized Parato distribution given the data.
    Parameters
    ----------
    x : ndarray
        One dimensional data array
    sort : bool or ndarray, optional
        If known in advance, one can provide an array of indices that would
        sort the input array `x`. If the input array is already sorted, provide
        False. If True (default behaviour), the array is sorted internally.
    sort_in_place : bool, optional
        If `sort` is True and `sort_in_place` is True, the array is sorted
        in-place (False by default).
    return_quadrature : bool, optional
        If True, quadrature points and weight `ks` and `w` of the marginal posterior distribution of k are also calculated and returned. False by
        default.
    Returns
    -------
    k, sigma : float
        estimated parameter values
    ks, w : ndarray
        Quadrature points and weights of the marginal posterior distribution
        of `k`. Returned only if `return_quadrature` is True.
    Notes
    -----
    This function returns a negative of Zhang and Stephens's k, because it is
    more common parameterisation.
    """
    if x.ndim != 1 or len(x) <= 1:
        raise ValueError("Invalid input array.")

    # check if x should be sorted
    if sort is True:
        if sort_in_place:
            x.sort()
            xsorted = True
        else:
            sort = np.argsort(x)
            xsorted = False
    elif sort is False:
        xsorted = True
    else:
        xsorted = False

    n = len(x)
    PRIOR = 3
    m = 30 + int(np.sqrt(n))

    bs = np.arange(1, m + 1, dtype=float)
    bs -= 0.5
    np.divide(m, bs, out=bs)
    np.sqrt(bs, out=bs)
    np.subtract(1, bs, out=bs)
    if xsorted:
        bs /= PRIOR * x[int(n/4 + 0.5) - 1]
        bs += 1 / x[-1]
    else:
        bs /= PRIOR * x[sort[int(n/4 + 0.5) - 1]]
        bs += 1 / x[sort[-1]]

    ks = np.negative(bs)
    temp = ks[:,None] * x
    np.log1p(temp, out=temp)
    np.mean(temp, axis=1, out=ks)

    L = bs / ks
    np.negative(L, out=L)
    np.log(L, out=L)
    L -= ks
    L -= 1
    L *= n

    temp = L - L[:,None]
    np.exp(temp, out=temp)
    w = np.sum(temp, axis=1)
    np.divide(1, w, out=w)

    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()

    # posterior mean for b
    b = np.sum(bs * w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    temp = (-b) * x
    np.log1p(temp, out=temp)
    k = np.mean(temp)
    if return_quadrature:
        np.negative(x, out=temp)
        temp = bs[:, None] * temp
        np.log1p(temp, out=temp)
        ks = np.mean(temp, axis=1)
    # estimate for sigma
    sigma = -k / b * n / (n - 0)
    # weakly informative prior for k
    a = 10
    k = k * n / (n+a) + a * 0.5 / (n+a)
    if return_quadrature:
        ks *= n / (n+a)
        ks += a * 0.5 / (n+a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma


def gpinv(p, k, sigma):
    """Inverse Generalised Pareto distribution function."""
    x = np.empty(p.shape)
    x.fill(np.nan)
    if sigma <= 0:
        return x
    ok = (p > 0) & (p < 1)
    if np.all(ok):
        if np.abs(k) < np.finfo(float).eps:
            np.negative(p, out=x)
            np.log1p(x, out=x)
            np.negative(x, out=x)
        else:
            np.negative(p, out=x)
            np.log1p(x, out=x)
            x *= -k
            np.expm1(x, out=x)
            x /= k
        x *= sigma
    else:
        if np.abs(k) < np.finfo(float).eps:
            # x[ok] = - np.log1p(-p[ok])
            temp = p[ok]
            np.negative(temp, out=temp)
            np.log1p(temp, out=temp)
            np.negative(temp, out=temp)
            x[ok] = temp
        else:
            # x[ok] = np.expm1(-k * np.log1p(-p[ok])) / k
            temp = p[ok]
            np.negative(temp, out=temp)
            np.log1p(temp, out=temp)
            temp *= -k
            np.expm1(temp, out=temp)
            temp /= k
            x[ok] = temp
        x *= sigma
        x[p == 0] = 0
        if k >= 0:
            x[p == 1] = np.inf
        else:
            x[p == 1] = -sigma / k
    return x


def sumlogs(x, axis=None, out=None):
    """Sum of vector where numbers are represented by their logarithms.
    Calculates ``np.log(np.sum(np.exp(x), axis=axis))`` in such a fashion that
    it works even when elements have large magnitude.
    """
    maxx = x.max(axis=axis, keepdims=True)
    xnorm = x - maxx
    np.exp(xnorm, out=xnorm)
    out = np.sum(xnorm, axis=axis, out=out)
    if isinstance(out, np.ndarray):
        np.log(out, out=out)
    else:
        out = np.log(out)
    out += np.squeeze(maxx)
    return out

def psis_k(fit, L, M):
    samples = fit.extract()
    log_lik = np.sum(samples["y_ll"], axis=2)[:, L:M] # The log-likelihood of each time period from L+1
    log_lik = np.sum(log_lik, axis=1)                 # Cumulative sum in order to estimate the raw importance weights

    lw = log_lik
    _, ks = psislw(lw) # Compute Pareto smoothed log weights given raw log weights

    return ks

def elpd(fit, L, M):
    samples = fit.extract()
    log_lik = np.sum(samples["y_ll"], axis=2)[:, L:M] # The log-likelihood of each time period from L+1
    log_lik = np.sum(log_lik, axis=1)              # Cumulative sum in order to estimate the raw importance weights

    lw    = log_lik
    lw, _ = psislw(lw) # compute Pareto smoothed log weights given raw log weights
    lw    += log_lik       # Log weights
    loos  = sumlogs(lw, axis=0)
    loo   = loos.sum()

    return loo

def psis_lfo_cv(model, data, L_0):
  data["L"]= L = L_0
  M = L + 2
  n = data["N_row"]-L_0-2 # n iterations. We iterate from L_0 to the end of our data, -2 because we are predicting one step further, and the first is the exact

  # Initialize the resulting lists
  loo  = []
  ks   = []
  re_i = []

  # Fit the initial model
  fit = model.sampling(data=data, iter=2000, warmup=1000, chains=4, algorithm="NUTS", seed=42, verbose=True,
                control={"adapt_delta":0.9, "max_treedepth":15})
  ks.append(psis_k(fit, L+1, M))
  loo.append(elpd(fit, L+1, M))

  print("{} future observation evaluated".format(M-1))

  for i in range(n):
    M += 1
    k = psis_k(fit, L+1, M)
    ks.append(k)

    if k > 0.7:
      data["L"] = L = M - 2
      re_i.append(L) # Keep track of the re estimations
      fit = model.sampling(data=data, iter=2000, warmup=1000, chains=4, algorithm="NUTS", seed=42, verbose=True,
                control={"adapt_delta":0.9, "max_treedepth":15})
      loo.append(elpd(fit, L+1, M))

    else:
      loo.append(elpd(fit, L+1, M))

    print("{} future observation evaluated".format(M-1))

  return {'loo':loo, 'ks':ks,'re_i': re_i} 




parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/formated_data/agregated.csv',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='./models/ar_1.stan',
                    help="Directory containing model def")

parser.add_argument('--results_folder', default='./results/',
                    help="Optional, name of the where the previous fit is")

parser.add_argument('--fit', default=None,
                    help="Optional, name of the where the previous fit is")  

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    data_dir = args.data_dir 
    modeldir = args.model_dir
    results_folder = args.results_folder
    
    with open(modeldir, 'r') as f:
        model_definition = f.read()
        
    

    
    sm = pystan.StanModel(model_code=model_definition)


    aggregated = genfromtxt(data_dir, delimiter=',')

    data = {
        "N_row": aggregated.shape[0],
        "N_col": aggregated.shape[1], 
        "y": aggregated, 
        "L": 14
    }
    

    t0 = time.time()
    
    dictresults = psis_lfo_cv(sm, data, 14)
    
    t1 = time.time()
    
    total = t1-t0
    
    modelnn = modeldir.split('/')[-1].split('.')[0]
    
    name = results_folder +'/'+ modelnn
    try:
        os.mkdir(name)
    except FileExistsError:
        pass
    timefile = name + '/' + 'time.txt'
    
    output_model = name + '/' + 'output.txt'
    
    pickledresults = name + '/' + 'dict.py'
    
# TODO rename it with the name of the model
    with open(timefile, 'w') as f:
        f.write('The run time was: {}'.format(str(total)))
        
#    with open(output_model , 'w') as f:
#        f.write(str(fit))
#    
    with open(pickledresults , 'wb') as f:
        pickle.dump(dictresults,f)
    
    