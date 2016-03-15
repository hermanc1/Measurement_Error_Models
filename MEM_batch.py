# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:26:36 2016

@author: herman
"""
# ================================= Preliminaries ============================
from pymc3 import *
import time
import theano.tensor as T
import numpy as np
import pandas as pd
import scipy.stats
import scipy

# for handling theano errors more accurately:
theano.config.exception_verbosity = 'high'  

# Parameter definition:
GAIN_ERROR = 0.2
PHASE_ERROR = 0.2
BIAS_ERROR_MEAN = 5
BIAS_ERROR_SD = BIAS_ERROR_MEAN*0.5
X_PRECISION = 0.1  # precision of calibrator at 95% confidence level
X_CONF = 1.96  # z = 1.96 is the 95% confidence level.
RUNS = 120

# ================================= Functions ===============================
# Function for sum(|y*-y*(X*)| / X*)
def Abs_error_x_star(params):
    [yg, yp, yb]=[params[0], params[1], params[2]]
    return np.sum((np.abs(obs.y_star-((1 + yg)*obs.x_star*np.cos(yp+obs.phi)
                   + yb)))/(obs.x_star))
                  

def Abs_error_x(params):
    [yg, yp, yb] = [params[0], params[1], params[2]]
    #y_hat=(1 + yg)*obs.x*np.cos(yp+obs.phi) + yb
    #return np.sqrt(np.sum((obs.y_star-y_hat)**2)/obs.y_star.size)/obs.y_star.mean(axis=0)*100
    return np.sum(np.abs((obs.y_star-((1 + yg)*obs.x*np.cos(yp+obs.phi) + yb)))
                  /obs.x_star)


def x_to_y_star(x, yg, yp, ybm, ybsd, phi):
    # generate individual bias errors for every point of x, according to the
    # normal distribution with parameters ybm and ybsd
    return ((1 + yg)*x*np.cos(yp+phi) + np.random.normal(loc=ybm, scale=ybsd,
            size=x.size))
            
def OLS_min(params):
    [yg, yp, yb]=[params[0], params[1], params[2]]
    return np.sum((obs.y_star-((1 + yg)*obs.x_star*np.cos(yp+obs.phi)
                   + yb))**2)

# ============================ RV Setup =====================================
try: # If the file of previous RUNS does exist: load results
    results = pd.read_csv('results.csv', index_col=0)
    start_i = np.where(results['X* gain'].isnull())[0][0]
    pass
except OSError: # When the file doesn't exist
    results = pd.DataFrame(index=np.arange(0, RUNS),
                           columns=('X* gain', 'X* phase', 'X* bias',
                                    'Bayes1 gain', 'Bayes1 phase', 'Bayes1 bias',
                                    'Bayes2 gain', 'Bayes2 phase', 'Bayes2 bias'
                                    'OLS_gain, OLS_phase, OLS_bias))
    start_i = 1

start_time = time.clock()
for i in range(start_i, RUNS):
    run_start_time = time.clock()
    np.random.seed = i

    testpoints = pd.Series(np.arange(5, 105, 10))  # Testpoints are the
# points at which readings are taken, i.e. where the calibrator
# is set to 5, 10, 15...
    phipoints_ones = pd.Series(np.ones_like(testpoints))
    x_star_arr = pd.concat([testpoints, testpoints, testpoints],
                           ignore_index=True)

    x_err = np.random.normal(1, X_PRECISION/X_CONF, size=x_star_arr.size)
    # Since the distribution is symmetric, the error term can by multiplied
    # into x or x_star
    x = x_star_arr*x_err
    phi_v = pd.concat([np.pi/3*phipoints_ones, np.pi/6*phipoints_ones,
                       0*phipoints_ones], ignore_index=True)
    y_star = x_to_y_star(x, GAIN_ERROR, PHASE_ERROR,
                         BIAS_ERROR_MEAN, BIAS_ERROR_SD, phi_v)
    
    obs = pd.DataFrame({'x_star':x_star_arr, 'phi':phi_v, 'y_star':y_star,
                        'x':x})
# ================================== Minimization methods ====================
                        
    res2 = scipy.optimize.minimize(Abs_error_x,
                                   np.array([0.0, 0.0, 0.0]),
                                   method='Powell')

    res = scipy.optimize.minimize(Abs_error_x_star,
                                  np.array([0.0, 0.0, 0.0]),
                                  method='Powell')
    
    res_OLS = scipy.optimize.minimize(OLS_min, np.array([0.0, 0.0, 0.0]),
                                  method='Powell')
                                  
    AbsX_star_err = [(res2.x[0] - res.x[0])/res2.x[0]*100,
                     (res2.x[1] - res.x[1])/res2.x[1]*100, (res2.x[2] -
                                                           res.x[2]) /
                     res2.x[2]*100]
    
    OLS_err = [(res2.x[0] - res_OLS.x[0])/res2.x[0]*100,
                     (res2.x[1] - res_OLS.x[1])/res2.x[1]*100, (res2.x[2] -
                                                           res_OLS.x[2]) /
                     res2.x[2]*100]
# =================================== Bayes1 =================================

    with Model(verbose=0) as model:

        # Definition of priors
        gain_error = Normal('gain_error', mu=res.x[0], sd=res.x[0]*0.25)
        phase_error = Normal('phase_error', mu=res.x[1], sd=res.x[1]*0.25)
        bias_error = Normal('bias_error', mu=res.x[2], sd=res.x[2]*0.25)
        sd_x = obs.x_star.as_matrix()*(X_PRECISION/(X_CONF))
        sd_y = obs.y_star.as_matrix()*(X_PRECISION/(X_CONF**2))
        shape = obs.x_star.size
        mu_x = obs.x_star

        x_prior = Normal('x_prior',
                         mu=mu_x,
                         sd=sd_x,
                         shape=shape,
                         testval=obs.x_star)

        mu_y = (1 + gain_error)*x_prior*T.cos(phase_error + obs.phi) + bias_error

    # Likelihood
        y_likelihood = StudentT('y_est',
                                nu=1,
                                mu=mu_y,
                                sd=sd_y,
                                shape=shape,
                                observed=obs.y_star)

        start={'gain_error': res.x[0], 'phase_error': res.x[1], 'bias_error': res.x[2], 'x_prior': obs.x_star}
        trace = sample(200000, step=NUTS(), start=start)

    hist = pd.DataFrame()
    Bayes1=np.array([np.nan, np.nan, np.nan])
    hist['index'] = pd.Series(np.histogram(trace.gain_error, bins=200)[1])
    hist['count'] = pd.Series(np.histogram(trace.gain_error, bins=200)[0])
    Bayes1[0] = hist['index'][hist.loc[hist['count']==hist['count'].max()].index[0]]

    hist['index'] = pd.Series(np.histogram(trace.phase_error, bins=200)[1])
    hist['count'] = pd.Series(np.histogram(trace.phase_error, bins=200)[0])
    Bayes1[1] = hist['index'][hist.loc[hist['count']==hist['count'].max()].index[0]]

    hist['index'] = pd.Series(np.histogram(trace.bias_error, bins=200)[1])
    hist['count'] = pd.Series(np.histogram(trace.bias_error, bins=200)[0])
    Bayes1[2] = hist['index'][hist.loc[hist['count']==hist['count'].max()].index[0]]
        
    Bayes1_err = [(res2.x[0] - Bayes1[0])/res2.x[0]*100,
              (res2.x[1] - Bayes1[1])/res2.x[1]*100, (res2.x[2] -
                                                      Bayes1[2]) /
              res2.x[2]*100]

# ================================= Bayes2 ===================================

    with Model(verbose=0) as model2:

        # Definition of priors
        gain_error2 = Normal('gain_error2', mu=res.x[0], sd=res.x[0]*0.25)
        phase_error2 = Normal('phase_error2', mu=res.x[1], sd=res.x[1]*0.25)
        bias_error2 = Normal('bias_error2', mu=res.x[2], sd=res.x[2]*0.25)
        SD_Y=obs.y_star.as_matrix()*(X_PRECISION/X_CONF**2)
        MU_Y = (1 + gain_error2)*obs.x_star*T.cos(phase_error2 + obs.phi) + bias_error2
        
        y_likelihood2 = Normal('y_est', mu=MU_Y, sd=SD_Y, observed=obs.y_star)
        
        
        #start2 = find_MAP(model=model2, fmin=scipy.optimize.fmin_powell)  
        start2={'gain_error2':res.x[0],'phase_error2':res.x[1],'bias_error2':res.x[2]}                    
        trace_naive = sample(50000, start=start2, step=NUTS(), progressbar=False)

# ======================== Processing of Results ===========================
    Bayes2=np.array([np.nan, np.nan, np.nan])
    hist['index'] = pd.Series(np.histogram(trace_naive.gain_error2, bins=200)[1])
    hist['count'] = pd.Series(np.histogram(trace_naive.gain_error2, bins=200)[0])
    Bayes2[0] = hist['index'][hist.loc[hist['count']==hist['count'].max()].index[0]]

    hist['index'] = pd.Series(np.histogram(trace_naive.phase_error2, bins=200)[1])
    hist['count'] = pd.Series(np.histogram(trace_naive.phase_error2, bins=200)[0])
    Bayes2[1] = hist['index'][hist.loc[hist['count']==hist['count'].max()].index[0]]

    hist['index'] = pd.Series(np.histogram(trace_naive.bias_error2, bins=200)[1])
    hist['count'] = pd.Series(np.histogram(trace_naive.bias_error2, bins=200)[0])
    Bayes2[2] = hist['index'][hist.loc[hist['count']==hist['count'].max()].index[0]]
        
    Bayes2_err = [(res2.x[0] - Bayes2[0])/res2.x[0]*100,
              (res2.x[1] - Bayes2[1])/res2.x[1]*100, (res2.x[2] -
                                                      Bayes2[2]) /
              res2.x[2]*100]    
    
    results.loc[i] = [AbsX_star_err[0], AbsX_star_err[1], AbsX_star_err[2],
                      Bayes1_err[0], Bayes1_err[1], Bayes1_err[2],
                      Bayes2_err[0], Bayes2_err[1], Bayes2_err[2],
                      OLS_err[0], OLS_err[1], OLS_err[2]]

    results.to_csv(path_or_buf='results.csv')
    run_duration = time.clock()-run_start_time
    total_duration = time.clock()-start_time
    print("\n Run", i," duration:", "{:.1f}".format(run_duration), "s, total:",
          "{:.1f}".format(total_duration), "s")
