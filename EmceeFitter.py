
__author__='Andrea Chiappo'
__email__='chiappo.andrea@gmail.com'

import numpy as np
from sys import float_info
from emcee import EnsembleSampler

class Fitter(object):
    """Base class for EmceeChi2Fitter"""
    def __init__(self, ranges, priors='uniform', scale='linear', **kwargs):
        super(Fitter, self).__init__()
        self.nWalkers = kwargs['walkers'] if 'walkers' in kwargs else 100
        self.nThreads = kwargs['threads'] if 'threads' in kwargs else None
        self.nBurnin = kwargs['burnin'] if 'burnin' in kwargs else None
        self.nSteps = kwargs['steps'] if 'steps' in kwargs else 1000
        self.Xargs = ranges.keys()
        self.nDim = len(ranges)
        self.ranges = ranges
        self.priors = priors
        self.scale = scale
        self._initial_parameters()

    def _initial_parameters(self):
        """ 
        initialise positions of random walkers 
        randomly sampled points from prior distribution
        """
        if self.priors=='uniform':
            pos0 = np.empty([self.nWalkers,self.nDim])
            for w in range(self.nWalkers):
                for p,par in enumerate(self.Xargs):
                    pL,pR = self.ranges[par]
                    p0 = np.random.uniform(low=pL,high=pR)
                    pos0[w,p] = p0
            self.pos0 = pos0
        else:
            raise Exception("only uniform priors implemented")

    def _uniform_prior(self, theta):
        for val,par in zip(theta, self.Xargs):
            pi, pf = self.ranges[par]
            if not pi < val < pf:
                return -np.inf
        return 0.0

    def _prior_choice(self, theta):
        if self.priors=='uniform':
            return self._uniform_prior(theta)
        else:
            raise Exception("only uniform priors implemented")

class EmceeChi2Fitter(Fitter):
    """
    function fitter minimising Chi-squared
    using MCMC parameter space sampling via emcee package

    input parameters
    - function      : function to fit
    - xobs          : abscissa of the observational points
    - yobs          : ordinate of the observational points
    - ranges        : dictionary with prior ranges on parameters
    - priors        : prior distribution of choice (for now, only uniform)
    - kwargs        : keyword arguments for setting emcee sampler
    """
    def __init__(self, function, xobs, yobs, errors=None, *args, **kwargs):
        super(EmceeChi2Fitter, self).__init__(*args, **kwargs)
        self.func = function
        self.errs = errors
        self.xobs = xobs
        self.yobs = yobs

    def lnLike(self, theta):
        if self.scale=="log":
            theta = np.power(10, theta)
        try:
            tmod = self.func(self.xobs, *theta)
            tobs = self.yobs
            terr = self.errs
            if terr is None:
                chi = (tobs-tmod)**2 / tmod
            else:
                chi = (tobs-tmod)**2 / terr**2
            LL = -sum(chi) / 2
        except:
            return -float_info.max
        if not np.isfinite(LL):
            return -float_info.max
        return LL

    def lnProb(self, theta):
        lnPrior = self._prior_choice(theta)
        if not np.isfinite(lnPrior):
            return -np.inf
        return lnPrior + self.lnLike(theta)

    def __call__(self, nw=None, nt=None, nb=None, ns=None):
        if nw is None:
            nw = self.nWalkers
        else:
            self.nWalkers = nw
            self._initial_parameters()
        if nt is None:
            nt = self.nThreads
        if nb is None:
            nb = self.nBurnin
        if ns is None:
            ns = self.nSteps
        
        # setup emcee sampler
        sampler = EnsembleSampler(nw, self.nDim, self.lnProb, threads=nt)

        if nb:
            # Run burn-in steps
            pos, prob, state = sampler.run_mcmc(self.pos0, nb)

            # Reset the chain to remove the burn-in samples
            sampler.reset()

            # from the final position in burn-in chain, sample for nsteps
            sampler.run_mcmc(pos, ns, rstate0=state)
        else:
            # sample for nsteps
            sampler.run_mcmc(self.pos0, ns)

        samples = sampler.flatchain
        lnprobs = sampler.flatlnprobability

        indxs = np.where(lnprobs>-float_info.max)[0]
        if self.scale=='linear':
            samples = samples[indxs]
        elif self.scale=='log':
            samples = np.power(10,samples[indxs])
        else:
            raise Exception("prior scale must be set")
        lnprobs = lnprobs[indxs]
        
        Xmin = max(lnprobs)
        indmin = np.where(lnprobs==Xmin)[0][0]

        vals = samples[indmin]

        return vals, samples, lnprobs
