# EmceeFitter
## Function optimisation interface via Chi-squared MCMC sampling and minimisation

Simple module containing a Python 3 interface to perform function optimisation by minimising the Chi-squared, given some observational data.  

The Chi-squared function is sampled using the Affine Invariant MCMC sampler contained in the **emcee** package.  
Subsequent minimisation of the Chi-squared over all possible samples, returns the _least squares_ combination of parameters.  

### Characteristics  

`class Fitter` : base class meant for  
- initialising default parameters for the the **emcee** sampler  
- defining the (log)prior density distribution function (default='*uniform*')  
- initialise the initial parameters coordinate array for the sampling  
