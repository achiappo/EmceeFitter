# EmceeFitter
## Function optimisation interface via MCMC sampling and Chi-squared minimisation

Simple module containing a Python 3 interface to perform function optimisation by minimising the Chi-squared, given some observational data.  

The Chi-squared function is sampled using the Affine Invariant MCMC sampler contained in the **emcee** package.  
Subsequent minimisation of the Chi-squared over all possible samples returns the _least squares_ combination of parameters.  

### Characteristics  

#### `Fitter` base class 
Meant for  
- initialising default parameters for the **emcee** sampler  
- defining the (log)prior density distribution function (default='*uniform*')  

Input arguments  
- `ranges`  : dictionary consisting of parameter names as keys, and tuple of numerical ranges allowed as items  
- `priors`  : categorical name of prior density distribution function (default=*uniform*, available *log*)
- `walkers` : number of walkers in the Affine Invariant sampler (default=100)  
- `threads` : number of threads to open to parallelise the computations (default=None) 
- `burnin`  : number of burn-in steps to perform (default=None)
- `steps`   : number of steps to perform in the Affine Invariant sampling

#### `EmceeChi2Fitter` main class 
Meant to perform the function optimisation  
The instance is callable performing the *Affine Invariant* sampling of the Chi-squared.  
Input arguments  
- `function` : function to fit  
- `xobs`     : abscissa of the observational points  
- `yobs`     : ordinate of the observational points  
- `errors`   : uncertainties in the observational values

Output values  
- `values`   : *least squares* parameters array  
- `samples`  : array containing all parameters values probed  
- `lnprobs`  : array containing all Chi-squared values corresponding to `samples` coordinates  
