class Distribution():
    """
    Define a continuous distribution, with, at minimum methods f(x) and F(x).
    
    A Distribution can be defined in a variety of ways, using subclasses, for 
    example: Scipy, OpenTURNS or custom function. Some distributions may be 
    stored in a library (dictionary?), a summary of which can be printed by 
    the Distribution class via a static method.
    
    An instance of Distribution can be defined in several ways:
      1. DistScipy('logn', mu, sigma)
          this can alternatively be done via parameters, as follows:
            DistScipy('logn', mu, sigma)
      2. Distribution('logn', mu, sigma, source='scipy')
          this way automatically uses the subclass DistScipy (i.e., #1)
      3. custom distribution (manually write PDF, CDF)
      4. empirical distribution
          - use Dataset class to import
          - can also use this for Monte Carlo samples, also using Dataset
      5+. others: non-parametric, mixed, etc (lowest priority)
    
    
    Use the same plotting functions as the rest of the bivariate package!
    """
    def __init__(self):
        """
        Using methods in this class and in subclass, the following is required
        mean
        variance
        range = [lower, upper]: range over which the random variable is defined
        range_plot = [lower, upper]: 
            plotting range (differs from above when bounds are infinied)
            default: define based on probability in tail, 
                e.g., 1e-3*np.array([1, 1])
            
        perhaps a few other things are also required to help other parts of 
        the package function, for example descriptors about how the 
        Distribution was defined, number of parameters, etc...perhaps some of 
        this can be stored in a dictionary?
        
        parhaps also a tolerance for stopping calculations in the tails?
        """
        pass
    
    def pdf(self, x):
        """
        Must be able to take np.array as in/out
        No other in/out allowed (but can )
        """
        f = []
        return f
    
    def cdf(self, x):
        """
        Must be able to take np.array as in/out
        """
        F = []
        return f
    
    def pdf_inv(self, f):
        """
        Must be able to take np.array as in/out
        """
        x = []
        return x
    
    def cdf_inv(self, F):
        """
        Must be able to take np.array as in/out
        """
        x = []
        return x
    
    