"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn
from scipy.integrate._ivp.common import warn_extraneous


class ForwardEuler(scipy.integrate.OdeSolver):
    """Solve an IVP using forward Euler method.
    A class which is derived from scipy.integrate.OdeSolver.
    """
    def __init__(self, fun, t0, y0, t_bound, vectorized, support_complex=False, h=None, **extraneous):
        """Need y_old store the previous solution y(t)
        """
        warn_extraneous(extraneous) #accept but warn extraneous arguments
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex) # inherit parameters/attributes of parent class

        if h is None:
            h = (t_bound - t0) / 100 #set to default step size
        self.h = h

        self.direction = +1 #the direction is always +1
        self.y_old = None 

    def _step_impl(self):
        """ Private method, propagates a solver one step further. Return tuple (success:Boolean, message:string).
        Use 'fun(self, t, y)' method for the system rhs evaluation.
        Nees y_old = y(t) and y_new = y(t+h)
        """
        self.y_old = self.y.copy()
        self.t_old = self.t
        f = self.fun(self.t, self.y) # RHS evaluation
        t_new = self.t + self.h #update time
        y_new = self.y + self.h * f # forward Euler
        
        if t_new > self.t_bound: #make sure t doesnt pass the bound
            t_new = self.t_bound #set t to the bound
            y_new = self.y + (t_new - self.t) * f 

        self.t = t_new #update new t
        self.y = y_new #update new y
        return (True, None)    

    def _dense_output_impl(self):
        """Return a `DenseOutput` object covering the last successful step
        """
        return EulerDenseOutput(self.t_old, self.t, self.y_old, self.y)

class EulerDenseOutput(DenseOutput):
    """Linear dense output for Forward Euler.
    """
    def __init__(self, t_old, t, y_old, y_new):
        super().__init__(t_old, t) #inherit parent class parameters
        self.y_old = y_old #y(t)
        self.y_new = y_new # y(t+h)

    def _call_impl(self, t):
        """estimating the solution between y_old and y_new with linear slope
		y(t)=y_old+alpha(y_new−y_old)
        """
        alpha = (t - self.t_old) / (self.t - self.t_old) #estimate the slope
        return self.y_old + alpha * (self.y_new - self.y_old)
    
    