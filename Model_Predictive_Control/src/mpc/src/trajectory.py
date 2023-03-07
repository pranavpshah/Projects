import numpy as np
from scipy.interpolate import CubicSpline

class SplineTraj():

    def __init__(self):
        pass

    def make_spline(self, waypoints, T0, Tf):

        x = waypoints[:,0]
        y = waypoints[:,1]
        t = np.linspace(T0, Tf, waypoints.shape[0])

        xspline = CubicSpline(t, x)
        yspline = CubicSpline(t, y)

        return xspline, yspline

    def get_state(t, xspline, yspline):
        x = xspline(t)
        x_dot = xspline(t,1)
        y = yspline(t)
        y_dot = yspline(t,1)
        
        theta = np.arctan(y_dot/x_dot)

        return np.array([x, y, theta])

    #TODO: find desired reference inputs
