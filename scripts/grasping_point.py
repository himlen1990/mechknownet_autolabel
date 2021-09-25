import numpy as np


a = np.array([1.,1.,0.])
b = np.array([-1.,-1.,0.])
o = np.array([2.,1.,0.])
def project_point(a, b, o):
    v_ab = b - a
    v_ao = o - a
    dis_ab_square = np.sum(v_ab**2)
    print dis_ab_square
    v_ap = np.sum(v_ab*v_ao)/(dis_ab_square)*v_ab

    p = a + v_ap
    print p


project_point(a,b,o)
