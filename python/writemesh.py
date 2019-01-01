import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ

'''All meshes are 2 dimensional. The first dimension is the membrane potential v, in whatever
units you chose to deliver them. In order for these scripts to work you need to provide a two dimensional vector
field, in a format specified below. The vector field is determined by the model that you want to simulate. For
a neuronal model called adaptive exponetial and fire the first diemnsion is the membrane potential V, the second one
is an adaptation parameter. In conductance-based models the first diemnsion is again the membrane potential V,
the second parameter tracks the state of the conductance. By convention, we will construct a 2D mesh using
tuples (v,w), where v usually (but not necessarily) labels the membrane potential and w whatever second
variable is present.'''


def CreateIntegralCurveStep(forward, forward_jacobian, I, v_0, w_0, t_end, delta_t, params = None):
    '''Create a forward integrated integral curve. I is the input
    current. (v_0, w_0) the starting point. Start time is t = 0,
    t_end is the end time, delta_t is the step of the time mesh.
    The time mesh and the resulting curve is returned. Mainly for visualization
    purposes in conjunction with DisPlayIntegralCurve. Avoid otherwise as
    forward integration towards the spike may result in problems.'''
    t = np.arange(0,t_end, delta_t)
    y_0 = [v_0, w_0]
    curve = integ.odeint(forward, y_0, t, args = ([I,params], ), Dfun=forward_jacobian, atol = 1e-11, rtol = 1e-11 )
    return t, curve


def MopUp(t, curve,n, V_min, w_min, v_c, w_c):

    last = 0
    for i in range(len(t)):
        if curve[i,0] <= V_min or curve[i,1] <= w_min:
            last = i
            break
    t_tot = 0

    if n > 0:
        t_tot = (n-1)*t[-1]
    if last > 0:
        t_tot += t[last-1]
        return n, t_tot, curve[last-1,0], curve[last-1,1]
    else:
        return n, t_tot, curve[-1,0], curve[-1,1]

def DetermineNegativeTime(reverse, reverse_jacobian, I, t, v, w, V_min, w_min, t_max):
    ''' From a given (v,w), step back until either the line V = V_min,
    or the line  w= W_min is crossed. Do this in repeated steps on a time mesh t, which
    determines the resolution of how this is achieved. The step back is performed in repeated
    integrations over time mesh t. Step out if this is not achieved in a time t_max.
    Return the index of crossing point; the time taken; the crossing point v, w as a four tuple.'''
    v_c = v
    w_c = w

    n = 0
    n_max = t_max/t[-1]

    while v_c > V_min and w_c > w_min:
        y_0 = list([v_c, w_c])
        curve = integ.odeint(reverse, y_0, t, args = ([I],), Dfun=reverse_jacobian)
        v_c = curve[-1,0]
        w_c = curve[-1,1]

        n+=1
        if n > n_max:
            raise ValueError

    return MopUp(t, curve, n, V_min, w_min, v_c, w_c)


def testShowMaximumSpikeValueInstance1():
    ''' This test shows the problem with forward integration into the spike.'''
    v = V_min
    w = 0.0*w_max
    I = 800
    delta_t = 1e-5
    t_overshoot = 20.55
    DisplayIntegralCurve(I,v,w,t_overshoot,delta_t,CreateIntegralCurveStep)

def testShowReverseIntegration():
    '''Approximately the same curve as testShowMaximumSpikeValueInstance1, but stepping back
    from the spike.'''
    v = -20.0
    w = 6.71
    I = 800
    delta_t = 1e-5
    t_estimate = 27
    DisplayIntegralCurve(I,v,w,t_estimate,delta_t)




def ReduceCurve(curve,fact):
    '''Reduces a slice of the curve.  fact should be an integer, indicating by what factor the curve
    should be reduced.'''
    nold = len(curve[:,0])

    lv = np.array([curve[i,0] for i in range(0,nold,fact)])
    lw = np.array([curve[i,1] for i in range(0,nold,fact)])
    cret = np.array([lv,lw])

    return cret.transpose()

def WriteOutCurve(f,curve, forward = True):
    if forward:
        vs = curve[:,0]
        ws = curve[:,1]
    else:
        vs = reversed(curve[:,0])
        ws = reversed(curve[:,1])

    for el in vs:
        f.write("{:.12f}".format(el))
        f.write(' ')
    f.write('\n')
    for el in ws:
        f.write("{:.12f}".format(el))
        f.write(' ')
    f.write('\n')



def create_results_directories(path):
    meshname =  path + '/' + path + '.mesh'

    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    simdirname = path + '/' + 'simresults'
    try:
        os.makedirs(simdirname)
    except OSError:
        if not os.path.isdir(simdirname):
            raise

    visdirname = path + '/' + 'visresults'
    try:
        os.makedirs(visdirname)
    except OSError:
        if not os.path.isdir(visdirname):
            raise


    return meshname, simdirname, visdirname

def GenerateTestMesh():
    ''' Don't ever change these values. meshtest.py; relies on them.'''

    res     = 1.0
    delta_t = 1e-6
    fact    = int(res/delta_t)
    t       =  np.arange(0, 30., 1e-5)
    I       =  800
    v       = -20
    w_start =  0.0
    V_min   = -80.0
    t_max   = 100.

    w_end = w_start + 100*aexpdevelop.b
    n_w   = 20
    ws = np.linspace(w_start, w_end,n_w)
    f = open('aexp.mesh','w')
    f.write(str(w_start) + ' ' + str(w_end) + ' ' +  str(n_w))
    f.write('\n')
    f.write(str(res))
    f.write('\n')

    for w in ws:
        n, t_d, v_lim, w_lim = DetermineNegativeTime(aexpdevelop.AEIFrev, aexpdevelop.Jrev, I, t, v, w, V_min, aexpdevelop.w_min, t_max)
        t, curve = CreateIntegralCurveStep(aexpdevelop.AEIFrev, aexpdevelop.Jrev, I, v, w, t_d, delta_t)
        red = ReduceCurve(curve,fact)
        print (w, t_d, len(red[:,0]))
        WriteOutCurve(f,red,False)





if __name__ == "__main__":
    print( 'Only for module use.')
