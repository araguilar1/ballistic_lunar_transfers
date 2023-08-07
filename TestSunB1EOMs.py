import asset_asrl as ast
import asset_asrl.Astro.Constants as c
from BCR4BPModel import BCR4BPSB1
from asset_asrl.Astro.AstroModels import CR3BP

import numpy as np
import matplotlib.pyplot as plt

oc = ast.OptimalControl
vf = ast.VectorFunctions
phaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes

Args = vf.Arguments

# System parameters from McCarthy and Initial Theta from Boudad
LEM  = 384747.992e3
LSB1 = 149597894.0e3
THETA0 = 0

ratio = LEM/LSB1

def makeCircle(x,y,r):
    t = np.linspace(0, 2*np.pi, 100)
    return x+r*np.cos(t), y+r*np.sin(t)

x0 = np.zeros((8))

# Test Orbit ICs from Boudad
x0[0] = 1.00745
x0[1] = -0.00149
x0[2] = -0.00278
x0[3] = -0.00045
x0[4] = 0.01238
x0[5] = -0.0023
x0[6] = THETA0

pp1 = np.zeros((8))
pp1[0:6] = x0[0:6]
pp1[6]   = 38.22
pp1[7]   = 3.086

ode = BCR4BPSB1(c.MuEarth, c.MuMoon, c.MuSun, LEM, LSB1, THETA0)
print(ode.thetadot)

# Integration Time
tf = 179.4*86400/ode.tstarSB1

# Initialize Integrator
integ = ode.integrator(0.0001)

direction = 0
stopcode = True
def YCrossing():
    R,V,theta = Args(7).tolist([(0,3),(3,3),(6,1)])
    return R[1]
YCrossingEvent = (YCrossing(), direction, stopcode)
Events = [YCrossingEvent]
# Integrate
traj, EventLocs = integ.integrate_dense(x0, tf, 10000, Events)

# def makePeriodic(ig, tf, pp, ode, integ, fixInit=[0,1,2,3,4,5,6,7]):
    
#     steps = 10000
#     trajGuess = integ.integrate_dense(ig, tf, steps)

#     odePhase = ode.phase(tModes.LGL3)

#     nSeg = 150
#     odePhase.setTraj(trajGuess, nSeg)

#     for idx in fixInit:
#         odePhase.addBoundaryValue(phaseRegs.Front, [idx], [trajGuess[0][idx]])

#     odePhase.addBoundaryValue(phaseRegs.Back,  [0,1,2,3,4,5,6,7], pp[0:8])

#     tol=1e-14

#     odePhase.optimizer.set_EContol(tol)
#     odePhase.solve()

#     return odePhase.returnTraj()

# traj = makePeriodic(x0, tf, pp1, ode, integ)


# Plot Trajectory in 3D
traj = np.array(traj).T
plt.figure()
#3D Plot
ax = plt.axes(projection='3d')
ax.plot3D(traj[0]*ode.lstarSB1, traj[1]*ode.lstarSB1, traj[2]*ode.lstarSB1, 'r', label='12:73 L2 Halo Orbit')
x, y = makeCircle((1-ode.muprime)*ode.lstarSB1, 0, ratio*ode.lstarSB1)
ax.scatter3D((1-ode.muprime)*ode.lstarSB1, 0, 0, color='green', marker='o', label='Earth')
ax.plot3D(x, y, 0, 'k', label='Lunar Orbit')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()


# Plot Theta
plt.figure()
plt.plot(traj[7], traj[6], 'b', label='Theta')
ax = plt.gca()
ax.grid()
ax.legend()
ax.set_xlabel('Time [ND]')
ax.set_ylabel('Theta [rad]')

plt.show()
