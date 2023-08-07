import asset_asrl as ast
from asset_asrl.Astro import Constants as c
from BCR4BPModel import BCR4BP
from asset_asrl.Astro.AstroModels import CR3BP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

oc = ast.OptimalControl
vf = ast.VectorFunctions
Args = vf.Arguments
phaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes


# System Parameters from McCarthy and Inital Theta
LEM  = 384747.992e3 # m
LSB1 = 149597894.0e3 # m
THETA0 = 0

# x y z vx vy vz theta0 time
x0 = np.zeros((8))

# Test Orbit ICs from Scheuerle
x0[0] = 0.5
x0[1] = 0.1
x0[4] = 0.55
x0[6] = THETA0

# Initialize ODE
ode = BCR4BP(c.MuEarth, c.MuMoon, c.MuSun, LEM, LSB1, THETA0, 1)
integ = ode.integrator(0.0001)

tstar = ode.tstarEM
lstar = ode.lstarEM

tf = 5*86400/tstar
upLimTf = 7*86400/tstar
lowLimTf = 3*86400/tstar
time_step = 3600/tstar 

#target state is only pos
targ = np.zeros((3))
targ[0] = -0.1492
targ[1] = -0.3233
targ[2] = 0.0

def target(ode, ig, tf, targ, nsegs=250):

    integ = ode.integrator(0.0001)
    
    orbitIG = integ.integrate_dense(ig, tf, 10000)

    phase = ode.phase(tModes.LGL3)

    phase.setTraj(orbitIG, nsegs)

    # Enforce arrival at target state
    phase.addBoundaryValue(phaseRegs.Front, [0,1,2,6], [x0[0], x0[1], x0[2], x0[6]])
    phase.addBoundaryValue(phaseRegs.Back, [0,1,2], [targ[0], targ[1], targ[2]])
    
    tol=1e-14

    phase.optimizer.set_EContol(tol)

    phase.optimizer.PrintLevel = -1
    phase.solve() 

    return phase.returnTraj()


def constraintVector(xf, xd):
    return np.array([xf[0]-xd[0], xf[1]-xd[1], xf[2]-xd[2]])


def ftSingleShooting(ode, x0, tf, target, maxIter):
    # Fixed time single shooting

    x0 = np.copy(x0)
    integ = ode.integrator(0.0001)
    
    xf, jac = integ.integrate_stm(x0, tf)

    xF = xf[0:3]
    xd = target

    F = constraintVector(xf, xd)
    Fnorm = np.linalg.norm(F)

    tol = 1e-12
    i=0

    while Fnorm > tol:
        xF = xf[0:3]
        F = constraintVector(xF, xd)
        
        DF = np.reshape(np.array([jac[0,3], jac[0,4], jac[0,5], jac[1,3], jac[1,4], jac[1,5], jac[2,3], jac[2,4], jac[2,5]]), (3,3))
        dx = np.linalg.inv(DF).dot(F)

        x0[3] -= dx[0]
        x0[4] -= dx[1]
        x0[5] -= dx[2]

        Fnorm = np.linalg.norm(F)
        
        xf, jac = integ.integrate_stm(x0, tf)
        i+=1
        if i > maxIter:
            print('Targeter could not converge in ', maxIter, ' iterations')
            break

    targetTraj = integ.integrate_dense(x0, tf, 10000)

    return targetTraj

def transferContinuation(ode, x0, tf, target, tStep, upLim, lowLim, maxIter):
    # Varible time single shooting
    trajList = []
    tf0 = tf
    initial_transfer = ftSingleShooting(ode, x0, tf, target, maxIter)

    # Start with lower limit
    sign = np.sign(lowLim - initial_transfer[-1][7])
    signLast = sign
    tF = tf0
    IG = initial_transfer[0]
    while sign == signLast:
        tF -= tStep
        try:
            transfer = ftSingleShooting(ode, IG, tF, target, maxIter)
            trajList.append(transfer)

            signLast = sign
            sign = np.sign(lowLim - transfer[-1][7])
            IG = transfer[0]
        except:
            continue

    # Now upper limit
    sign = np.sign(upLim - initial_transfer[-1][7])
    signLast = sign
    tF = tf0
    IG = initial_transfer[0]
    while sign == signLast:
        tF += tStep
        try:
            transfer = ftSingleShooting(ode, IG, tF, target, maxIter)
            trajList.append(transfer)

            signLast = sign
            sign = np.sign(upLim - transfer[-1][7])
            IG = transfer[0]
        except:
            continue

    return trajList

         


# Do targeting
trajIG = integ.integrate_dense(x0, tf, 10000)
trajTargAST = target(ode, x0, tf, targ)
trajTarg = ftSingleShooting(ode, x0, tf, targ, 100)

tList = transferContinuation(ode, x0, tf, targ, time_step, upLimTf, lowLimTf, 100)

# Compute Hamiltonian
HamEM = []
for i in range(0, len(trajIG)):
    HamEM.append(ode.HamiltonianEM(trajIG[i][0:6], trajIG[i][6]))
HamTarg = []
for i in range(0, len(trajTarg)):
    HamTarg.append(ode.HamiltonianEM(trajTarg[i][0:6], trajTarg[i][6]))

mu = ode.mu

vIG = np.array(trajIG[0][3:6])
vTarg = np.array(trajTarg[0][3:6])

dv = np.linalg.norm(vTarg - vIG)

print('Delta V: ', dv*lstar/tstar, 'm/s')

# Plot Trajectory
trajIG= np.array(trajIG).T
trajTarg = np.array(trajTarg).T
trajTargAST = np.array(trajTargAST).T
print(trajTargAST[-1,-1]*tstar/86400)
plt.figure()
plt.plot(-mu, 0, 'go', label='Earth')
plt.plot(1-mu, 0, 'ko', label='Moon')
plt.plot(trajIG[0], trajIG[1], 'r', label='Initial Guess')
plt.scatter(targ[0], targ[1], s=25, c='k', marker='x', label='Target')
plt.plot(trajTarg[0], trajTarg[1], 'b--', label='Single Shooting')
plt.plot(trajTargAST[0], trajTargAST[1], 'm-.', label='ASSET Solution')
ax = plt.gca()
ax.set_aspect('equal')
ax.grid()
ax.legend()
ax.set_xlabel('x [ND]')
ax.set_ylabel('y [ND]')

# Plot Family

plt.figure()
plt.plot(-mu, 0, 'go', label='Earth')
plt.plot(1-mu, 0, 'ko', label='Moon')
T = []
dv = []
for t in tList:
    vTarg = np.array(t[0][3:6])
    dv.append(np.linalg.norm(vTarg - vIG)*lstar/tstar)
    T.append(np.array(t).T)

norm = mpl.colors.Normalize(vmin=min(dv), vmax=max(dv))
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])
for i, t  in enumerate(T):
    plt.plot(t[0], t[1], c=cmap.to_rgba(dv[i]))
plt.colorbar(cmap, label='Delta V [m/s]')
plt.plot(trajIG[0], trajIG[1], 'r', label='Initial Guess')
plt.scatter(targ[0], targ[1], s=25, c='k', marker='x', label='Target')
plt.plot(trajTarg[0], trajTarg[1], 'k', label='Single Shooting')
plt.plot(trajTargAST[0], trajTargAST[1], 'm', label='ASSET Solution')
ax = plt.gca()
ax.set_aspect('equal')
ax.grid()
ax.legend()
ax.set_xlabel('x [ND]')
ax.set_ylabel('y [ND]')

# Plot Hamiltonian
plt.figure()
plt.plot(trajIG[7]*tstar/86400, HamEM, 'b', label='Trajectory Hamiltonian')
plt.plot(trajTarg[7]*tstar/86400, HamTarg, 'b--', label='Target Hamiltonian')
ax = plt.gca()
ax.grid()
ax.ticklabel_format(useOffset=False)
ax.legend()
ax.set_xlabel('Time Along Transfer [Days]')
ax.set_ylabel('Earth-Moon Hamiltonian [ndim]')

# Plot Theta
plt.figure()
plt.plot(trajIG[7], trajIG[6], 'r', label='Theta')
plt.plot(trajTarg[7], trajTarg[6], 'b--', label='Theta Target')
ax = plt.gca()
ax.grid()
ax.set_xlabel('t [ND]')
ax.set_ylabel('Theta [rad]')

plt.show()

