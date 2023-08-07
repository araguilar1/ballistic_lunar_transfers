import numpy as np
import asset_asrl as ast
from astro import jacobi, lagrange_points
import matplotlib.pyplot as plt
import scipy

from asset_asrl.Astro import Constants as c
from BCR4BPModel import BCR4BP
from asset_asrl.Astro.AstroModels import CR3BP

oc = ast.OptimalControl
vf = ast.VectorFunctions
Args = vf.Arguments
phaseRegs = oc.PhaseRegionFlags

# System Parameters
LEM  = 384747.992e3 # m
LSB1 = 149597894.0e3 # m
THETA0 = 0.0


bcr4bp_ode = BCR4BP(c.MuEarth, c.MuMoon, c.MuSun, LEM, LSB1, THETA0, eta=1)
lstar_em = bcr4bp_ode.lstarEM
tstar_em = bcr4bp_ode.tstarEM
mu       = bcr4bp_ode.mu

cr3bp_ode = CR3BP(c.MuEarth, c.MuMoon, LEM)
# cr3bp_ode = BCR4BP(c.MuEarth, c.MuMoon,c.MuSun, LEM, LSB1, THETA0, eta=0)
lag_points = lagrange_points(mu)

t_syn = 6.79117413
t_syn_dim = t_syn*tstar_em/86400

# Solve for CR3BP orbits first
# L1 Lyaupunov
x0_lyapunov = np.zeros((7))
x0_lyapunov[0] = 0.8547835
x0_lyapunov[4] = -0.1336242
tf_lyapunov = 2.743

def lyapunovConstaints(rv):
    x = np.copy(rv)

    return np.array([x[1], x[3]]) # y = xDot = 0


def cr3bpXCrossing():
    R, _ = Args(7).tolist([(0,3), (3,3)])
    
    return R[1] # y = 0
direction=1
stop_code=True
cr3bp_x_crossing_event = [(cr3bpXCrossing(), direction, stop_code)]


def cr3bpSingleShooting(ode, x0, tf, mu, events, tol=1e-12):

    integ = ode.integrator(0.0001)
    integ.EventTol = tol
    integ.MaxEventIters = 12

    xF, jac, _ = integ.integrate_stm(x0, tf, events)

    tF = xF[-1]

    F = lyapunovConstaints(xF)

    Fnorm = np.linalg.norm(F)

    while Fnorm > tol:

        x = xF[0]
        y = xF[1]
        yDot = xF[4]
        d = np.sqrt((x+mu)**2 + y**2)
        r = np.sqrt((x-1+mu)**2 + y**2)

        xDotDot = -(1-mu)*(x+mu)/d**3 - mu*(x-1+mu)/r**3 + 2*yDot + x

        F = lyapunovConstaints(xF)
        Fnorm = np.linalg.norm(F)

        DF = np.reshape(np.array([jac[1,4], yDot, jac[3,4], xDotDot]), (2,2))
        dx = np.linalg.inv(DF).dot(F)

        x0[4] -= dx[0]
        tF    -= dx[1]

        
        xF, jac, _ = integ.integrate_stm(x0, tF, events)

    traj = integ.integrate_dense(x0, 2*tF, 10000)

    return traj


def pseudoArclengthContinuation(ode, x0, tf, mu, events=cr3bp_x_crossing_event, s=0.001, lim=0.9):
    trajList = []

    integ = ode.integrator(0.0001)

    # first solution
    x1, jac1, _ = integ.integrate_stm(x0, tf, events)
    tF = x1[-1]
    F = lyapunovConstaints(x1)

    x = x1[0]
    y = x1[1]
    yDot = x1[4]
    d = np.sqrt((x+mu)**2 + y**2)
    r = np.sqrt((x-1+mu)**2 + y**2)
    xDotDot = -(1-mu)*(x+mu)/d**3 - mu*(x-1+mu)/r**3 + 2*yDot + x

    
    free_variable_vector  = np.array([yDot, tF])
    return 'foo'


def lyapunovContinuation(ode, x0, tf, mu, events=cr3bp_x_crossing_event, dx=0.001, lim=0.9, direction=1):
    trajList = []

    # first orbit
    xF1 = cr3bpSingleShooting(ode, x0, tf, mu, events)
    trajList.append(xF1)
    sign = np.sign(lim-trajList[-1][0][0])
    signLast = sign
    while sign == signLast:
        IG  = trajList[-1][0]
        tIG = trajList[-1][-1][-1]

        IG[0] += dx
        try:
            traj = cr3bpSingleShooting(ode, IG, tIG, mu, events)

            trajList.append(traj)
            signLast = sign
            sign = np.sign(lim-trajList[-1][0][0])
        except:
            continue

    return trajList


def bcr4bpConstaintVector(rv):
    x = np.copy(rv)

    return np.array([x[1], x[3], x[5]]) # y = xDot = zDot = 0


def bcr4bpXCrossing():
    R, V, theta = Args(8).tolist([(0,3), (3,3), (6,1)])
    return R[1] # y = 0

    # R, V, theta = Args(8).tolist([(0,3), (3,3), (6,1)])
    # return theta + np.pi/2
direction=-1
stop_code=True
bcr4bp_x_crossing_event = [(bcr4bpXCrossing(), direction, stop_code)]

def bcr4bpThetaEvent():
    R, V, theta = Args(8).tolist([(0,3), (3,3), (6,1)])
    return theta + np.pi
direction=-1
stop_code=True
bcr4bp_theta_event = [(bcr4bpThetaEvent(), direction, stop_code)]

def bcr4bpSingleShooting(ode, x0, tf, events, tol=1e-12):
    integ = ode.integrator(0.0001)
    integ.EventTol = tol
    integ.MaxEventIters = 50

    # make period greater to ensure y-crossing happens, event will stop integrations
    # tF = tf*2
    # _, jac, event_locs = integ.integrate_stm(x0, tF, events) # intial guess
    
    xF, jac = integ.integrate_stm(x0,tf)

    F = bcr4bpConstaintVector(xF)
    Fnorm = np.linalg.norm(F)
    
    while Fnorm > tol:
        F = bcr4bpConstaintVector(xF)
        Fnorm = np.linalg.norm(F)
        DF = np.reshape(np.array([jac[1,0], jac[1,2], jac[1,4], jac[3,0], jac[3,2], jac[3,4], jac[5,0], jac[5,2], jac[5,4]]), (3,3))
        dx = np.linalg.inv(DF).dot(F)
        x0[0] -= dx[0]
        x0[2] -= dx[1]
        x0[4] -= dx[2]

        xF, jac = integ.integrate_stm(x0, tf)

    traj = integ.integrate_dense(x0, tf, 10000)
    return traj

# DEBUG
# test_traj_ic = np.zeros((8))
# test_traj_ic[0] = 0.9101834999999939
# test_traj_ic[4] = -0.46612388449301123
# test_traj_tf = 3.395527166894923

# cr3bp_integ = cr3bp_ode.integrator(0.0001)
# cr3bp_integ.setAbsTol(1e-12)
# cr3bp_integ.setRelTol(1e-12)
# cr3bp_integ.EventTol = 1e-12
# cr3bp_integ.MaxEventIters = 20
# cr3bp_traj, cr3bp_event_locs = cr3bp_integ.integrate_dense(test_traj_ic[0:7], test_traj_tf, cr3bp_x_crossing_event)

# bcr4bp_integ = bcr4bp_ode.integrator(0.0001)
# bcr4bp_integ.setAbsTol(1e-12)
# bcr4bp_integ.setRelTol(1e-12)
# bcr4bp_integ.EventTol = 1e-12
# bcr4bp_integ.MaxEventIters = 20
# traj, event_locs = bcr4bp_integ.integrate_dense(test_traj_ic, 20, bcr4bp_x_crossing_event)

# plt.figure()
# traj = np.array(traj).T
# sun_x = []
# sun_y = []
# for theta  in traj[6]:
#     sun_x.append(bcr4bp_ode.a4*np.cos(theta))
#     sun_y.append(bcr4bp_ode.a4*np.sin(theta))

# cr3bp_traj = np.array(cr3bp_traj).T

# # plt.plot(sun_x, sun_y, 'y', linewidth=2, label='Sun')
# # plt.scatter(-mu, 0, color='g', label='Earth')
# plt.scatter(1-mu, 0, color='k', label='Moon')
# plt.scatter(lag_points[0][0], lag_points[0][1], color='m', marker='>', label='L1')
# # plt.scatter(lag_points[1][0], lag_points[1][1], color='orange', marker='<', label='L2')
# plt.scatter(traj[0][0], traj[1][0], color='g', marker='x', label='Initial Condition')
# plt.plot(traj[0], traj[1], 'b-', label='BCR4BP Initial Guess')
# plt.plot(cr3bp_traj[0], cr3bp_traj[1], 'r--', label='CR3BP Initial Guess')
# plt.scatter(event_locs[0][0][0], event_locs[0][0][1], color='b', marker='x', label='BCR4BP X-Crossing Event')
# plt.scatter(cr3bp_event_locs[0][0][0], cr3bp_event_locs[0][0][1], color='r', marker='x', label='CR3BP X-Crossing Event')

# plt.xlabel('x [nd]')
# plt.ylabel('y [nd]')
# plt.grid()
# plt.legend(loc='lower right')
# plt.axis('equal')
# plt.show(block=True)

# END DEBUG


traj_list = lyapunovContinuation(cr3bp_ode, x0_lyapunov, tf_lyapunov, mu, dx=0.0001, lim=0.964)

p_list = []
j_list = []
r_list = []
for t in traj_list:
    p_list.append(t[-1][-1])
    j_list.append(jacobi(t[0],mu))
    r_list.append(t[-1][-1]/t_syn)

# find specific resonance ratios
r_list = np.array(r_list)

# 4:3 Resonance Ratio
four_to_three_idx = np.argmin(np.abs(r_list-0.75))

# 3:2 Resonance Ratio
three_to_two_idx = np.argmin(np.abs(r_list-0.6667))

# 2:1 Resonance Ratio
two_to_one_idx = np.argmin(np.abs(r_list-0.5))

two_to_one_candidate = traj_list[two_to_one_idx]
two_to_one_bcr4bp_ic = two_to_one_candidate[0].tolist()
# Ensure this IC is BCR4BP
two_to_one_bcr4bp_ic[6] = THETA0
two_to_one_bcr4bp_ic.append(0.0)


solved_lyapunov = bcr4bpSingleShooting(bcr4bp_ode, two_to_one_bcr4bp_ic, p_list[two_to_one_idx], bcr4bp_theta_event)

period = solved_lyapunov[-1][-1]
print('CR3BP Period Info')
print(f'Period {two_to_one_candidate[-1][-1]} [ndim], {two_to_one_candidate[-1][-1]*tstar_em/86400} [days]')
print(f'Synodic Resonance Ratio: {two_to_one_candidate[-1][-1]/t_syn} [ndim]')
print('-----------------------------------')
print('BCR4BP Period Info')
print(f'Period: {period} [ndim], {period*tstar_em/86400} [days]')
print(f'Synodic Resonance Ratio: {period/t_syn} [ndim]')

plt.figure()
solved_lyapunov = np.array(solved_lyapunov).T
plt.plot(solved_lyapunov[0], solved_lyapunov[1], label='Periodic L1 Lyapunov')
plt.scatter(solved_lyapunov[0][0], solved_lyapunov[1][0], color='g', marker='x', label='Initial Condition')
plt.scatter(solved_lyapunov[0][-1], solved_lyapunov[1][-1], color='r', marker='x', label='Final State')
plt.scatter(lag_points[0][0], lag_points[0][1], color='m', marker='x', label='L1')
plt.scatter(lag_points[1][0], lag_points[1][1], color='orange', marker='x', label='L2')
plt.scatter(1-mu, 0, color='k', label='Moon')
ax = plt.gca()
ax.grid()
ax.set_xlabel('x [nd]')
ax.set_ylabel('y [nd]')
ax.set_aspect('equal')
ax.legend(loc='upper right')


plt.figure()
for traj in traj_list:
    t = np.array(traj).T
    plt.plot(t[0], t[1], color='lightgrey')

four_to_three_traj = np.array(traj_list[four_to_three_idx]).T
three_to_two_traj = np.array(traj_list[three_to_two_idx]).T
two_to_one_traj = np.array(traj_list[two_to_one_idx]).T


plt.scatter(-mu, 0, color='g', label='Earth')
plt.scatter(1-mu, 0, color='k', label='Moon')
plt.scatter(lag_points[0][0], lag_points[0][1], color='m', marker='x', label='L1')
plt.scatter(lag_points[1][0], lag_points[1][1], color='orange', marker='x', label='L2')

plt.plot(four_to_three_traj[0], four_to_three_traj[1], 'r', linewidth=2, label='4:3 Resonance')
plt.plot(three_to_two_traj[0], three_to_two_traj[1], 'g', linewidth=2, label='3:2 Resonance')
plt.plot(two_to_one_traj[0], two_to_one_traj[1], 'b', linewidth=2, label='2:1 Resonance')

ax = plt.gca()
ax.grid()
ax.set_aspect('equal')
ax.set_xlabel('x [nd]')
ax.set_ylabel('y [nd]')
ax.legend()

plt.figure()
plt.plot(j_list, r_list)
plt.scatter(j_list[four_to_three_idx], r_list[four_to_three_idx], color='r', marker='o', label='4:3 Resonance')
plt.scatter(j_list[three_to_two_idx], r_list[three_to_two_idx], color='g', marker='o', label='3:2 Resonance')
plt.scatter(j_list[two_to_one_idx], r_list[two_to_one_idx], color='b', marker='o', label='2:1 Resonance')
ax = plt.gca()
ax.set_xlabel('Jacobi Constant [nd]')
ax.set_ylabel('Synodic Resonance Ratio [nd]')
ax.legend(loc='upper right')
ax.grid()

plt.show()


