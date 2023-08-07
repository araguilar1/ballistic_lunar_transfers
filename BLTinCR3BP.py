import numpy as np
from numpy import pi, sin, cos
import plotly.graph_objects as go
import plotly
import plotly.io as pio

import asset_asrl as ast
from asset_asrl.Astro import Constants as c
from asset_asrl.Astro.AstroModels import CR3BP
import astro

pio.templates.default = "plotly_dark"

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

# Plotting functions
def makeCircle(x, y, r):
    t = np.linspace(0, 2*np.pi, 1000)
    return x + r*np.cos(t), y + r*np.sin(t)

# periapse event
def periapseEvent():
    R, V = Args(7).tolist([(0,3),(3,3)])
    return R.dot(V)
stop_code = True
direction = -1
periapse_event = [(periapseEvent(), direction, stop_code)]

# x-crossing event
def xCrossingEvent():
    R,V = Args(7).tolist([(0,3),(3,3)])
    return R[1]
stop_code = True
direction = 1
x_crossing_event = [(xCrossingEvent(), direction, stop_code)]

def DVObj():
   v_init, v_final = Args(6).tolist([(0,3),(3,3)])
   dv = vf.sqrt(v_init.squared_norm() + vf.squared_norm() - 2*v_init.norm()*v_final.norm()*vf.cos(0))
   return dv

def solveVf(x0, v_init, v_final, x_points):
    new_x0s = []
    
    ocp = oc.OptimalControlProblem

    
ode = CR3BP(c.MuSun, c.MuEarth, c.AU)

integ = ode.integrator(0.0001)

x0 = np.zeros((7))

# 150 km circular parking orbit, centered about earth
x0[0] = 6528.137e3 / ode.lstar + (1-ode.mu)
x0[4] = 7.814015311e3 * ode.tstar / ode.lstar
tf = 2 * 86400 / ode.tstar 

# integrate
parking_orbit, _ = integ.integrate_dense(x0, tf, 10000, x_crossing_event)

parking_orbit_plot = np.array(parking_orbit).T

# Earth radius in non-dimensional units
earth_radius = 6378.137e3 / ode.lstar
earth_cirlce_plot = makeCircle(1-ode.mu, 0, earth_radius)

# moon orbit radius
moon_radius = c.LD / ode.lstar
moon_cirlce_plot = makeCircle(1-ode.mu, 0, moon_radius)

# create a list of initial conditions based on n evenly spaced points on the parking orbit
n_points = 5000
x_points = np.zeros((n_points, 7))
step = int(np.floor(np.shape(parking_orbit)[0] / n_points))
for i in range(n_points):
    x_points[i, 0:6] = parking_orbit[i * step][0:6]

# Find what the jacobi constants of the lagrange points are
lagrange_points = astro.lagrange_points(ode.mu)
# for i in range(len(lagrange_points)):
#     lp = np.append(lagrange_points[i], np.zeros(4))
#     u_star_lag_i = astro.jacobi(lp, ode.mu)
    # print(f'Jacobi Constant for L{i+1}: {u_star_lag_i}')

# Find the jacobi constant of the parking orbit
# parking_orbit_jacobi = astro.jacobi(x0, ode.mu)

# We need to escape L1 and L2 portals, but not L3
jacobi_map = 3.000804

# create new x0s
for i in range(n_points):
    v_i_u = x_points[i, 3:6] / np.linalg.norm(x_points[i, 3:6])
    v_f = np.sqrt(2*astro.uStar(x_points[i],ode.mu) - jacobi_map)

    v_f_vec = v_f * v_i_u

    x_points[i, 3:6] = v_f_vec
    # print(f'Error in Jacobi: {astro.jacobi(x_points[i], ode.mu) - jacobi_map}')


# integrate new x0s
tf = 365*86400 / ode.tstar # 1 year
periapses = []
for i in range(n_points):
    _, periapse = integ.integrate(x_points[i], tf, periapse_event)
    periapses.append(periapse)

peri_x = []
peri_y = []

for i in range(len(periapses)):
    for j in range(len(periapses[i][0])):
        peri_x.append(periapses[i][0][j][0])
        peri_y.append(periapses[i][0][j][1])

# plot
# plot earth first
fig = go.Figure(data=go.Scattergl(x=earth_cirlce_plot[0], y=earth_cirlce_plot[1], mode='lines', line=dict(color='green', width=2), name='Earth', fill='toself'))
fig.add_trace(go.Scattergl(x=moon_cirlce_plot[0], y=moon_cirlce_plot[1], mode='lines', line=dict(color='lightslategrey', width=2), name='Moon'))
fig.add_trace(go.Scattergl(x=[lagrange_points[0][0]], y=[lagrange_points[0][1]], mode='markers', marker=dict(size=5, color='yellow'), name='L1'))
fig.add_trace(go.Scattergl(x=[lagrange_points[1][0]], y=[lagrange_points[1][1]], mode='markers', marker=dict(size=5, color='orange'), name='L2'))
fig.add_trace(go.Scattergl(x=parking_orbit_plot[0], y=parking_orbit_plot[1], mode='lines', line=dict(color='lightseagreen', width=2), name='Parking Orbit'))
fig.add_trace(go.Scattergl(x=peri_x, y=peri_y, mode='markers', marker=dict(size=5, color='powderblue'), name='Periapses'))
fig.update_layout(
    xaxis_title="x [nd]",
    yaxis_title="y [nd]",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
fig.show()


