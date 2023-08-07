import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.io as pio
import asset_asrl as ast
from asset_asrl.Astro import Constants as c
from asset_asrl.Astro.AstroModels import CR3BP
import astro

vf = ast.VectorFunctions
Args = vf.Arguments




pio.templates.default = "plotly_dark"


ode = CR3BP(c.MuEarth,c.MuMoon, c.LD)

x0 = np.zeros((7))
x0[0] = 1.17
x0[4] = -0.489780292125578

p_ndim = 3.042534324464009

tf = 50*p_ndim

integ = ode.integrator(0.0001)
traj = integ.integrate_dense(x0, tf, 10000)

traj = np.array(traj).T

lagrange_points_xy = astro.lagrange_points(ode.mu)
# plot x-y of traj (traj[0], traj[1]), set axes equal
fig = go.Figure(data=go.Scatter(x=traj[0], y=traj[1], mode='lines', line=dict(color='lightseagreen', width=2), name='DRO'))
# plot the earth and moon locations
fig.add_trace(go.Scatter(x=[-ode.mu], y=[0], mode='markers', marker=dict(size=20, color='dodgerblue'), name='Earth'))
fig.add_trace(go.Scatter(x=[1-ode.mu], y=[0], mode='markers', marker=dict(size=10, color='lightslategrey'), name='Moon'))
fig.add_trace(go.Scatter(x=[lagrange_points_xy[0][0]], y=[lagrange_points_xy[0][1]], mode='markers', marker=dict(size=5, color='magenta'), name='L1'))
fig.add_trace(go.Scatter(x=[lagrange_points_xy[1][0]], y=[lagrange_points_xy[1][1]], mode='markers', marker=dict(size=5, color='orange'), name='L2'))
fig.update_layout(
    xaxis_title="x",
    yaxis_title="y",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.show()

def xCrossingEvent():
    R,V = Args(7).tolist([(0,3),(3,3)])
    return R[1]
stop_code = False
direction = 0
x_crossing_event = [(xCrossingEvent(), direction, stop_code)]

_, event_locs = integ.integrate_dense(x0, tf, 10000, x_crossing_event)

# plot x, xdot 
x_event = []
xdot_event = []

for event in event_locs[0]:
    x_event.append(event[0])
    xdot_event.append(event[3])

# now make plotly scatter plot of x, xdot
fig = go.Figure(data=go.Scatter(x=x_event, y=xdot_event, mode='markers', marker=dict(size=5, color='lightseagreen'), name='x0 = 1.17'))
fig.update_layout(
    xaxis_title="x",
    yaxis_title="xdot",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

# perturb x direction by 0.01 3 times and re-integrate with events
x0[0] = 1.18
_, event_locs1 = integ.integrate_dense(x0, tf, 10000, x_crossing_event)
x0[0] = 1.19
_, event_locs2 = integ.integrate_dense(x0, tf, 10000, x_crossing_event)
x0[0] = 1.20
_, event_locs3 = integ.integrate_dense(x0, tf, 10000, x_crossing_event)

# plot x, xdot
x_event1 = []
xdot_event1 = []
x_event2 = []
xdot_event2 = []
x_event3 = []
xdot_event3 = []

for event in event_locs1[0]:
    x_event1.append(event[0])
    xdot_event1.append(event[3])

for event in event_locs2[0]:
    x_event2.append(event[0])
    xdot_event2.append(event[3])

for event in event_locs3[0]:
    x_event3.append(event[0])
    xdot_event3.append(event[3])

# now make plotly scatter plot of x, xdot
fig.add_trace(go.Scatter(x=x_event1, y=xdot_event1, mode='markers', marker=dict(size=5, color='red'), name='x0 = 1.18'))
fig.add_trace(go.Scatter(x=x_event2, y=xdot_event2, mode='markers', marker=dict(size=5, color='dodgerblue'), name='x0 = 1.19'))
fig.add_trace(go.Scatter(x=x_event3, y=xdot_event3, mode='markers', marker=dict(size=5, color='orange'), name='x0 = 1.20'))
fig.show()
