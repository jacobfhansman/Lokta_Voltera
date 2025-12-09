# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:50:18 2022

@author: argen
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.integrate import odeint

"""
def f(r,t):
    a = initial_coeff[0]
    b = initial_coeff[1]
    c = initial_coeff[2]
    d = initial_coeff[3]
    prey = r[0]
    predator = r[1]
    dx = a*prey -b*prey*predator
    dy = c*prey*predator -d*predator
    
    return np.array([dx,dy])

def rk4(f, start, end, init, steps=100):
    
    time, timestep = np.linspace(start_time, end_time, steps, retstep='true')
    x = np.zeros([len(init),steps])
    x[:,0] = initial_condition
    
    for t in range(steps-1):
        k1 = f(x[:,t], time[t])
        k2 = f(x[:,t]+.5*timestep*k1, time[t]+.5*timestep)
        k3 = f(x[:,t]+.5*timestep*k2, time[t]+.5*timestep)
        k4 = f(x[:,t]+timestep*k3, time[t]+timestep)
        x[:,t+1] = x[:,t] + timestep/6. * (k1+2*k2+2*k3+k4)
        
    return time, x

initial_coeff = ([1,.5,.5,2])
a_0 = initial_coeff[0]
b_0 = initial_coeff[1]
c_0 = initial_coeff[2]
d_0 = initial_coeff[3]
   
    
start_time = 0
end_time = 60
num_steps = 10000
initial_condition = np.array([2,2])

t,solution = rk4(f,start_time,end_time,initial_condition,num_steps)

predator = solution[0]
prey = solution[1]

plt.plot(t,prey)
plt.plot(t,predator)
plt.show()
"""


def lotka(x, t, params):
    N, P = x
    alpha,beta = params 
    derivs = [N - N*P - (alpha * N**2), N*P - (beta*P)] 
    return derivs



# Parameters
Nmin = 1
Nmax = 100

Pmin = 1
Pmax = 100

Alphamin = .01
Betamin = .01

Alphamax = 10
Betamax = 10

N0 = 2
P0 = 2
alpha0 = 1
beta0 = .5

alpha = 1
beta = 0.5


params = [alpha, beta]
x0=[N0,P0]
maxt = 60
tstep = 0.01

# Initial function values
t = np.arange(0, maxt, tstep)
prey, predator = odeint(lotka, x0, t, args=(params,)).T
# odeint returne a shape (2000, 2) array, with the value for
# each population in [[n_preys, n_predators], ...]
# The .T at the end transponses the array, so now we get each population
# over time in each line of the resultint (2, 2000) array.

# Create a figure and an axis to plot in:
fig = plt.figure()
ax = fig.add_axes([0.10, 0.5, 0.8, 0.3])
prey_plot = ax.plot(t, prey, label="Rabbits")[0]
predator_plot = ax.plot(t, predator, label="Foxes")[0]

ax.set_xlabel("Time")
ax.set_ylabel("Population size")
ax.legend(loc="upper right")
ax.set_title('Rabbit & Foxes Static Model')
ax.grid(color="b", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax.set_ylim([0, np.max([prey, predator])])

# create a space in the figure to place the two sliders:
axcolor = 'lightgoldenrodyellow'
axis_N = fig.add_axes([0.10, 0.1, 0.8, 0.03], facecolor=axcolor)
axis_P = fig.add_axes([0.10, 0.15, 0.8, 0.03], facecolor=axcolor)
axis_Alpha = fig.add_axes([0.10, 0.20, 0.8, 0.03], facecolor=axcolor)
axis_Beta = fig.add_axes([0.10, 0.25, 0.8, 0.03], facecolor=axcolor)
# the first argument is the rectangle, with values in percentage of the figure
# size: [left, bottom, width, height]

# create each slider on its corresponding place:
slider_N = Slider(axis_N, 'N', Nmin, Nmax, valinit=N0, valstep=1)
slider_P = Slider(axis_P, 'P', Pmin, Pmax, valinit=P0, valstep=1)
slider_Alpha = Slider(axis_Alpha, 'Alpha', Alphamin, Alphamax, valinit=alpha0)
slider_Beta = Slider(axis_Beta, 'Beta', Betamin, Betamax, valinit=beta0)


def update(val):
    # retrieve the values from the sliders
    x = [slider_N.val, slider_P.val]
    newparams = [slider_Alpha.val, slider_Beta.val]
    # recalculate the function values
    prey, predator = odeint(lotka, x, t, args=(newparams,)).T
    # update the value on the graph
    prey_plot.set_ydata(prey)
    predator_plot.set_ydata(predator)
    # redraw the graph
    fig.canvas.draw_idle()
    ax.set_ylim([0, np.max([prey, predator])])

# set both sliders to call update when their value is changed:
slider_N.on_changed(update)
slider_P.on_changed(update)
slider_Alpha.on_changed(update)
slider_Beta.on_changed(update)


# create the reset button axis (where its drawn)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# and the button itself
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_N.reset()
    slider_P.reset()
    slider_Alpha.reset()
    slider_Beta.reset()

button.on_clicked(reset)

plt.show()
