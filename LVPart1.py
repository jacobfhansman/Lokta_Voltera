#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun Oct 30 11:11:15 2022

Lokta-Volterra Project!

Jacob Hansman, Nick Argentieri

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.integrate import odeint
"""
Code adapted from: @berna1111's answer to https://stackoverflow.com/questions/43381449/how-to-make-two-sliders-in-matplotlib

"""


def lotka(x, t, params):
    N, P = x
    a, b, c, d = params
    """
    growth rate zero for steady states
    
    derivs = [0,0]
    """
    derivs = [a*N - b*N*P, c*N*P - d*P]
    return derivs



# Parameters
Preymin = 1
Preymax = 100
Predmin = 1
Predmax = 100
Amin = .01
Bmin = .01
Cmin = .01
Dmin = .01
Amax = 20
Bmax = 20
Cmax = 20
Dmax = 20
Prey_0 = 2
Pred_0 = 2
a0 = 1
b0 = .5
c0 = .5
d0 = 2
"""
steady state params:

a0 = 1
b0 = .1
c0 = .5
d0 = .02
"""
"""
steady state initial populations, growth rate 0

Prey_0 = c0/d0
Pred_0 = a0/b0
"""
a = 1
b = 0.5
c = 0.5
d = 2

params = [a, b, c, d]
x0=[Prey_0,Pred_0]
end_time = 60
tstep = 0.001

# Initial function values
t = np.arange(0, end_time, tstep)
prey, predator = odeint(lotka, x0, t, args=(params,)).T



fig = plt.figure()
ax = fig.add_axes([0.15, 0.5, 0.8, 0.3])
prey_plot = ax.plot(t, prey, label="Rabbits")[0]
predator_plot = ax.plot(t, predator, label="Foxes")[0]

ax.set_xlabel("Time")
ax.set_ylabel("Population size")
ax.legend(loc="upper right")
ax.set_title('Rabbit & Foxes Population Model')
ax.grid(color="b", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax.set_ylim([0, np.max([prey, predator])])


axcolor = 'lightgoldenrodyellow'
axis_Prey = fig.add_axes([0.4, 0.1, 0.45, 0.015], facecolor=axcolor)
axis_Pred = fig.add_axes([0.4, 0.15, 0.45, 0.015], facecolor=axcolor)
axis_A = fig.add_axes([0.4, 0.20, 0.45, 0.015], facecolor=axcolor)
axis_B = fig.add_axes([0.4, 0.25, 0.45, 0.015], facecolor=axcolor)
axis_C = fig.add_axes([0.4, 0.30, 0.45, 0.015], facecolor=axcolor)
axis_D = fig.add_axes([0.4, 0.35, 0.45, 0.015], facecolor=axcolor)

# size: [left, bottom, width, height]

# create each slider on its corresponding place:
slider_Prey = Slider(axis_Prey, 'Inital Prey Population', Preymin, Preymax, valinit=Prey_0, valstep=1)
slider_Pred = Slider(axis_Pred, 'Initial Predator Population', Predmin, Predmax, valinit=Pred_0, valstep=1)
slider_A = Slider(axis_A, 'Growth rate of rabbits', Amin, Amax, valinit=a0)
slider_B = Slider(axis_B, 'Death rate of rabbits by predation', Bmin, Bmax, valinit=b0)
slider_C = Slider(axis_C, 'Natural death rate of foxes', Cmin, Cmax, valinit=c0)
slider_D = Slider(axis_D, 'Rabbits eaten per fox birth', Dmin, Dmax, valinit=d0)

def update(val):

    x = [slider_Prey.val, slider_Pred.val]
    newparams = [slider_A.val, slider_B.val, slider_C.val, slider_D.val]
    # recalculate the function values
    prey, predator = odeint(lotka, x, t, args=(newparams,)).T
    # update the value on the graph
    prey_plot.set_ydata(prey)
    predator_plot.set_ydata(predator)
    # redraw the graph
    fig.canvas.draw_idle()
    ax.set_ylim([0, np.max([prey, predator])])

# set both sliders to call update when their value is changed:
slider_Prey.on_changed(update)
slider_Pred.on_changed(update)
slider_A.on_changed(update)
slider_B.on_changed(update)
slider_C.on_changed(update)
slider_D.on_changed(update)

# create the reset button axis (where its drawn)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# and the button itself
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_Prey.reset()
    slider_Pred.reset()
    slider_A.reset()
    slider_B.reset()
    slider_C.reset()
    slider_D.reset()

button.on_clicked(reset)

plt.show()


"""



#part 3:
    
    
def lotka(x, t, params):
    N, P = x
    alpha,beta = params 
    derivs = [N - N*P - (alpha * N**2), N*P - (beta*P)] 
    return derivs



# Parameters
Preymin = 1
Preymax = 100

Predmin = 1
Predmax = 100

Alphamin = 1
Betamin = .1

Alphamax = 10
Betamax = 10

Prey_0 = 2
Pred_0 = 2
alpha0 = 1
beta0 = .5

alpha = 1
beta = 0.5


params = [alpha, beta]
x0=[Prey_0,Pred_0]
maxt = 30
tstep = 0.01


t = np.arange(0, maxt, tstep)
prey, predator = odeint(lotka, x0, t, args=(params,)).T



fig = plt.figure()
ax = fig.add_axes([0.10, 0.5, 0.8, 0.45])
prey_plot = ax.plot(t, prey, label="Rabbits")[0]
predator_plot = ax.plot(t, predator, label="Foxes")[0]

ax.set_xlabel("Time")
ax.set_ylabel("Population size")
ax.legend(loc="upper right")
ax.set_title('Rabbit & Foxes Static Model')
ax.grid(color="b", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax.set_ylim([0, np.max([prey, predator])])


axcolor = 'lightgoldenrodyellow'
axis_Prey = fig.add_axes([0.10, 0.1, 0.8, 0.0225], facecolor=axcolor)
axis_Pred = fig.add_axes([0.10, 0.15, 0.8, 0.0225], facecolor=axcolor)
axis_Alpha = fig.add_axes([0.10, 0.20, 0.8, 0.0225], facecolor=axcolor)
axis_Beta = fig.add_axes([0.10, 0.25, 0.8, 0.0225], facecolor=axcolor)



slider_Prey = Slider(axis_Prey, 'N', Preymin, Preymax, valinit=Prey_0, valstep=1)
slider_Pred = Slider(axis_Pred, 'P', Predmin, Predmax, valinit=Pred_0, valstep=1)
slider_Alpha = Slider(axis_Alpha, 'Alpha', Alphamin, Alphamax, valinit=alpha0)
slider_Beta = Slider(axis_Beta, 'Beta', Betamin, Betamax, valinit=beta0)


def update(val):

    x = [slider_Prey.val, slider_Pred.val]
    newparams = [slider_Alpha.val, slider_Beta.val]

    prey, predator = odeint(lotka, x, t, args=(newparams,)).T

    prey_plot.set_ydata(prey)
    predator_plot.set_ydata(predator)

    fig.canvas.draw_idle()
    ax.set_ylim([0, np.max([prey, predator])])


slider_Prey.on_changed(update)
slider_Pred.on_changed(update)
slider_Alpha.on_changed(update)
slider_Beta.on_changed(update)



resetax = plt.axes([0.8, 0.025, 0.1, 0.04])

button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_Prey.reset()
    slider_Pred.reset()
    slider_Alpha.reset()
    slider_Beta.reset()

button.on_clicked(reset)

plt.show()
"""
