#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:55:39 2022

@author: firstborn
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgetas import Slider, Button, RadioButtons
from scipy.integrate import odeint
"""
Code adapted from: @berna1111's answer to https://stackoverflow.com/questions/43381449/how-to-make-two-sliders-in-matplotlib

"""


def lotka(x, t, params):
    N, P, Dr = x
    a, b, c = params 
    derivs = [N - N*P - a*N**2, N*P - Dr*P - b*P, P*Dr - c*Dr] 
    return derivs



# Parameters
Preymin = 1
Preymax = 100
Predmin = 1
Predmax = 100
Dragmin = 1
Dragmax = 100
Amin = .001
Bmin = .001
Cmin = .001
Amax = 5
Bmax = 5
Cmax = 5
Prey_0 = 5
Pred_0 = 5
Drag_0 = 2
a0 = .1
b0 = .5
c0 = .5
a = .3
b = 0.5
c = 0.5
params = [a, b, c]
x0=[Prey_0,Pred_0,Drag_0]
end_time = 60
tstep = 0.001

# Initial function values
t = np.arange(0, end_time, tstep)
prey, predator, dragons = odeint(lotka, x0, t, args=(params,)).T



fig = plt.figure()
ax = fig.add_axes([0.15, 0.45, 0.8, .4])
prey_plot = ax.plot(t, prey, label="Rabbits")[0]
predator_plot = ax.plot(t, predator, label="Foxes")[0]
dragon_plot = ax.plot(t,dragons,label='Dragons')[0]

ax.set_xlabel("Time")
ax.set_ylabel("Population size")
ax.legend(loc="upper right")
ax.set_title('Rabbits, Foxes, and Dragons, Oh my')
ax.grid(color="b", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax.set_ylim([0, np.max([prey, predator])])


axcolor = 'lightgoldenrodyellow'
axis_Prey = fig.add_axes([0.4, 0.074, 0.45, 0.025], facecolor=axcolor)
axis_Pred = fig.add_axes([0.4, 0.115, 0.45, 0.025], facecolor=axcolor)
axis_Drag = fig.add_axes([0.4, 0.16, 0.45, 0.025], facecolor=axcolor)
axis_A = fig.add_axes([0.4, 0.2, 0.45, 0.025], facecolor=axcolor)
axis_B = fig.add_axes([0.4, 0.25, 0.45, 0.025], facecolor=axcolor)
axis_C = fig.add_axes([0.4, 0.3, 0.45, 0.025], facecolor=axcolor)


# size: [left, bottom, width, height]

# create each slider on its corresponding place:
slider_Prey = Slider(axis_Prey, 'Inital Rabbit Population', Preymin, Preymax, valinit=Prey_0, valstep=1)
slider_Pred = Slider(axis_Pred, 'Initial Fox Population', Predmin, Predmax, valinit=Pred_0, valstep=1)
slider_Drag = Slider(axis_Drag, 'Initial Dragon Population',Dragmin,Dragmax, valinit=Drag_0,valstep =1)
slider_A = Slider(axis_A, 'Rabbit Food Pressure', Amin, Amax, valinit=a0)
slider_B = Slider(axis_B, 'Natural Death Rate of Foxes', Bmin, Bmax, valinit=b0)
slider_C = Slider(axis_C, 'Natural Death Rate of Dragons', Cmin, Cmax, valinit=c0)


def update(val):

    x = [slider_Prey.val, slider_Pred.val, slider_Drag.val]
    newparams = [slider_A.val, slider_B.val, slider_C.val]
    # recalculate the function values
    prey, predator, dragons = odeint(lotka, x, t, args=(newparams,)).T
    # update the value on the graph
    prey_plot.set_ydata(prey)
    predator_plot.set_ydata(predator)
    dragon_plot.set_ydata(dragons)
    # redraw the graph
    fig.canvas.draw_idle()
    ax.set_ylim([0, np.max([prey, predator])])

# set both sliders to call update when their value is changed:
slider_Prey.on_changed(update)
slider_Pred.on_changed(update)
slider_A.on_changed(update)
slider_B.on_changed(update)
slider_C.on_changed(update)

# create the reset button axis (where its drawn)
resetax = plt.axes([0.01, 0.1, 0.1, 0.04])
# and the button itself
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_Prey.reset()
    slider_Pred.reset()
    slider_A.reset()
    slider_B.reset()
    slider_C.reset()


button.on_clicked(reset)

plt.show()

