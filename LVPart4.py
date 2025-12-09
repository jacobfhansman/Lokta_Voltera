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
    N, P, Dr = x
    a, b, c, d, e, f, g, h, i = params 
    derivs = [a*N - b*N*P - e*N*Dr, c*N*P - f*Dr*P - d*P, g*N*Dr + h*P*Dr - i*Dr] 
    return derivs



# Parameters
Preymin = 1
Preymax = 100
Predmin = 1
Predmax = 100
Dragmin = 1
Dragmax = 100
Amin = .01
Bmin = .01
Cmin = .01
Dmin = .01
Emin = .01
Fmin = .01
Gmin = .01
Hmin = .01
Imin = .01
Amax = 20
Bmax = 20
Cmax = 20
Dmax = 20
Emax = 20
Fmax = 20
Gmax = 20
Hmax = 20
Imax = 20
Prey_0 = 20
Pred_0 = 20
Drag_0 = 1
a0 = 1
b0 = .5
c0 = .5
d0 = 2
e0 = 2
f0 = 2
g0 = 2
h0 = 2
i0 = 2
a = .3
b = 0.5
c = 0.5
d = .5
e = .5
f = .5
g = .3
h = .8
i = .5
params = [a, b, c, d, e, f, g, h, i]
x0=[Prey_0,Pred_0,Drag_0]
end_time = 60
tstep = 0.001

# Initial function values
t = np.arange(0, end_time, tstep)
prey, predator, dragons = odeint(lotka, x0, t, args=(params,)).T



fig = plt.figure()
ax = fig.add_axes([0.15, 0.75, 0.8, .2])
prey_plot = ax.plot(t, prey, label="Rabbits")[0]
predator_plot = ax.plot(t, predator, label="Foxes")[0]
dragon_plot = ax.plot(t,dragons,label='Dragons')[0]

ax.set_xlabel("Time")
ax.set_ylabel("Population size")
ax.legend(loc="upper right")
ax.set_title('Rabbit, Foxes, and Dragons, Oh my')
ax.grid(color="b", alpha=0.5, linestyle="dashed", linewidth=0.5)
ax.set_ylim([0, np.max([prey, predator])])


axcolor = 'lightgoldenrodyellow'
axis_Prey = fig.add_axes([0.4, 0.074, 0.45, 0.025], facecolor=axcolor)
axis_Pred = fig.add_axes([0.4, 0.115, 0.45, 0.025], facecolor=axcolor)
axis_Drag = fig.add_axes([0.4, 0.16, 0.45, 0.025], facecolor=axcolor)
axis_A = fig.add_axes([0.4, 0.2, 0.45, 0.025], facecolor=axcolor)
axis_B = fig.add_axes([0.4, 0.25, 0.45, 0.025], facecolor=axcolor)
axis_C = fig.add_axes([0.4, 0.3, 0.45, 0.025], facecolor=axcolor)
axis_D = fig.add_axes([0.4, 0.35, 0.45, 0.025], facecolor=axcolor)
axis_E = fig.add_axes([0.4, 0.4, 0.45, 0.025], facecolor=axcolor)
axis_F = fig.add_axes([0.4, 0.45, 0.45, 0.025], facecolor=axcolor)
axis_G = fig.add_axes([0.4, 0.5, 0.45, 0.025], facecolor=axcolor)
axis_H = fig.add_axes([0.4, 0.55, 0.45, 0.025], facecolor=axcolor)
axis_I = fig.add_axes([0.4, 0.6, 0.45, 0.025], facecolor=axcolor)

# size: [left, bottom, width, height]

# create each slider on its corresponding place:
slider_Prey = Slider(axis_Prey, 'Inital Rabbit Population', Preymin, Preymax, valinit=Prey_0, valstep=1)
slider_Pred = Slider(axis_Pred, 'Initial Fox Population', Predmin, Predmax, valinit=Pred_0, valstep=1)
slider_Drag = Slider(axis_Drag, 'Initial Dragon Population',Dragmin,Dragmax, valinit=Drag_0,valstep =1)
slider_A = Slider(axis_A, 'Birth rate of rabbits', Amin, Amax, valinit=a0)
slider_B = Slider(axis_B, 'Death rate of rabbits by foxes', Bmin, Bmax, valinit=b0)
slider_C = Slider(axis_C, 'Rabbits eaten per fox birth', Cmin, Cmax, valinit=c0)
slider_D = Slider(axis_D, 'Natural death rate of foxes', Dmin, Dmax, valinit=d0)
slider_E = Slider(axis_E, 'Death rate of rabbits by dragons', Emin, Emax, valinit=e0)
slider_F = Slider(axis_F, 'Death rate of foxes by dragons', Fmin, Fmax, valinit=f0)
slider_G = Slider(axis_G, 'Rabbits eaten per dragon birth', Gmin, Gmax, valinit=g0)
slider_H = Slider(axis_H, 'Foxes eaten per dragon birth', Hmin, Hmax, valinit=h0)
slider_I = Slider(axis_I, 'Natural death rate of dragons', Imin, Imax, valinit=i0)

def update(val):

    x = [slider_Prey.val, slider_Pred.val, slider_Drag.val]
    newparams = [slider_A.val, slider_B.val, slider_C.val, slider_D.val,slider_E.val, slider_F.val, slider_G.val, slider_H.val, slider_I.val]
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
slider_D.on_changed(update)
slider_E.on_changed(update)
slider_F.on_changed(update)
slider_G.on_changed(update)
slider_H.on_changed(update)
slider_I.on_changed(update)

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
    slider_D.reset()
    slider_E.reset()
    slider_F.reset()
    slider_G.reset()
    slider_H.reset()
    slider_I.reset()

button.on_clicked(reset)

plt.show()



