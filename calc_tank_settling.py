#!/usr/bin/env python

"""
Compute and plot the maximum 1DPAMZT settling temperature at any pitch as a
function of time.
"""

import numpy as np
import matplotlib.pyplot as plt

import json
import xija
from Chandra.Time import DateTime


def CtoF(cs):
    try:
        return [c * 1.8 + 32 for c in cs]
    except TypeError:
        return cs * 1.8 + 32


def calc_model_pyger(self, states, times, T0s, state_only=False):
    print 'TANK:', CtoF(T0s)
    model = xija.ThermalModel('tank', start=states['tstart'][0],
                              stop=states['tstop'][-1],
                              model_spec=self.model_spec)

    state_times = np.array([states['tstart'], states['tstop']])
    model.comp['pitch'].set_data(states['pitch'], state_times)
    model.comp['eclipse'].set_data(False)
    model.comp['pf0tank2t'].set_data(T0s[0])
    model.comp['pftank2t'].set_data(T0s[0])

    model.make()
    model.calc()
    # T_tank = Ska.Numpy.interpolate(model.comp['pftank2t'].mvals,
    #                       xin=model.times, xout=times, sorted=True)

    # return np.vstack([T_tank])


def calc_model(start, pitch, model_spec='pftank2t_model_spec.json'):
    stop = DateTime(start) + 20
    model = xija.ThermalModel('dpa', start=start, stop=stop,
                              model_spec=model_spec)

    model.comp['eclipse'].set_data(False)
    model.comp['pf0tank2t'].set_data(20.0)
    model.comp['pftank2t'].set_data(20.0)
    model.comp['pitch'].set_data(pitch)

    model.make()
    model.calc()
    return model


def main():
    plt.figure()
    plt.clf()
    pitches = range(45, 170, 2)  # only consider tail sun for this plot
    model_spec = json.load(open('pyger/pftank2t_model_spec.json', 'r'))

    pitch_temps = []
    pitch0_temps = []
    tstart = '2013:001'
    for pitch in pitches:
        model = calc_model(tstart, pitch, model_spec)
        pitch_temps.append(model.comp['pftank2t'].mvals[-1])
        pitch0_temps.append(model.comp['pf0tank2t'].mvals[-1])
    plt.plot(pitches, np.array(pitch_temps) * 1.8 + 32.0)
    # plt.plot(pitches, (pitch0_temps))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
