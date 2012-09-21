import numpy as np

import xija
import matplotlib.pyplot as plt


def calc_model(pitch):
    model = xija.ThermalModel('pftank2t', '2013:001', '2013:040',
                              model_spec='pftank2t_model_spec.json')

    times = model.times
    days = (model.times - model.times[0]) / 86400.
    ok = (days > 10.0) & (days < 30.0)
    pitches = np.zeros_like(times) + 160.0
    pitches[ok] = pitch

    model.comp['pftank2t'].set_data(22.0)
    model.comp['pf0tank2t'].set_data(22.0)
    model.comp['pitch'].set_data(pitches, times)
    model.comp['eclipse'].set_data(False)

    model.make()
    model.calc()

    return model


plt.figure(1)
plt.clf()
for pitch in range(45, 161, 15):
    model = calc_model(pitch)
    days = (model.times - model.times[0]) / 86400.
    pftank2t_f = model.comp['pftank2t'].mvals * 1.8 + 32
    plt.plot(days, pftank2t_f, label='{} deg'.format(pitch))
    plt.text(30, np.max(pftank2t_f), '{} deg'.format(pitch),
             va='bottom', ha='right')
plt.grid()
plt.xlabel('Days')
plt.ylabel('PFTANK2T (degF)')
plt.title('PFTANK2T profile at different pitches')
plt.savefig('pftank2t_timescales.png')
