import os
import socket
import sys

import numpy as np
from itertools import izip

import xija


PITCH_BINS = (45, 70, 100, 135, 170)


def C2F(val):
    return val * 1.8 + 32


def F2C(val):
    return (val - 32) / 1.8


def get_pitches(n_times, pitch_pdf):
    pitch_pdf = np.asarray(pitch_pdf, dtype=np.float64)
    pitch_pdf = np.cumsum(pitch_pdf) / np.sum(pitch_pdf)
    obs_bins = range(0, n_times, 30) + [n_times]
    rands = np.random.uniform(size=len(obs_bins) - 1)
    idxs = np.searchsorted(pitch_pdf, rands)
    pitches = np.zeros(n_times, dtype=np.float64)
    for i0, i1, idx in izip(obs_bins[:-1], obs_bins[1:], idxs):
        pitches[i0:i1] = np.random.uniform(PITCH_BINS[idx],
                                           PITCH_BINS[idx + 1])

    return pitches


def calc_model(pitch_pdf=[1, 2, 3, 4], tank_start=75):
    tank_start = F2C(tank_start)  # convert to degC
    model = xija.ThermalModel('pftank2t', '2013:001', '2013:008',
                              model_spec='pftank2t_spec.json')

    times = model.times
    pitches = get_pitches(len(times), pitch_pdf)

    model.comp['pftank2t'].set_data(tank_start)
    model.comp['pf0tank2t'].set_data(tank_start)
    model.comp['pitch'].set_data(pitches, times)
    model.comp['eclipse'].set_data(False)

    model.make()
    model.calc()

    return model


def get_args(args):
    import argparse
    parser = argparse.ArgumentParser(description='Calculate tank scenarios')
    parser.add_argument('--tank-start',
                        type=int,
                        default=75,
                        help='Starting pftank2t temperature')
    parser.add_argument('--n-sim', type=int,
                        default=5000000,
                        help='Max number of simulations')
    parser.add_argument('--frac-mid', type=int,
                        default=0.2,
                        help='Fraction of hot pitch that is in 100 to 135')

    args = parser.parse_args(args)
    return args


def main(args=None):
    args = get_args(args)
    pid = os.getpid()
    hostname = socket.gethostname()
    datfile = '{}-{}-{}.dat'.format(hostname, pid, args.tank_start)

    for _ in xrange(args.n_sim):
        hot = np.random.uniform()
        cold = np.random.uniform()
        if (hot + cold) > 1.0:
            hot = 1.0 - hot
            cold = 1.0 - cold
        warm = 1.0 - hot - cold
        p0 = hot * (1 - args.frac_mid)
        p1 = warm
        p2 = hot * args.frac_mid
        p3 = cold

        model = calc_model([p0, p1, p2, p3], F2C(args.tank_start))
        t_vals = model.comp['pftank2t'].mvals
        t_max = C2F(np.max(t_vals))
        t_end = C2F(t_vals[-1])

        with open(datfile, 'a') as f:
            f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.1f} {:.1f}\n'
                    .format(p0, p1, p2, p3,
                            t_max, t_end))

if __name__ == '__main__':
    main()
