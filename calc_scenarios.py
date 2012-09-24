import os
import socket

import numpy as np
from itertools import izip

import xija


PITCH_BINS = (45, 70, 100, 135, 170)
obsids = None


def C2F(val):
    return val * 1.8 + 32


def F2C(val):
    return (val - 32) / 1.8


def make_obsids_file(filename='obsids.npy', min_dur=3000, start='2008:001'):
    """
    Make a numpy save file with all obsids longer than ``min_dur`` since
    ``start``.
    """
    from Chandra.cmd_states import fetch_states
    from Ska.Numpy import structured_array
    dat = fetch_states(start, vals=['pitch', 'obsid'])
    dur = dat['tstop'] - dat['tstart']
    ok = dur > min_dur
    dat = dat[ok]
    dur = dur[ok]
    newdat = structured_array({'pitch': dat['pitch'],
                               'obsid': dat['obsid'],
                               'dur': dur})
    np.save(open(filename, 'w'), newdat)


def get_pitches(n_times, pitch_pdf):
    """
    Get a simulated set of pitch values for a duration of ``n_times * 328.0``
    seconds.  Draw from real mission obsids pitch / duration values.
    """
    global obsids
    if obsids is None:
        print 'Loading obsids.npy'
        obsids = np.load('obsids.npy')
    dt = 328.0  # fixed time per bin
    pitch_pdf = np.asarray(pitch_pdf, dtype=np.float64)
    pitch_times = pitch_pdf / np.sum(pitch_pdf) * n_times * dt
    obs_pitches = []
    obs_durs = []
    for p0, p1, pitch_time in zip(PITCH_BINS[:-1], PITCH_BINS[1:],
                                  pitch_times):
        if pitch_time < 1:
            continue
        ok = (obsids['pitch'] > p0) & (obsids['pitch'] < p1)
        obss = obsids[ok]
        cumtime = 0.0
        while cumtime < pitch_time:
            i = np.random.randint(len(obss))
            dur = obss[i]['dur']
            obs_durs.append(dur)
            cumtime += dur
            obs_pitches.append(obss[i]['pitch'])

        # Fix the overrun
        obs_durs[-1] -= cumtime - pitch_time

    i_shuffle = np.arange(len(obs_durs))
    np.random.shuffle(i_shuffle)
    obs_pitches = np.array(obs_pitches)[i_shuffle]
    obs_durs = np.hstack([0, np.array(obs_durs)[i_shuffle]])
    obs_times = np.cumsum(obs_durs)
    pitches = np.zeros(n_times, dtype=np.float) - 90.0
    times = np.arange(n_times) * dt
    for t0, t1, pitch in zip(obs_times[:-1], obs_times[1:], obs_pitches):
        ok = (times >= t0) & (times <= t1)
        pitches[ok] = pitch
    return pitches


def get_pitches_simple(n_times, pitch_pdf):
    """
    Original simple version of get_pitches that just assumes every
    observation is 10 ksec and places them randomly.
    """
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


def calc_model(pitch_pdf=[1, 2, 3, 4], tank_start=None):
    """
    Calculate tank model for a simulated set of observations that follow
    the specified pitch probability density function ``pitch_pdf``.  This
    argument corresponds to the fraction of observations in the
    four PITCH_BINS.  This will be normalized to one by the function.
    """
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
                        help='Starting pftank2t temperature')
    parser.add_argument('--n-sim', type=int,
                        default=100000,
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

        model = calc_model([p0, p1, p2, p3], args.tank_start)
        t_vals = model.comp['pftank2t'].mvals
        t_max = C2F(np.max(t_vals))
        t_end = C2F(t_vals[-1])

        with open(datfile, 'a') as f:
            f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.1f} {:.1f}\n'
                    .format(p0, p1, p2, p3,
                            t_max, t_end))

if __name__ == '__main__':
    main()
