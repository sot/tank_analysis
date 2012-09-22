from __future__ import division
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

import asciitable


def get_args(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Calculate tank scenarios')
    parser.add_argument('--tank-start',
                        type=int,
                        help='Starting pftank2t temperature')

    args = parser.parse_args(args)
    return args


def main(args=None):
    args = get_args()
    tank_limit = 93.0
    files = glob('*-*-{}.dat'.format(args.tank_start))
    print files

    dats = []
    for fn in files:
        dat = asciitable.read(fn, guess=False, Reader=asciitable.NoHeader,
                              names=['p0', 'p1', 'p2', 'p3',
                                     't_max', 't_end'])
        dats.append(dat)

    colds = np.hstack([dat['p3'] for dat in dats])
    hots = np.hstack([dat['p0'] + dat['p2'] for dat in dats])
    t_maxes = np.hstack([dat['t_max'] for dat in dats])
    t_ends = np.hstack([dat['t_end'] for dat in dats])

    ok_colds = []
    ok_hots = []
    n_bin = 20
    bins = np.linspace(0., 1., n_bin + 1)
    t_50 = np.zeros((n_bin, n_bin), dtype=np.float) + 70.0
    t_90 = np.zeros((n_bin, n_bin), dtype=np.float) + 70.0
    frac_bad = np.zeros((n_bin, n_bin), dtype=np.float) - 0.5

    for x0, x1 in zip(bins[:-1], bins[1:]):
        ok_colds.append((colds > x0) & (colds <= x1))
        ok_hots.append((hots > x0) & (hots <= x1))

    for i in range(n_bin):
        for j in range(n_bin):
            ok = ok_colds[i] & ok_hots[j]  # select realizations within bin
            t_ends_ok = t_ends[ok]
            if len(t_ends_ok) > 3:
                t_50[i, j], t_90[i, j] = np.percentile(t_ends_ok, [50, 90])
            t_maxes_ok = t_maxes[ok]
            if len(t_maxes_ok) > 0:
                frac_bad[i, j] = np.sum(t_maxes_ok > tank_limit) / len(t_maxes_ok)

    x = (bins[1:] + bins[:-1]) / 2.0
    y = x.copy()
    for t, pref in ((t_50, 'Median'),
                     (t_90, '90%')):
        title = '{} T(1 week) for start {} F'.format(
            pref, args.tank_start)
        plot_t_img_contour(x, y, t, title)


def plot_t_img_contour(x, y, z, title):
    X, Y = np.meshgrid(x, y)
    plt.figure()
    plt.clf()
    plt.imshow(z, origin='lower', vmin=70.0, vmax=100.0,
               extent=[0, 1, 0, 1])
    for v in (0.2, 0.4, 0.6, 0.8):
        plt.plot([0., v], [v, 0], '--k', alpha=0.5)
        plt.text(v / 2, v / 2, 'W={}'.format(1 - v),
                 ha='left', va='bottom', alpha=1, size=10)

    CS = plt.contour(X, Y, z, colors='k')
    plt.clabel(CS, inline=1, fontsize=10, fmt='%.0f')
    plt.xlabel('Hot fraction')
    plt.ylabel('Cold fraction')
    plt.title(title)


def plot_frac_bad(frac_bad):
    plt.figure(2)
    plt.clf()
    plt.imshow(frac_bad, origin='lower', vmin=0.0, vmax=1.0,
               extent=[0, 1, 0, 1])

if __name__ == '__main__':
    main()
