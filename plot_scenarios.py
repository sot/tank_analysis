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
                n_bad = np.sum(t_maxes_ok > tank_limit)
                frac_bad[i, j] = n_bad / len(t_maxes_ok)

    x = (bins[1:] + bins[:-1]) / 2.0
    y = x.copy()
    for perc, t, pref in ((50, t_50, 'Median'),
                          (90, t_90, '90%')):
        title = '{} temp after 7 days, start={} F'.format(
            pref, args.tank_start)
        plot_t_img_contour(x, y, t, title, 'tank_perc{}_start{}.png'.format(
                perc, args.tank_start))

    title = 'Fraction exceeding 93F limit, start={}'.format(args.tank_start)
    filename = 'tank_bad_start{}.png'.format(args.tank_start)
    plot_frac_bad(x, y, frac_bad, title, filename)


def plot_t_img_contour(x, y, z, title, filename=None):
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
    if filename:
        plt.savefig(filename)


def plot_frac_bad(x, y, z, title, filename=None):
    X, Y = np.meshgrid(x, y)
    plt.figure(2)
    plt.clf()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    CS = plt.contour(X, Y, z, [0.01, 0.1, 0.5, 0.9, 0.99],
                     colors=('k', 'b', 'c', 'm', 'r'))
    plt.clabel(CS, inline=1, fontsize=10, fmt='%.2f')

    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        plt.plot([0., v], [v, 0], '--k', alpha=0.5)
        plt.text(v / 2, v / 2, 'W={}'.format(1 - v),
                 ha='left', va='bottom', alpha=1, size=10)

    plt.xlabel('Hot fraction')
    plt.ylabel('Cold fraction')
    plt.title(title)

    plt.grid()

    if filename:
        plt.savefig(filename)

if __name__ == '__main__':
    main()
