from __future__ import division
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

import Ska.Numpy
import asciitable


def get_args(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Calculate tank scenarios')
    parser.add_argument('--tank-start',
                        type=int,
                        help='Starting pftank2t temperature')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='Do not save plots')

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
        filename = None if args.no_save else 'tank_perc{}_start{}.png'.format(
            perc, args.tank_start)
        plot_t_img_contour(x, y, t, title, filename)

    title = 'Fraction exceeding 93F limit, start={}'.format(args.tank_start)
    filename = None if args.no_save else 'tank_bad_start{}.png'.format(
        args.tank_start)
    plot_frac_bad(x, y, frac_bad, title, filename)

    title = 'blah start={}'.format(args.tank_start)
    filename = None if args.no_save else 'tank_end_10_start{}.png'.format(
        args.tank_start)
    plot_t_50_at_frac_bad_10(args.tank_start, x, y, t_50,
                             title, filename=None)


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


def plot_t_50_at_frac_bad_10(tank_start, x, y, t_50, title, filename=None):
    """
    Plot the value of ending temperature (roughly) along the 0.10 contour
    in the frac_bad plot.  The data values are actually taken along a line
    that was eyeballed from the plots.

    This routine plots a line corresponding to a single starting temperature.
    To make the plot with all four starting temps you need to run the
    whole program four times::

      run plot_scenarios.py --tank-start 75 --no-save
      run plot_scenarios.py --tank-start 80 --no-save
      run plot_scenarios.py --tank-start 85 --no-save
      run plot_scenarios.py --tank-start 90 --no-save
    """
    points = {75: [(0.1, 0), (0.6, 0.3)],
              80: [(0.03, 0), (0.6, 0.34)],
              85: [(0.0, 0.05), (0.52, 0.4)],
              90: [(0.0, 0.13), (0.4, 0.51)],
              }
    points = points[tank_start]
    xs = [points[0][0], points[1][0]]
    ys = [points[0][1], points[1][1]]
    r = np.polyfit(xs, ys, 1)
    hot = x.copy()
    cold = np.polyval(r, hot)
    warm = 1 - hot - cold
    ok = warm > 0.0
    cold = cold[ok]
    hot = hot[ok]
    t_50_10 = []
    for j in range(len(hot)):
        t_50_10.append(Ska.Numpy.interpolate(t_50[:, j], y, [cold[j]],
                                             method='linear')[0])

    plt.figure(10)
    plt.xlim(0, 1)
    plt.plot(hot, t_50_10)

    plt.xlabel('Hot fraction')
    plt.ylabel('Median temp after 7 days (F)')
    plt.title('Start=75 blue, 80 green, 85 red, 90 cyan')


if __name__ == '__main__':
    main()
