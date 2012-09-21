import numpy as np

from calc_scenarios import calc_model, C2F, F2C


def run_sim(pitch_pdf, tank_start, n_sim=20):
    t_vals = []
    for _ in range(n_sim):
        model = calc_model(pitch_pdf, tank_start)
        t_vals.append(model.comp['pftank2t'].mvals)
    t_vals = np.vstack(t_vals)
    t10, t50, t90 = np.percentile(C2F(t_vals), [10, 50, 90], axis=0)
    days = (model.times - model.times[0]) / 86400.
    return days, t10, t50, t90


def get_args(args):
    import argparse
    parser = argparse.ArgumentParser(description='Calculate tank scenarios')
    parser.add_argument('--tank-start',
                        type=int,
                        default=75,
                        help='Starting pftank2t temperature')
    parser.add_argument('--n-sim', type=int,
                        default=20,
                        help='Max number of simulations')
    parser.add_argument('--frac-mid', type=int,
                        default=0.2,
                        help='Fraction of hot pitch that is in 100 to 135')

    args = parser.parse_args(args)
    return args


def main(args=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    args = get_args(args)

    plt.clf()
    ax = plt.subplot(1, 1, 1)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    ax_color = 'lightgoldenrodyellow'
    ax_s1 = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=ax_color)
    ax_s2 = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=ax_color)
    ax_t0 = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=ax_color)

    s1 = Slider(ax_s1, 'Hot', 0.0, 1.0, valinit=0.1)
    s2 = Slider(ax_s2, 'Cold', 0.0, 1.0, valinit=0.2)
    t0 = Slider(ax_t0, 'Tank', 70, 100, valinit=75)

    def slider_update(val=None):
        global l10, l50, l90
        hot = s1.val
        cold = s2.val
        if hot + cold > 1.0:
            cold = 1 - hot
            s2.set_val(cold)
        warm = 1.0 - hot - cold
        print cold, warm, hot

        p0 = hot * (1 - args.frac_mid)
        p1 = warm
        p2 = hot * args.frac_mid
        p3 = cold
        days, t10, t50, t90 = run_sim([p0, p1, p2, p3],
                                      t0.val, n_sim=args.n_sim)
        if val is None:
            l10, = ax.plot(days, t10, label='10%')
            l50, = ax.plot(days, t50, label='50%')
            l90, = ax.plot(days, t90, label='90%')
            ax.set_ylim(70, 110)
        else:
            l10.set_ydata(t10)
            l50.set_ydata(t50)
            l90.set_ydata(t90)

    slider_update()
    s1.on_changed(slider_update)
    s2.on_changed(slider_update)
    t0.on_changed(slider_update)

    plt.show()
    fig = plt.gcf()
    fig.canvas.start_event_loop_default()


if __name__ == '__main__':
    main()
