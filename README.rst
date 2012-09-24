IPS tank thermal analysis
============================

This repository contains code to estimate the requirements for cold, warm, and hot dwell
times for the IPS fuel tank temperature.

- ``calc_scenarios.py``: Calculate Monte-Carlo simulations of different cold, warm, and hot fraction scenarios for a given starting temperature.
- ``calc_tank_settling.py``: Compute tank settling temperature as a function of pitch angle.
- ``gui_scenarios.py``: Interactively show the results of cold/warm/hot scenarios.
- ``plot_scenarios.py``: Plot the results from the outputs of ``calc_scenarios.py``.
- ``plot_timescales.py``: Make plots illustrating the time behavior of the tank model.
