#!/usr/bin/env bash
python graphcut_baseline.py --sr 0.01 &
python graphcut_baseline.py --sr 0.03 &
python graphcut_baseline.py --sr 0.05 &
python graphcut_baseline.py --sr 0.1 &
python graphcut_baseline.py --sr 0.2
