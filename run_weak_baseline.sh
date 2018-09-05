#!/usr/bin/env bash
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python weakly-main.py --baseline ADMM_weak --lowbound 1 --highbound 2200 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python weakly-main.py --baseline ADMM_weak_gc --lowbound 1 --highbound 2200 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3  python weakly-main.py --baseline ADMM_weak_size --lowbound 1 --highbound 2200

