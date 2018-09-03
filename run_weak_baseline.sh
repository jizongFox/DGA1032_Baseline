#!/usr/bin/env bash
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python weakly-main.py --baseline ADMM_weak &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python weakly-main.py --baseline ADMM_weak_gc &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python weakly-main.py --baseline ADMM_weak_size

