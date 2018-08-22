#!/usr/bin/env bash
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python weakly-main.py --baseline ADMM_weak &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python weakly-main.py --baseline ADMM_weak_gc &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3  python weakly-main.py --baseline ADMM_weak_size

MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python weakly-main.py --baseline ADMM_weak --lamda 0.1 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python weakly-main.py --baseline ADMM_weak_gc --lamda 0.1 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3  python weakly-main.py --baseline ADMM_weak_size --lamda 0.1

MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python weakly-main.py --baseline ADMM_weak --lamda 10 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python weakly-main.py --baseline ADMM_weak_gc --lamda 10 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3  python weakly-main.py --baseline ADMM_weak_size --lamda 10


