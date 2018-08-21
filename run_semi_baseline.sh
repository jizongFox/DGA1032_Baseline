#!/usr/bin/env bash
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python semi-main.py &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python semi-main.py --baseline ADMM_gc &

MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python semi-main.py --lamda 0.1  &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python semi-main.py --baseline ADMM_gc --lamda 0.1

MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python semi-main.py --lamda 10 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0  python semi-main.py --baseline ADMM_gc --lamda 10 &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python semi-main.py --lamda 20  &
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python semi-main.py --baseline ADMM_gc --lamda 20


