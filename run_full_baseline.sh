#!/usr/bin/env bash
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1  python full-main.py --loss_function CE
MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2  python full-main.py --loss_function MSE
