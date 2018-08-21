#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python semi-main.py & CUDA_VISIBLE_DEVICES=1 python semi-main.py --baseline ADMM_size

