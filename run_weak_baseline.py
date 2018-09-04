import os,numpy as np
epses = [0.05, 0.1, 0.2, 0.4, 0.6]
baselines = ['ADMM_weak','ADMM_weak_gc','ADMM_weak_size']
choice_of_GPU = [0,1]
cmds = []
for eps in epses:
    for baseline in baselines:
        cmd = "MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --baseline %s --eps %f"%(np.random.choice(choice_of_GPU),baseline,eps)
        cmds.append(cmd)

if __name__ =='__main__':
    from multiprocessing import Pool
    p = Pool(4)
    p.map(os.system, cmds)



