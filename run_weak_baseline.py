import os, numpy as np

epses = [0.05, 0.1]
baselines = ['ADMM_weak']
choice_of_GPU = [1, 2]
cmds = []
for eps in epses:
    for baseline in baselines:
        cmd = "OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --baseline %s --eps %f" % (
            np.random.choice(choice_of_GPU), baseline, eps)
        cmds.append(cmd)
cmds.append("OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --baseline %s --eps %f" % (
    np.random.choice(choice_of_GPU), 'ADMM_weak_gc', 0))

if __name__ == '__main__':
    from multiprocessing import Pool

    print(cmds)
    p = Pool(4)
    p.map(os.system, cmds)
