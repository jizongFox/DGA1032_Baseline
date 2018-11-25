import os
from multiprocessing import Pool

modelnames = [x for x in os.listdir('../../semi_pretrain_checkpoint') if x.find('.pth')>0]
kernel_sizes =[5,7]
lamdas = [0.01,  0.1, 1,10  ]
sigmas = [ 0.01, 0.1, 0.2, 1]

cmd_package = []

for m in modelnames:
    for k in kernel_sizes:
        for l in lamdas:
            for s in sigmas:
                cmd = 'python graphcut_pretrain_baseline.py --model_name %s --kernelsize %d  --lamda %.2f --sigma  %.3f '%(m,k,l,s)
                cmd_package.append(cmd)

print(cmd_package.__len__())

if __name__ =='__main__':
    p =Pool(4)
    p.map(os.system, cmd_package)