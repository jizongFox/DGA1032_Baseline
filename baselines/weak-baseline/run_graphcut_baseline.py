import os
from multiprocessing import Pool

kernel_sizes =[5,7]
lamdas = [0.01,  0.1, 1,10  ]
sigmas = [ 0.01, 0.1, 0.2, 1]
dilation_levels = [3,  5, 7,   9]



cmd_package = []

for k in kernel_sizes:
    for l in lamdas:
        for s in sigmas:
            for d in dilation_levels:
                cmd = 'python graphcut_baseline.py --kernel_size %d  --lamda %.2f --sigma  %.3f --dilation_level %d'%(k,l,s,d)
                cmd_package.append(cmd)

print(cmd_package.__len__())

if __name__ =='__main__':
    p =Pool(4)
    p.map(os.system, cmd_package)


