import pandas as pd
import os

filenames= [x for x in os.listdir() if x.find('.csv')>0]

model_names = list(set(['_'.join(x.split('_')[0:2]) for x in filenames]))
model_names.sort()

print(model_names)

for model_name in model_names:
	best_f_dice=-1
	best_config = None

	for f in [x for x in filenames if x.find(model_name)>=0]:
		results = pd.read_csv(f)
		if results['graphcut_f_dice'].values>best_f_dice:
			best_f_dice = results['graphcut_f_dice'].values
			best_config = f

	print('Best performance for the model %s is: graphcut_f_dice of %.4f, with config of %s'%(
		model_name,best_f_dice,best_config))