import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

filenames = [x for x in os.listdir() if x.find('csv')>0]
df =0
content=None
best_filename =''

for filename in filenames:
    filecontent = pd.read_csv(filename)
    _df = filecontent['mean'].values
    if df<_df:
        df = _df
        content=filecontent
        best_filename = filename

print(best_filename, 'with', content)





