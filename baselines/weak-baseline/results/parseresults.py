import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

filenames = [x for x in os.listdir() if x.find('csv')>0]
df =0
content=None
best_filename =''
d = {}

for filename in filenames:
    filecontent = pd.read_csv(filename)
    _df = filecontent['mean'].values
    _df_f = filecontent['df'].values
    d[_df_f.item()]=filename
    if df<_df:
        df = _df
        content=filecontent
        best_filename = filename

print(best_filename, 'with', content)
a = sorted(d.keys(),reverse=True)[0:30]
for i in a:
    print(i,d[i])






