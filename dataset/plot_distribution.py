# coding=utf-8
import copy, os, sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import sqlite3
import warnings
sys.path.insert(-1, os.getcwd())
warnings.filterwarnings('ignore')

db = sqlite3.connect('size_distribution')
cursor = db.cursor()

#
# sql= lambda x : '''
# select basetable.patient_number,
#        basetable.scan_number,
#        basetable.slice_number,
#        round(cast(basetable.slice_number as float)/cast(basetable.max as float) *10,2) as 'equvalent_slice_number',
#        basetable.class1
# from (select sizeinfo.patient_number,
#              scan_number,
#              cast(slice_number as decimal)               as 'slice_number',
#              class1,
#              cast(max_slices_per_patient.max as decimal) as `max`
#       from sizeinfo
#              left join (select patient_number, max(slice_number) as max
#                         from sizeinfo
#                         group by patient_number
#                         order by patient_number) as max_slices_per_patient
#                on sizeinfo.patient_number = max_slices_per_patient.patient_number
#       order by sizeinfo.patient_number, sizeinfo.scan_number, slice_number) as basetable
# where basetable.scan_number%s'01'
# order by basetable.patient_number,basetable.scan_number,basetable.slice_number
# '''%x
#
# data = pd.read_sql(sql('=='),db)
#
# plt.scatter(data['equvalent_slice_number'],data['class1'],label='scan_01')
# plt.grid()
#
# data = pd.read_sql(sql('<>'),db)
#
# plt.scatter(data['equvalent_slice_number'],data['class1'],label='other 01')
# plt.grid()
# plt.legend()
# plt.show()
#
#
#
# sql = '''
# select t1.patient_number,t1.scan_number, sum(t1.class1) as sum_
# from (select basetable.patient_number,
#              basetable.scan_number,
#              basetable.slice_number,
#              round(cast(basetable.slice_number as float) / cast(basetable.max as float) * 10,
#                    2) as 'equvalent_slice_number',
#              basetable.class1
#       from (select sizeinfo.patient_number,
#                    scan_number,
#                    cast(slice_number as decimal)               as 'slice_number',
#                    class1,
#                    cast(max_slices_per_patient.max as decimal) as `max`
#             from sizeinfo
#                    left join (select patient_number, max(slice_number) as max
#                               from sizeinfo
#                               group by patient_number
#                               order by patient_number) as max_slices_per_patient
#                      on sizeinfo.patient_number = max_slices_per_patient.patient_number
#             order by sizeinfo.patient_number, sizeinfo.scan_number, slice_number) as basetable
#     --       where basetable.scan_number == '01'
#       order by basetable.patient_number, basetable.scan_number, basetable.slice_number) as t1
# group by t1.patient_number,t1.scan_number
# '''
# data = pd.read_sql(sql,db)
# plt.figure()
# plt.scatter(data['patient_number'].astype(int),data['sum_'],c=(data['scan_number']=='01').astype(int))
# plt.xlabel('patient name')
# plt.legend()
# plt.grid()
# plt.ylabel('total segmentation volums')
#
#
#
# ##
sql = r'''
select basetable.patient_number,
       basetable.scan_number,
       basetable.slice_number,
       round(cast(basetable.slice_number as float) / cast(basetable.max as float) * 10, 2) as 'equvalent_slice_number',
       basetable.class1
from (
     select sizeinfo.patient_number,
             scan_number,
             cast(slice_number as decimal)               as 'slice_number',
             class1,
             cast(max_slices_per_patient.max as decimal) as `max`
      from sizeinfo
             left join (select patient_number, max(slice_number) as max
                        from sizeinfo
                        group by patient_number
                        order by patient_number) as max_slices_per_patient
               on sizeinfo.patient_number = max_slices_per_patient.patient_number
      order by sizeinfo.patient_number, sizeinfo.scan_number, slice_number) as basetable
-- where basetable.scan_number == '01'
order by basetable.patient_number, basetable.scan_number, basetable.slice_number
'''


data = pd.read_sql(sql,db)
data['equvalent_slice_number']= data['equvalent_slice_number'].map(int)

data_01 = data[data['scan_number']=='01']
data_02 = data[data['scan_number']!='01']

t1=data_01.groupby(by=['equvalent_slice_number']).mean()
t2=data_01.groupby(by=['equvalent_slice_number']).std()
t1=t1.join(t2,lsuffix='_caller', rsuffix='_other')[['class1_caller','class1_other']]
t1.columns=['mean','std']
print(t1)
t3=data_02.groupby(by=['equvalent_slice_number']).mean()
t4=data_02.groupby(by=['equvalent_slice_number']).std()
t3=t3.join(t4,lsuffix='_caller', rsuffix='_other')[['class1_caller','class1_other']]
t3.columns=['mean','std']
print(t3)