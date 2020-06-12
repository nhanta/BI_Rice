# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:44:36 2020

@author: Nhan TV
"""

# Import the libraries 
import pandas as pd

# Import data by chunksize
chunksize = 100
X_ct = pd.DataFrame()
for chunk in pd.read_csv("Xcont.csv",low_memory=False, chunksize=chunksize, memory_map=True):
    X_ct = pd.concat([X_ct, chunk])
X_ct.set_index("IID", inplace = True)
X_cont = X_ct.drop(X_ct.columns[404388:].to_list(), axis = 1)

# Change names of columns
names = X_cont.columns.values
col = []
for x in names:
  col.append(x[:x.index('_')])
new_col = dict(zip(names.tolist(), col))
X_cont = X_cont.rename(columns = new_col)

# Import data after filtering 
grain_weight = pd.read_csv("cont_LMM_grain.weight_p0.05_sort.tsv", sep = "\t")
time_flowering = pd.read_csv("cont_LMM_time.flowering_p0.05_sort.tsv", sep = "\t")
y_cont = pd.read_csv("Ycont.txt", sep = "\t")

# Transform SNP id to string
grain_weight["SNP"] = grain_weight["SNP"].astype('str')
time_flowering["SNP"] = time_flowering["SNP"].astype('str')

# Output variables
y_tfl = y_cont.iloc[:, 1]
y_grw = y_cont['Grain.weight']
y_tfl_grw = y_cont.iloc[:, [1, 3]]

# Rename p-value feature of each data
time_flowering = time_flowering.rename(columns = {'P':'P_tfl'})
grain_weight = grain_weight.rename(columns = {'P':'P_grw'})

# Find SNPs of time to flowering and grain weight satisfying p_value threshold
var_tfl_grw = []
var_tfl_grw.append(list(set(time_flowering["SNP"][time_flowering["P_tfl"]<=5*10**(-2)]).intersection(set(grain_weight["SNP"][grain_weight["P_grw"]<=5*10**(-2)]))))
var_tfl_grw.append(list(set(time_flowering["SNP"][time_flowering["P_tfl"]<=2.5*10**(-2)]).intersection(set(grain_weight["SNP"][grain_weight["P_grw"]<=2.5*10**(-2)]))))
var_tfl_grw.append(list(set(time_flowering["SNP"][time_flowering["P_tfl"]<=1*10**(-2)]).intersection(set(grain_weight["SNP"][grain_weight["P_grw"]<=1*10**(-2)]))))
var_tfl_grw.append(list(set(time_flowering["SNP"][time_flowering["P_tfl"]<=7.5*10**(-3)]).intersection(set(grain_weight["SNP"][grain_weight["P_grw"]<=7.5*10**(-3)]))))
var_tfl_grw.append(list(set(time_flowering["SNP"][time_flowering["P_tfl"]<=5*10**(-3)]).intersection(set(grain_weight["SNP"][grain_weight["P_grw"]<=5*10**(-3)]))))

# Find SNPs of time to flowering and grain weight satisfying p_value threshold
var_grain_weight = []
var_time_flowering = []
for i in range(2, 9):
  var_grain_weight.append(grain_weight['SNP'][grain_weight['P_grw']<= 5*10**(-i)])
  var_time_flowering.append(time_flowering['SNP'][time_flowering['P_tfl']<= 5*10**(-i)])
  
# SNPs for advanced regression
var_ls_el = pd.read_csv('indep_1000_10_0.3.prune-43k.in', sep = "\t", header=None)
var_ls_el[0] = var_ls_el[0].astype("str")

# Saving files after filtering  
y_tfl.to_csv('y_tfl')
y_grw.to_csv('y_grw')
y_tfl_grw.to_csv('y_tfl_grw')

X_cont_ls_el = X_cont[var_ls_el[0]]
X_cont_ls_el.to_csv('X_cont_ls_el.csv')

X_tfl_2 = X_cont[var_time_flowering[0]]
X_tfl_2.to_csv('X_tfl_2.csv')
X_tfl_3 = X_cont[var_time_flowering[1]]
X_tfl_3.to_csv('X_tfl_3.csv')
X_tfl_4 = X_cont[var_time_flowering[2]]
X_tfl_4.to_csv('X_tfl_4.csv')
X_tfl_5 = X_cont[var_time_flowering[3]]
X_tfl_5.to_csv('X_tfl_5.csv')
X_tfl_6 = X_cont[var_time_flowering[4]]
X_tfl_6.to_csv('X_tfl_6.csv')

X_grw_2 = X_cont[var_grain_weight[0]]
X_grw_2.to_csv('X_grw_2.csv')
X_grw_3 = X_cont[var_grain_weight[1]]
X_grw_3.to_csv('X_grw_3.csv')
X_grw_4 = X_cont[var_grain_weight[2]]
X_grw_4.to_csv('X_grw_4.csv')
X_grw_5 = X_cont[var_grain_weight[3]]
X_grw_5.to_csv('X_grw_5.csv')

X_tfl_grw_2= X_cont[var_tfl_grw[0]]
X_tfl_grw_2.to_csv('X_tfl_grw_2.csv')
X_tfl_grw_25= X_cont[var_tfl_grw[1]]
X_tfl_grw_25.to_csv('X_tfl_grw_25.csv')
X_tfl_grw_1= X_cont[var_tfl_grw[2]]
X_tfl_grw_1.to_csv('X_tfl_grw_1.csv')
X_tfl_grw_75= X_cont[var_tfl_grw[3]]
X_tfl_grw_75.to_csv('X_tfl_grw_75.csv')
X_tfl_grw_3= X_cont[var_tfl_grw[4]]
X_tfl_grw_3.to_csv('X_tfl_grw_3.csv')

    
