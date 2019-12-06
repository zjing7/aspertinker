#!/home/zj2244/Public/Software/anaconda3/bin/python

import time
import os,sys
sys.path.append('/home/zj2244/Projects/zjing7/aspertinker/src')
import numpy as np
import copy
#from qm_input import QMInput
#from quantum_io import QMResult
from quantum_inp import QMInput
from quantum_out import QMResult
import pandas as pd
from qm_utils import plot_cost, plot_time, scale_spin, extrapolate_mp2
import seaborn as sns
import matplotlib.pyplot as plt
col_pal = sns.color_palette('Set2', 8)
sns.set( style='ticks', palette=col_pal, color_codes=False )
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

DATA_DF1 = 'dat/df_data0.csv'
DATA_DF2 = 'dat/df_time0.csv'

def main2():
    inps = sys.argv[1:3]
    qA = QMInput()
    qB = QMInput()
    qAB = QMInput()
    qA.read_input(inps[0], ftype='tinker')
    qB.read_input(inps[1], ftype='tinker')

    qAB.verbose = 2
    qAB.assign_geo(qA)
    #print(qAB.topology)
    qAB.append_atom(qB)
    #print(qAB.topology)
    #print(qAB.groups)
    qAB.delete_atom(range(4, 50))
    print(qAB.topology)
    print(qAB.groups)
    qAB.write_struct('tmp.xyz', ftype='tinker')

def main():
    pass
    inp1 = sys.argv[1]
    nA = int(sys.argv[2])
    qAB = QMInput()
    qAB.read_input(inp1, ftype='tinker')
    nAB = qAB.natoms
    print(nA, nAB, qAB.group_idx, sep='\n')
    #qAB.group_idx = [list(range(1, nA+1)), list(range(nA+1, nAB+1))]
    qAB.set_group_index([list(range(1, nA+1)), list(range(nA+1, nAB+1))])
    #qAB.find_interface()
    qAB.add_disp(list(range(nA+1, nAB+1)), mode='auto', rcutoff=3.0, direction='y', dr=3)
    qAB.write_struct('test_y2.xyz', ftype='xyz')

    qAB.debug = True
    qAB.guess_mass()
    #qAB.convert_top()
    #print(qAB.topology)
    qAB.verbose = 2
    #qAB.reorder_index(start=5)
    #print(qAB.topology)
    qAB.reorder_index([k*5+1 for k in range(qAB.natoms)])
    print(qAB.index_to_rank([36, 41, 50]))
    print(qAB.topology)
    print(qAB.groups)

main2()
