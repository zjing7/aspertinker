#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
col_pal = sns.color_palette('Set2', 8)
sns.set( style='ticks', palette=col_pal, color_codes=False )
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

def extrapolate_mp2(df, names_in, name_out, coefs=[], zetas=[], spin_out='Total', spin_names=['Ref', 'MP2_SS', 'MP2_OS']):
    nspin_names = len(spin_names)
    if len(coefs) == 0 and len(zetas) == 2 and len(names_in) == 2:
        # defining coefs from zetas
        c1 = zetas[0]**3.0 / (zetas[0]**3.0  - zetas[1]**3.0)
        c2 = -zetas[1]**3.0 / (zetas[0]**3.0  - zetas[1]**3.0)
        coefs = np.ones((nspin_names, 2))
        if zetas[0] < zetas[1]:
            coefs[0, 0] = 0
        else:
            coefs[0, 1] = 0
        coefs[1:, 0] = c1
        coefs[1:, 1] = c2
    elif (type(coefs) is np.ndarray and len(coefs.shape) == 1 and len(coefs) == 2) \
        or (type(coefs) in (list, tuple) and len(coefs) == 2 and isinstance(coefs[0], float)):
        c1, c2 = coefs
        coefs = np.ones((nspin_names, 2))
        if c1 < c2:
            coefs[0, 0] = 0
        else:
            coefs[0, 1] = 0
        coefs[1:, 0] = c1
        coefs[1:, 1] = c2
    elif (type(coefs) is np.ndarray and coefs.shape == (nspin_names, 2)):
        pass
    else:
        print("ERROR: no rules found")
        return None
    df.loc[(name_out, spin_out), :] = 0
    for coef, spin_name in zip(coefs, spin_names):
        df.loc[(name_out, spin_out), :] += df.loc[(names_in[0], spin_name), :] * coef[0]
        df.loc[(name_out, spin_out), :] += df.loc[(names_in[1], spin_name), :] * coef[1]
        
def scale_spin(df, name_in, name_out, spin_out='Total', spin_names=['Ref', 'MP2_SS', 'MP2_OS'], coefs=(1.0, 1.29, 0.4)):
    if len(coefs) != len(spin_names):
        print('ERROR: Number of Components does not match number of coeficients')
        return None
    for spin_name in spin_names:
        if (name_in, spin_name) not in df.index:
            print("ERROR: Component %s not found"%(spin_name))
            return None
    coefs = np.array(coefs)
    df.loc[(name_out, spin_out), :] = 0
    for coef, spin_name in zip(coefs, spin_names):
        df.loc[(name_out, spin_out), :] += df.loc[(name_in, spin_name), :] * coef

def plot_time(df, outf='fig_efficiency.png', col_time='Time', col_err='RMSE'):
    if not (col_time in df):
        return
    x = np.arange(len(df.index))
    if len(x) == 0:
        return
    y2 = df[col_time]
    width = 0.35

    fig, ax1 = plt.subplots()
    color = col_pal[0]
    ax1.set_xlabel('Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=90)
    ax2 = ax1
    ax2.set_ylabel('Time (s)', color=color)
    ax2.set_yscale('log')
    ax2.bar(x+width/2, y2, width, color=color, label=col_time)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper right')


    fig.tight_layout()
    plt.savefig(outf, dpi=300)
    plt.close()

def plot_cost(df, outf='fig_efficiency.png', col_time='Time', col_err='RMSE'):
    if not (col_time in df and col_err in df):
        return
    x = np.arange(len(df.index))
    if len(x) == 0:
        return
    width = 0.35
    y1 = df[col_err]
    y2 = df[col_time]

    outliers = []
    for i in range(len(x)-1):
        e0 = df[col_err][i]
        t0 = df[col_time][i]

        if e0 < 1.0*np.min(df[col_err][df[col_time] < t0]):
            outliers.append(i)
        elif e0 > 1.5*np.min(df[col_err][df[col_time] < t0]):
            pass
            #outliers.append(i)


    fig, ax1 = plt.subplots()
    color = col_pal[1]

    ax1.set_xlabel('Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df.index, rotation=90)
    ticks = ax1.xaxis.get_ticklabels()
    for i in outliers:
        ticks[i].set_color('r')

    #y1max = min(2, max(y1)*1.2)
    #y1max = min(max(y1)*1.1, min(y1[y1>0])*4)
    y1max = min(max(y1)*1.1, np.median(y1)*2)
    ax1.set_ylim(0, y1max)
    ax1.set_ylabel('Error (kcal/mol)', color=color)
    ax1.bar(x-width/2, y1, width, color=color, label=col_err)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper left')

    color = col_pal[0]
    ax2 = ax1.twinx()
    ax2.set_ylabel('Time (s)', color=color)
    ax2.set_yscale('log')
    ax2.bar(x+width/2, y2, width, color=color, label=col_time)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper right')


    fig.tight_layout()
    plt.savefig(outf, dpi=300)
    plt.close()


    return
    figb, axb1 = plt.subplots()
    axb1.plot(y1, y2, 'o')
    axb1.set_xlabel(col_err)
    axb1.set_ylabel(col_time)
    axb1.set_yscale('log')
    #figb.savefig('correlation.png', dpi=300)

