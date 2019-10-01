

import os,sys
sys.path.append('/home/zj2244/Projects/zjing7/aspertinker/src')
import numpy as np
import copy
from qm_input import Assemble
from quantum_io import QMResult
import pandas as pd


def main():
    ligand_lib = {
    'Water':    ['../monomers_current/00_Water.xyz',   'tinker', [0,1], [1, 2, 3]],
    'Water_H':  ['../monomers_current/00_Water.xyz',   'tinker', [0,1], [2, 1, 3]],
    'Peptide_O':['../monomers_current/05_Peptide.xyz', 'tinker', [0,1], [6, 1, 7]],
    'Peptide_N':['../monomers_current/05_Peptide.xyz', 'tinker', [0,1], [8, 9, 5]],
    'MeOH':     ['../monomers_current/04_MeOH.xyz',    'tinker', [0,1], [1, 2, 3]],
    'Benzene':  ['../monomers_current/10_Benzene.xyz', 'tinker', [0,1], [4, 2, 6]],
    'Uracil':   ['../monomers_current/08_Uracil.xyz',  'tinker', [0,1], [12, 9, 1]],
    'PO4H2':    ['../monomers_current/32_PO4H2.xyz',   'tinker', [-1,1], [2, 3, 5]],
    'Acetate':  ['../monomers/Acetate.xyz',            'xyz',    [-1,1], [4, 5, 7]],

    'Isobutane':          ['../monomers/Isobutane.xyz',         'xyz', [0,1],  [1, 6, 12]],
    '4_Thiouracil':       ['../monomers/4_Thiouracil.xyz',      'xyz', [0,1],  [1, 3, 7]],
    'Acetonitrile':       ['../monomers/Acetonitrile.xyz',      'xyz', [0,1],  [1, 4, 5]],
    'Trimethylamine':     ['../monomers/Trimethylamine.xyz',    'xyz', [0,1],  [1, 7, 12]],
    'Trimethylammonium':  ['../monomers/Trimethylammonium.xyz', 'xyz', [+1,1], [5, 6, 14]],
    'Dimethylamine':      ['../monomers/Dimethylamine.xyz',     'xyz', [0,1],  [4, 2, 3]],
    'Dimethylammonium':   ['../monomers/Dimethylammonium.xyz',  'xyz', [+1,1], [4, 2, 3]],
    'Phosphorothioate':   ['../monomers/Phosphorothioate.xyz',  'xyz', [-1,1], [1, 3, 4]],
    }
    #4_Thiouracil Acetate Acetonitrile Dimethylamine Dimethylammonium Isobutane Phosphorothioate Trimethylamine Trimethylammonium
    ml1 = 'Water Peptide_O Peptide_N MeOH'.split()
    ml2 = 'Water Peptide_O Acetate PO4H2'.split()
    ml3 = 'self Benzene Uracil'.split()

    ms1 = '4_Thiouracil Acetonitrile Phosphorothioate'.split()
    ms2 = 'Dimethylamine Dimethylammonium Trimethylamine Trimethylammonium'.split()
    ms3 = 'Isobutane '.split()

    q_lib = {}
    for mol in ligand_lib:
        spec = ligand_lib[mol]
        newm = Assemble()
        newm.read_input(spec[0], ftype=spec[1])
        newm.multiplicity = [spec[2]]
        newm.get_template()
        #newm.write_qm('%s.gjf'%mol, 'opt/b3lyp')

        q_lib[mol] =  newm

    keywords = []


    for mol in ms2:
        for lig in ml2:
            for imode, mode in enumerate(['vertical', 'face']):
                kw = {}
                kw['mol'] = mol
                kw['lig'] = lig
                kw['mode'] = mode
                kw['imode'] = imode
                kw['r0'] = 3.5
                keywords.append(kw)

    for mol in ms1:
        for lig0 in ml1:
            if lig0 == 'self':
                lig = mol
            else:
                lig = lig0
            for imode, mode in enumerate(['vertical', 'T-shape']):
                kw = {}
                kw['mol'] = mol
                kw['lig'] = lig
                kw['mode'] = mode
                kw['imode'] = imode
                kw['r0'] = 3.0
                keywords.append(kw)
    for mol in ms3:
        for lig0 in ml3:
            if lig0 == 'self':
                lig = mol
            else:
                lig = lig0
            for imode, mode in enumerate(['vertical', 'parallel']):
                kw = {}
                kw['mol'] = mol
                kw['lig'] = lig
                kw['mode'] = mode
                kw['imode'] = imode
                kw['r0'] = 5.0
                keywords.append(kw)

    outdir = 'g16'
    outdir = 'opt2'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    qm_summary = pd.DataFrame()
    for kw in keywords:
        mol = kw['mol']
        lig = kw['lig']
        mode = kw['mode']
        imode = kw['imode']
        r0 = kw['r0']

        anch1 = ligand_lib[mol][3]
        q0 = q_lib[lig]
        anch2 = ligand_lib[lig][3]

        fname = '%s/%s-%s_%d.gjf'%(outdir, mol, lig, imode+1)
        q1 = copy.deepcopy(q_lib[mol])
        q1.add_mol(q0, anch1, anch2, mode=mode, r0=r0)
        if q0.multiplicity[0][0] != 0 or q1.multiplicity[0][0] != 0:
            theory = 'opt/b3lyp/pcm'
        else:
            theory = 'opt/b3lyp'
        #q1.write_qm(fname, theory)

        for outdir in ('opt1', 'opt2'):
            f2name = '%s/%s-%s_%d.out'%(outdir, mol, lig, imode+1)
            if os.path.isfile(fname) and os.path.isfile(f2name):
                q2 = QMResult()
                q2.assign_geo(q1)
                q2.read_qm(f2name, ftype='g16')

                q2.find_best_frame()

                newdf = (pd.DataFrame((q2.qm_data.loc[[q2.iframe],:])))
                newdf.index = [f2name]

                if len(qm_summary) == 0:
                    qm_summary = newdf
                else:
                    qm_summary = qm_summary.append(newdf)

                #q3 = QMInput()
                #q3.get_template()
                #q3.assign_geo(q0)
                #q3.write_qm('7.gjf', theory='opt/b3lyp')
    qm_summary.to_csv('9.csv')

if __name__ == '__main__':
    main()
    pass
