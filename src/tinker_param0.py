#!/usr/bin/env python
'''
Read, write and manipulate Tinker prm file

THIS IS LEGACY CODE. DO NOT CHANGE.
'''

import numpy as np
import pandas as pd
import sys
import os
import copy

class TinkerParam0:
    def __init__(self):
        self.floats = set((float, np.float32, np.float64, np.float128))

        self.syntax = {}
        self.params = {}
        self.termlist = []
        self.outfmt = {}
        self.use_atype = set(['multipole', 'polarize', 'atom', 'biotype'])

        self.header = ''

        # map
        self.d_type2class = {}

        self.init_syntax()
        self.init_params()
        self.init_termlist()
        self.init_outfmt()


    def init_termlist(self):
        syntax = list(self.syntax.items())
        # sort by (number of id, term name, id)
        for K in syntax:
            len(K[1][0][0])
        term_order = 'atom multipole polarize vdw bond bnd angle ureybrad pi opbend torsion tors'.split()
        sorting = lambda t : (t[0] == 'biotype', list((K not in t[0]) for K in term_order), len(t[1][0][0]))
        #syntax.sort(key = lambda t: ( (t[0] in ['biotype']), not (t[0] in self.use_atype), len(t[1][0][0]),t[0], t[1][0][0][-3:]))
        syntax.sort(key = sorting)
        self.termlist = [K[0] for K in syntax]

    def init_params(self):
        for term in self.syntax:
            self.params[term] = {}

    def init_outfmt(self):
        '''
        Define format for data types and energy terms
        '''
        self.outfmt[int] = '%4d' 
        # for index, type
        self.outfmt[np.int32] = '%4d' 
        # for biotype
        self.outfmt[np.int8] = '%2d' 
        # for coord nr, periodicity
        self.outfmt[np.float16] = '%.1f' 
        # for phase angle

        self.outfmt[float] = '%9.4f' 
        # for non-bond
        self.outfmt[np.float128] = '%10.5f' 
        # for multipole
        self.outfmt[np.float32] = '%7.3f' 
        # for bond param
        self.outfmt[str] = '%-5s'
        self.outfmt[np.string_] = '%s'

        self.outfmt['atom'] = 'atom %s %s %s %-30s %s %s %s\n'
        self.outfmt['multipole'] = 'multipole %-20s %s\n%63s\n%41s\n%52s\n%63s\n'
        self.outfmt['biotype'] = 'biotype %s %-5s %-30s %s\n'
        #self.outfmt['multipole'] = 'multipole %5s %-14s %s\n%63s\n%41s\n%52s\n%63s\n'

    def init_syntax(self):
        # defined by a range or a tuple containing the bounds of range
        # data types
        # whether subject to mutation. 0, no; 1, yes.
        self.syntax['atom'] = (([1], [2], [3],(4, -3), [-3], [-2], [-1]), 
                               (int, int, str, np.string_, np.int8, float, np.int8),
                               (0, 0, 0, 0, 0, 0, 0))
        self.syntax['biotype'] = (([1], [2], (3, -1), [-1]), 
                               (np.int32, str, np.string_, int),
                               (0, 0, 0, 0))
        self.syntax['multipole'] = (((1, -10),[-10],  [-9,-8,-7], [-6], [-5,-4], [-3,-2,-1]),
                                    (int, np.float128, np.float128, np.float128, np.float128, np.float128),
                                    (0, 1, 1, 1, 1, 1))
        self.syntax['vdw'] = (([1], [2,3], range(4,13)),
                                    (int, float, float),
                                    (0, 1, 0))
        self.syntax['polarize'] = (([1], [2], [3], range(4,33)),
                                    (int, float,float, int), 
                                    (0, 1, 0, 0))
        self.syntax['bond'] = (([1,2], [3],[4], ),
                                    (int, float, float),
                                    (0,1,0))
        self.syntax['angle'] = (([1,2,3], [4],[5], ),
                                    (int, np.float32, np.float32), 
                                    (0,1,0))
        self.syntax['strbnd'] = (([1,2,3], [4,5], ),
                                    (int, float),
                                    (0, 1))
        self.syntax['opbend'] = (([1,2,3,4], [5], ),
                                    (int, np.float32), 
                                    (0,1))
        self.syntax['torsion'] = (([1,2,3,4], [5], [6], [7], [8], [9], [10], [11], [12], [13]),
                                    (int, float, np.float16, np.int8, float, np.float16, np.int8, float, np.float16, np.int8), 
                                    (0, 1, 0, 0, 1, 0, 0, 1, 0, 0))
        #self.syntax['angtors'] = (([1,2,3,4], [5], [6], [7], [8], [9], [10]),
                                    #(int, float, float, float, float, float, float), 
                                    #(0, 1, 1, 1, 1, 1, 1))
        self.syntax['angtors'] = (([1,2,3,4], (5, 11)),
                                    (int, float ), 
                                    (0, 1))
        self.syntax['strtors'] = (([1,2,3,4], (5, 14)),
                                    (int, float), 
                                    (0, 1))
        self.syntax['pitors'] = (([1,2], [3], ),
                                    (int, float), 
                                    (0, 1))
        self.syntax['ureybrad'] = (([1,2, 3], [4], [5], ),
                                    (int, float, float), 
                                    (0, 1, 0))
        self.syntax['vdwpr'] = (([1,2], [3, 4], range(5,13)),
                                    (int, float, float), 
                                    (0, 1, 0))

    def update_atomclass(self):
        self.d_type2class = {}
        if 'atom' not in self.params:
            return
        for iden, value in (self.params['atom']).items():
            self.d_type2class[iden] = tuple(value[0])

    def read_prm(self, prmfile):
        with open(prmfile, 'r') as fin:
            iline = -1
            ilinemtp = -1
            inheader = True
            self.header = ''
            for line in fin:
                iline = iline + 1
                #w = line.split()
                w = line.split('#')[0].split()
                if len(w) == 0:
                    continue
                entry = w[0]
                if entry in self.syntax:
                    inheader = False
                if inheader and not line.startswith('##'):
                    self.header = self.header + line
                if entry == 'multipole':
                    ilinemtp = iline
                if (ilinemtp>=0) and (iline - ilinemtp in range(1,5)):
                    currw.extend(w)
                else:
                    currw = w[:]

                ncurrw = len(currw)
                endmtp = (ilinemtp >=0 and (iline - ilinemtp) == 4)
                if endmtp or (entry != 'multipole' and entry in self.syntax):
                    if endmtp:
                        curre = 'multipole'
                    else:
                        curre = entry
                    currs = self.syntax[curre]
                    currd = []
                    for ii in range(len(currs[0])):
                        iws = currs[0][ii]
                        tp = currs[1][ii]
                        if type(iws) == tuple and len(iws) == 2:
                            #i0 = iws[0]%(ncurrw)
                            #i1 = iws[1]%(ncurrw)
                            i0 = iws[0]
                            i1 = iws[1]
                            if i0 < 0:
                                i0 += ncurrw
                            if i1 < 0:
                                i1 += ncurrw
                            iws = range(i0, i1)
                        currd.append(tuple([tp(currw[K])
                                      for K in iws if K < ncurrw ]))
                    self.params[curre][tuple(currd[0])] = tuple(currd[1:])
        self.update_atomclass()

    def check_filename(self, outfile):
        newfile = outfile
        if os.path.exists(outfile):
            for iv in range(999,0,-1):
                newfile = outfile + '.#.%d'%iv
                if os.path.exists(newfile):
                    newfile = outfile + '.#.%d'%(iv+1)
                    break
        return newfile

    def import_prm(self, pnew, append_polgrp=False):
        '''update parameters'''
        conflict = []
        for term in pnew:
            overlap = set(self.params[term]) & set(pnew[term])
            conflict.append((term, overlap))

            if term not in self.params:
                self.params[term] = {}

            if append_polgrp and term == 'polarize':
                for iden in pnew[term]:
                    if iden in self.params[term]:
                        v0 = list(self.params[term][iden])
                        v1 = list(pnew[term][iden])
                        print('Check', iden, 'V0', v0, 'V1', v1)
                        v = v1[:-1] + [sorted(list(set(v0[-1]) | set(v1[-1])))]
                        self.params[term][iden] = v
                    else:
                        self.params[term][iden] = pnew[term][iden]
            else:
                self.params[term].update(pnew[term])
        for term, overlap in conflict:
            if len(overlap) > 0:
                print('%s: overwriting parameters for %s'
                      %(term, ', '.join([' '.join(map(str, K)) for K in (overlap)])))
        self.update_atomclass()

    def atom_range(self, grp):
        sgrp = list(sorted(grp))
        curr_blk = []
        out_idx = []
        for idx in sgrp:
            if len(curr_blk) == 0:
                curr_blk.append(idx)
            elif idx == curr_blk[-1] + 1:
                curr_blk.append(idx)
            else:
                if len(curr_blk) > 2:
                    out_idx.extend([-curr_blk[0], curr_blk[-1]])
                else:
                    out_idx.extend(curr_blk)
                curr_blk = [idx]
        if len(curr_blk) > 2:
            out_idx.extend([-curr_blk[0], curr_blk[-1]])
        else:
            out_idx.extend(curr_blk)
        return out_idx

    def connect_param(self, tmap, cmap, breakable=[], breakgrp=[]):
        '''
        Combine residue parameters
        '''
        self.oldparams = dict(self.params)
        self.params = {}
        self.params.update(copy.deepcopy(self.oldparams))

        breakbond = breakable

        # bond that connects new and old types
        mixed_b = []
        if len(breakbond) > 0:
            for iden in breakbond:
                if len(iden) == 1 and 'bond' in self.oldparams:
                    for iden1 in self.oldparams['bond']:
                        if iden[0] in iden1:
                            mixed_b.append((iden1[0], iden1[1]))

                elif len(iden) >= 2:
                    for ii in range(len(iden)-1):
                        mixed_b.append((iden[ii], iden[ii+1]))
        for iden in mixed_b:
            iden1 = (iden[1], iden[0])
            if iden1 not in mixed_b:
                mixed_b.append((iden[1], iden[0]))

        if 'torsion' in self.oldparams:
            term = 'torsion'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                if len(set(iden) & set(imap)) == 0:
                    continue
                for bkbnd in mixed_b:
                    bk_pt = []
                    # first atom index in a mixed bond
                    for iid in range(len(iden)-1):
                        if tuple(iden[iid:iid+2]) == bkbnd:
                            bk_pt.append(iid)

                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    for iid in bk_pt:
                        curr_id = tuple(list(iden[0:iid+1]) + list(iden0[iid+1:]))
                        idens.append(curr_id)
                        curr_id = tuple(list(iden0[0:iid+1]) + list(iden[iid+1:]))
                        idens.append(curr_id)
                    for curr_id in idens:
                        if curr_id not in self.oldparams[term]:
                            self.params[term][curr_id] = tuple(value)

        if 'angle' in self.oldparams:
            term = 'angle'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                if len(set(iden) & set(imap)) == 0:
                    continue
                for bkbnd in mixed_b:
                    bk_pt = []
                    # first atom index in a mixed bond
                    for iid in range(len(iden)-1):
                        if tuple(iden[iid:iid+2]) == bkbnd:
                            bk_pt.append(iid)

                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    for iid in bk_pt:
                        curr_id = tuple(list(iden[0:iid+1]) + list(iden0[iid+1:]))
                        idens.append(curr_id)
                        curr_id = tuple(list(iden0[0:iid+1]) + list(iden[iid+1:]))
                        idens.append(curr_id)
                    for curr_id in idens:
                        if curr_id not in self.oldparams[term]:
                            self.params[term][curr_id] = tuple(value)

        if 'bond' in self.oldparams:
            term = 'bond'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])
                # bond value does not contain atom types

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1]))
                        idens.append((iden0[0], iden[1]))
                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)

        # build bonds between atom types
        grp_bonds = []
        mixed_gmap = {}
        for iden in self.oldparams['polarize']:
            atomi = iden[0]
            value = self.oldparams['polarize'][iden]
            if atomi not in mixed_gmap:
                mixed_gmap[atomi] = set([])
            for atomj in value[-1]:
                bond = (atomi, atomj)
                grp_bonds.append(bond)
                mixed_gmap[atomi].add(atomj)
        # build bonds from multiple def
        mtp_bonds = []
        for iden in self.oldparams['multipole']:
            if iden[1] > 0 and iden[2] > 0:
                # Z-then-X
                mtp_bonds.append((iden[0], iden[1]))
                mtp_bonds.append((iden[1], iden[2]))
            elif iden[1] < 0 and iden[2] < 0:
                # Z-then-X
                mtp_bonds.append((iden[0], iden[1]))
                mtp_bonds.append((iden[0], iden[2]))
            elif iden[1] != 0 and iden[2] != 0:
                mtp_bonds.append((iden[0], iden[1]))
        for iden in mtp_bonds:
            iden1 = (iden[1], iden[0])
            if iden1 not in mtp_bonds:
                mtp_bonds.append((iden[1], iden[0]))
        # list of polarization group bonds
        mixed_g = []
        if len(breakgrp) > 0:
            for iden in breakgrp:
                if len(iden) == 1:
                    for iden1 in grp_bonds:
                        if iden[0] in iden1:
                            mixed_g.append((iden1[0], iden1[1]))

                elif len(iden) >= 2:
                    for ii in range(len(iden)-1):
                        mixed_g.append((iden[ii], iden[ii+1]))
        for iden in mixed_g:
            iden1 = (iden[1], iden[0])
            if iden1 not in mixed_g:
                mixed_g.append((iden[1], iden[0]))
        print('MIXED BOND', mixed_g)

        if 'polarize' in self.oldparams:
            term = 'polarize'
            imap = tmap
            synt = self.syntax[term]
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                #v = value[-1]
                #v1 = [np.sign(K)*imap[np.abs(K)] for K in v if np.abs(K) in imap]
                for ii in range(len(value)):
                    v = value[ii]
                    sx = synt[1][ii+1]
                    if sx == int:
                        value[ii] = list(v) + [np.sign(K)*imap[np.abs(K)] for K in v if np.abs(K) in imap]
                        value[ii] = list(sorted(set(value[ii])))

                if True or len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = [iden, iden0]
                    for curr_g in mixed_g:
                        if iden[0] == curr_g[0] and curr_g[1] in value[-1]:
                            print('Updating:', iden, value)
                            self.params[term][iden] = tuple(value)
                            self.params[term][iden0] = tuple(value)

        if 'multipole' in self.oldparams:
            term = 'multipole'
            imap = tmap
            synt = self.syntax[term]
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                if len(set(iden) & set(imap)) > 0 and True:
                    # convert all the atom ids into the smallest id in the group
                    cum_last = []
                    cum_layer = []
                    curr_layer = []
                    for ii in range(len(iden)):
                        K = iden[ii]
                        if np.abs(K) in imap:
                            curr_layer = (K, np.sign(K)*imap[np.abs(K)])
                        else:
                            curr_layer = (K, )
                        if len(cum_last) == 0:
                            cum_layer = [(K,) for K in curr_layer]
                        else:
                            cum_layer = []
                            for KK in cum_last:
                                cum_layer.extend([list(KK)+[K3] for K3 in curr_layer])
                                print('EXTEND', cum_layer)
                                
                            #cum_layer = [tuple(list(cum_last)+[KK]) for KK in curr_layer]
                        cum_last = tuple(cum_layer)
                        print('II', ii, cum_last)
                    #print('ALL POSS', cum_layer)
                    for iden1 in cum_layer:
                        K = tuple(iden1)
                        self.params[term][K] = tuple(value)
        self.update_atomclass()

    def mix_param(self, tmap, cmap, breakable=[], breakgrp=[]):
        '''
        Combine bonded parameters, multipole and polarize definations
        '''
        self.oldparams = dict(self.params)
        self.params = {}

        for term in self.termlist:
            self.params[term] = {}
            currtab = {}
            synt = self.syntax[term]

            if term in self.use_atype:
                # use atom type
                imap = tmap
            else:
                # use atom class
                imap = cmap

            tp_iden = synt[1][0]
            mutatable = (tp_iden == int)

            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                for ii in range(len(value)):
                    v = value[ii]
                    sx = synt[1][ii+1]

                    if term == 'atom' and ii == 0:
                        # atom term use both atom type and atom class
                        value[ii] = [cmap[K] if (K in cmap) else K for K in v]
                    elif sx == int:
                        value[ii] = [np.sign(K)*imap[np.abs(K)] if (np.abs(K) in imap)  else K for K in v]


                print("MUTA", mutatable)
                if len(set(iden) & set(imap)) > 0 and mutatable:
                    # convert all the atom ids into the smallest id in the group
                    cum_last = []
                    cum_layer = []
                    curr_layer = []
                    for ii in range(len(iden)):
                        K = iden[ii]
                        if np.abs(K) in imap:
                            curr_layer = (K, np.sign(K)*imap[np.abs(K)])
                        if len(cum_last) == 0:
                            cum_layer = ((curr_layer[0],),(curr_layer[1],))
                        else:
                            cum_layer = [tuple(list(cum_last)+[KK]) for KK in curr_layer]
                        cum_last = list(cum_layer)
                    #iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    #self.params[term][iden0] = tuple(value)
                    print("ALL POSSIBLE", cum_layer)
                else:
                    self.params[term][iden] = tuple(value)
                    continue

        # list of bonds that may contain mixed types
        mixed_b = []
        if len(breakable) > 0:
            for iden in breakable:
                if len(iden) == 1 and 'bond' in self.oldparams:
                    for iden1 in self.oldparams['bond']:
                        if iden[0] in iden1:
                            mixed_b.append((iden1[0], iden1[1]))

                elif len(iden) >= 2:
                    for ii in range(len(iden)-1):
                        mixed_b.append((iden[ii], iden[ii+1]))
        else:
            if 'torsion' in self.oldparams:
                term = 'torsion'
                imap = cmap
                for iden in self.oldparams[term]:
                    #mixed_b.append(tuple(sorted([iden[1], iden[2]])))
                    mixed_b.append((iden[1], iden[2]))
        for iden in mixed_b:
            iden1 = (iden[1], iden[0])
            if iden1 not in mixed_b:
                mixed_b.append((iden[1], iden[0]))

        if 'torsion' in self.oldparams:
            term = 'torsion'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []

                    if (iden[1], iden[2]) in mixed_b:
                        idens.append((iden[0], iden[1], iden0[2], iden0[3]))
                        idens.append((iden0[0], iden0[1], iden[2], iden[3]))
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1], iden0[2], iden0[3]))
                        idens.append((iden0[0], iden[1], iden[2], iden[3]))
                    if (iden[2], iden[3]) in mixed_b:
                        idens.append((iden[0], iden[1], iden[2], iden0[3]))
                        idens.append((iden0[0], iden0[1], iden0[2], iden[3]))

                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)

        if 'angle' in self.oldparams:
            term = 'angle'
            imap = cmap
            for iden in self.oldparams[term]:

                value = list(self.oldparams[term][iden])
                # angle value does not contain atom types

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1], iden0[2]))
                        idens.append((iden0[0], iden[1], iden[2]))
                    if (iden[1], iden[2]) in mixed_b:
                        idens.append((iden[0], iden[1], iden0[2]))
                        idens.append((iden0[0], iden0[1], iden[2]))
                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)
        if 'bond' in self.oldparams:
            term = 'bond'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])
                # bond value does not contain atom types

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1]))
                        idens.append((iden0[0], iden[1]))
                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)

        # build bonds between atom types
        grp_bonds = []
        mixed_gmap = {}
        for iden in self.oldparams['polarize']:
            atomi = iden[0]
            value = self.oldparams['polarize'][iden]
            if atomi not in mixed_gmap:
                mixed_gmap[atomi] = set([])
            for atomj in value[-1]:
                bond = (atomi, atomj)
                grp_bonds.append(bond)
                mixed_gmap[atomi].add(atomj)
        # list of polarization group bonds
        mixed_g = []
        if len(breakgrp) > 0:
            for iden in breakgrp:
                if len(iden) == 1:
                    for iden1 in grp_bonds:
                        if iden[0] in iden1:
                            mixed_g.append((iden1[0], iden1[1]))

                elif len(iden) >= 2:
                    for ii in range(len(iden)-1):
                        mixed_b.append((iden[ii], iden[ii+1]))
        for iden in mixed_g:
            iden1 = (iden[1], iden[0])
            if iden1 not in mixed_g:
                mixed_g.append((iden[1], iden[0]))

        if 'polarize' in self.oldparams:
            term = 'polarize'
            imap = tmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                #v = value[-1]
                #v1 = [np.sign(K)*imap[np.abs(K)] for K in v if np.abs(K) in imap]
                for ii in range(len(value)):
                    v = value[ii]
                    sx = synt[1][ii+1]
                    if sx == int:
                        value[ii] = list(v) + [np.sign(K)*imap[np.abs(K)] for K in v if np.abs(K) in imap]
                        value[ii] = list(sorted(set(value[ii])))

                if True or len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = [iden, iden0]
                    for curr_g in mixed_g:
                        if iden[0] == curr_g[0] and curr_g[1] in value[-1]:
                            print('Updating:', iden, value)
                            self.params[term][iden] = tuple(value)
                            self.params[term][iden0] = tuple(value)

        self.update_atomclass()
        pass

    def mix_bond(self, start=(401, 401), shift=0, shiftc=0, breakable=[]):

        # Based on function `renumber'
        # Renumber the atom type while creating bond, angle and torsion with mixed old/new types

        t2c = list(self.d_type2class.items())
        t2c.sort()
        oldt = list(set([K[0][0] for K in t2c]))
        oldc = list(set([K[1][0] for K in t2c]))
        oldt.sort()
        oldc.sort()
        if shift == 0:
            newt = range(start[0], start[0]+len(oldt))
            newc = range(start[1], start[1]+len(oldc))
        else:
            newt = [K+shift for K in oldt]
            if shiftc == 0:
                shiftc = shift
            newc = [K+shiftc for K in oldc]
        tmap = dict(zip(oldt, newt))
        cmap = dict(zip(oldc, newc))

        self.oldparams = dict(self.params)
        self.params = {}


        for term in self.termlist:
            self.params[term] = {}
            currtab = {}
            synt = self.syntax[term]

            if term in self.use_atype:
                # use atom type
                imap = tmap
            else:
                # use atom class
                imap = cmap

            tp_iden = synt[1][0]
            mutatable = (tp_iden == int)

            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                for ii in range(len(value)):
                    v = value[ii]
                    sx = synt[1][ii+1]

                    if term == 'atom' and ii == 0:
                        # atom term use both atom type and atom class
                        value[ii] = [cmap[K] if (K in cmap) else K for K in v]
                    elif sx == int:
                        value[ii] = [np.sign(K)*imap[np.abs(K)] if (np.abs(K) in imap)  else K for K in v]

                if len(set(iden) & set(imap)) > 0 and mutatable:
                    # convert all the atom ids into the smallest id in the group
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    self.params[term][iden0] = tuple(value)
                else:
                    self.params[term][iden] = tuple(value)
                    continue

        # list of bonds that may contain mixed types
        mixed_b = []
        if len(breakable) > 0:
            for iden in breakable:
                if len(iden) == 1 and 'bond' in self.oldparams:
                    for iden1 in self.oldparams['bond']:
                        if iden[0] in iden1:
                            mixed_b.append((iden1[0], iden1[1]))

                elif len(iden) >= 2:
                    for ii in range(len(iden)-1):
                        mixed_b.append((iden[ii], iden[ii+1]))
        else:
            if 'torsion' in self.oldparams:
                term = 'torsion'
                imap = cmap
                for iden in self.oldparams[term]:
                    #mixed_b.append(tuple(sorted([iden[1], iden[2]])))
                    mixed_b.append((iden[1], iden[2]))
        for iden in mixed_b:
            iden1 = (iden[1], iden[0])
            if iden1 not in mixed_b:
                mixed_b.append((iden[1], iden[0]))

        if 'torsion' in self.oldparams:
            term = 'torsion'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []

                    if (iden[1], iden[2]) in mixed_b:
                        idens.append((iden[0], iden[1], iden0[2], iden0[3]))
                        idens.append((iden0[0], iden0[1], iden[2], iden[3]))
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1], iden0[2], iden0[3]))
                        idens.append((iden0[0], iden[1], iden[2], iden[3]))
                    if (iden[2], iden[3]) in mixed_b:
                        idens.append((iden[0], iden[1], iden[2], iden0[3]))
                        idens.append((iden0[0], iden0[1], iden0[2], iden[3]))

                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)

        if 'angle' in self.oldparams:
            term = 'angle'
            imap = cmap
            for iden in self.oldparams[term]:

                value = list(self.oldparams[term][iden])
                # angle value does not contain atom types

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1], iden0[2]))
                        idens.append((iden0[0], iden[1], iden[2]))
                    if (iden[1], iden[2]) in mixed_b:
                        idens.append((iden[0], iden[1], iden0[2]))
                        idens.append((iden0[0], iden0[1], iden[2]))
                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)
        if 'bond' in self.oldparams:
            term = 'bond'
            imap = cmap
            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])
                # bond value does not contain atom types

                if len(set(iden) & set(imap)) > 0:
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    idens = []
                    if (iden[0], iden[1]) in mixed_b:
                        idens.append((iden[0], iden0[1]))
                        idens.append((iden0[0], iden[1]))
                    for iden1 in idens:
                        self.params[term][iden1] = tuple(value)

        self.update_atomclass()


    def renumber(self, start=(401, 401), shift=0, shiftc=0):
        t2c = list(self.d_type2class.items())
        t2c.sort()
        oldt = list(set([K[0][0] for K in t2c]))
        oldc = list(set([K[1][0] for K in t2c]))
        oldt.sort()
        oldc.sort()
        if shift == 0:
            newt = range(start[0], start[0]+len(oldt))
            newc = range(start[1], start[1]+len(oldc))
        else:
            newt = [K+shift for K in oldt]
            if shiftc == 0:
                shiftc = shift
            newc = [K+shiftc for K in oldc]
        tmap = dict(zip(oldt, newt))
        cmap = dict(zip(oldc, newc))

        self.oldparams = dict(self.params)
        self.params = {}

        # adapted form func `combine_type'
        # only int type is renumbered
        for term in self.termlist:
            self.params[term] = {}
            currtab = {}
            synt = self.syntax[term]

            if term in self.use_atype:
                # use atom type
                imap = tmap
            else:
                # use atom class
                imap = cmap

            tp_iden = synt[1][0]
            mutatable = (tp_iden == int)

            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                for ii in range(len(value)):
                    v = value[ii]
                    sx = synt[1][ii+1]

                    if term == 'atom' and ii == 0:
                        # atom term use both atom type and atom class
                        value[ii] = [cmap[K] if (K in cmap) else K for K in v]
                    elif sx == int:
                        value[ii] = [np.sign(K)*imap[np.abs(K)] if (np.abs(K) in imap)  else K for K in v]

                if len(set(iden) & set(imap)) > 0 and mutatable:
                    # convert all the atom ids into the smallest id in the group
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if np.abs(K) in imap else K for K in iden])
                    self.params[term][iden0] = tuple(value)
                else:
                    self.params[term][iden] = tuple(value)
                    continue
        self.update_atomclass()

    def write_prm(self, outfile='tinker.key'):
        outfile = self.check_filename(outfile)
        outp = self.header + ''
        for term in self.termlist:
            if term not in self.params:
                continue
            synt = self.syntax[term]
            currp = self.params[term]
            idlist = list(currp)
            idlist.sort(key = lambda t: (t[-3:], t))
            if term in ('strbnd',):
                idlist.sort(key = lambda t: (t[1], t[2], t[0]))
            outp = outp + '\n## %s\n\n'%term.upper()

            if term in self.outfmt:
                currfmt = self.outfmt[term]
            else:
                currfmt = 0
            for iden in idlist:
                currw = []
                value = [iden] + list(currp[iden])
                for ii in range(len(value)):
                    sx = synt[1][ii]
                    v = value[ii]
                    if sx in self.outfmt:
                        fieldfmt = self.outfmt[sx]
                        currw.append(' '.join([fieldfmt%K for K in v]))
                    else:
                        currw.append(' '.join(map(str, v)))
                if currfmt == 0:
                    outp = outp + term + ' ' + ' '.join(currw) + '\n'
                else:
                    outp = outp + currfmt%tuple(currw)
        fout = open(outfile, 'w')
        fout.write(outp)
        fout.close()
        
    def avg_prm(self, prms):
        '''
        update the params as the averge of [self.params prms]
        only try to do average for an entry if it appears in current params
        '''
        nshift = 10000
        nnewp = len(prms)
        atypes = list(self.d_type2class)
        for prm in prms:
            self.renumber(shift=nshift)
            self.import_prm(prm)
        self.combine_type([range(K[0], K[0]+nnewp*nshift+1, nshift) for K in atypes])

    
    def create_ghost(self, type=800):
        iden = (type, )
        pass

    def mutate_type(self, ele_lambda = 1.0, vdw_lambda = 1.0, tors_lambda = 1.0):
        pass
        fep_lambda = {'vdw':vdw_lambda, 'vdwpr':vdw_lambda, 'multipole':ele_lambda, 'polarize':ele_lambda, 'torsion':tors_lambda}
        for term in fep_lambda:
            curr_lam = fep_lambda[term]
            for iden in self.params[term]:
                self.params[term][iden] = list(self.params[term][iden])
                for ii, dofep in enumerate(self.syntax[term][2][1:]):
                    if dofep:
                        value0 = self.params[term][iden][ii]
                        self.params[term][iden][ii] = tuple(K*curr_lam for K in value0)

    def mutate_type2(self, ele_lambda = 1.0, vdw_lambda = 1.0, tors_lambda = 1.0):

        self.params['vdwpr'] = {}

        term = 'vdw'
        for iden in self.params[term]:
            value0 = self.params[term][iden][0]
            self.params[term][iden] = list(self.params[term][iden])
            self.params[term][iden][0] = (K*vdw_lambda for K in value0)
        term = 'multipole'
        for iden in self.params[term]:
            self.params[term][iden] = list(self.params[term][iden])
            for ii in range(0,5):
                value0 = self.params[term][iden][ii]
                self.params[term][iden][ii] = (K*ele_lambda for K in value0)
        term = 'polarize'
        for iden in self.params[term]:
            self.params[term][iden] = list(self.params[term][iden])
            value0 = list(self.params[term][iden][0])
            value0[0] *= ele_lambda
            self.params[term][iden][0] = value0
        term = 'torsion'
        for iden in self.params[term]:
            self.params[term][iden] = list(self.params[term][iden])
            for ii in [0, 3, 6]:
                if ii >= len(self.params[term][iden]):
                    break
                value0 = list(self.params[term][iden][ii])
                self.params[term][iden][ii] = tuple(K*tors_lambda for K in value0)

    def combine_type(self, groups=[], start=0):
        '''
        Combine atom types. All atom types in a group are mapped to the smallest type among them
        If `start > 0' and `groups' cover all types, then the new atom types are determined 
        according to the order of appearance in `groups', starting from `start'.
        '''
        # map ids in a group to the smallest id in that group
        #type map and class map
        tmap = {}
        cmap = {}
        ngrouptype = sum([len(K) for K in groups])
        for ii, g in enumerate(groups):
            if start > 0 and ngrouptype == len(self.d_type2class):
                gm = start + ii
                cgm = start + ii
            else:
                gm = min(g)
                cgm = self.d_type2class[(gm,)][0]
            for i in g:
                tmap[i] = gm
                #tmap[-i] = -gm
                ci = self.d_type2class[(i,)][0]
                cmap[ci] = cgm
        self.oldparams = dict(self.params)
        self.params = {}

        # lookup table that contains group params
        lutable = {}
        for term in self.termlist:
            self.params[term] = {}
            currtab = {}
            synt = self.syntax[term]

            tp_iden = synt[1][0]
            mutatable = (tp_iden == int)

            if term in self.use_atype:
                # use atom type
                imap = tmap
            else:
                # use atom class
                imap = cmap

            for iden in self.oldparams[term]:
                value = list(self.oldparams[term][iden])

                if len(set(iden) & set(imap)) > 0:
                    # convert all the atom ids into the smallest id in the group
                    iden0 = tuple([np.sign(K)*imap[np.abs(K)] if (np.abs(K) in imap) and mutatable else K for K in iden])
                else:
                    self.params[term][iden] = tuple(value)
                    continue

                for ii in range(len(value)):
                    v = value[ii]
                    sx = synt[1][ii+1]

                    if term == 'atom' and ii == 0:
                        # atom term use both atom type and atom class
                        value[ii] = [cmap[K] if (K in cmap)  else K for K in v]
                    elif sx == int:
                        #value[ii] = [imap[K] if K in imap else K for K in v]
                        value[ii] = [np.sign(K)*imap[np.abs(K)] if (np.abs(K) in imap) else K for K in v]

                if iden0 in currtab:
                    value0, nsum = currtab[iden0]
                        
                    valuenew = []
                    dismatch = False
                    for ii in range(len(value0)):
                        v0 = value0[ii]
                        v = value[ii]
                        sx = synt[1][ii+1]
                        if not sx in self.floats:
                            # fields other than float must match in two entries
                            if v0 != v:
                                dismatch = True
                                break
                            vnew = tuple(v0)
                        else:
                            vnew = tuple([(v0[K]*nsum + v[K])/(nsum+1) for K in range(len(v0))] )
                        valuenew.append(vnew)
                    if dismatch:
                        continue
                    else:
                        # store current value and weight
                        currtab[iden0] = (tuple(valuenew), nsum+1)
                        
                else:
                    currtab[iden0] = (value, 1)
            lutable[term] = {}
            for iden in currtab:
                lutable[term][iden] = currtab[iden][0]
            self.params[term].update(lutable[term])
        self.update_atomclass()

def demo():
    p = TinkerParam0()
    p.read_prm('../../ttt.key')
    p.write_prm('reference.key')
    p.combine_type([(402,403), (405, 406), (408, 409), (410, 411)])
    #p.renumber((401,401))
    p.write_prm('combined.key')
    p2 = TinkerParam0()
    p2.read_prm('tinker.key')
    p2.combine_type([(402,403), (405, 406), (408, 409), (410, 411)])
    #p2.renumber(shift=1000)
    #p.import_prm(p2.params)
    #p.write_prm('double.key')

    p.avg_prm([dict(p2.params), dict(p2.params)])
    p.write_prm('avgp.key')

def demo2():
    p = TinkerParam0()
    p.read_prm('tinker-1.key')
    p.write_prm('reference-1.key')
    p.combine_type([(402,403), (405, 406), (408, 409), (410, 411)])
    p.write_prm('combined-1.key')
    p.renumber((401,401))
    p.write_prm('reorder-1.key')



#demo()
    

