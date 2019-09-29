
from geom_io import GeomFile, GeomConvert, debug_w
import geom_io
from utils import align_slow, int_to_xyz
import os
from string import Template
import numpy as np
import scipy.optimize

class QMMethod():
    def __init__(self, program='', tplt_file='', keywords=[]):
        self.program = program
        self.tplt_file = tplt_file
        self.keywords = keywords


class QMInput(GeomConvert):
    def __init__(self):
        super(QMInput, self).__init__()
        #pkgpath = os.path.dirname(os.path.abspath(__file__))
        self.pkgpath = os.path.dirname((__file__))
        self.template_path = os.path.join(self.pkgpath, 'qm_template', 'template.list')
        self.QM_WRITE = {'psi4':self.write_psi4, 'g16':self.write_g16}
        self.QM_PROGRAMS = set(self.QM_WRITE)
        self.memory = 20
        self.numproc = 8
        self.disk = 100

    def get_template(self, flist = None):
        if flist == None:
            flist = self.template_path
        template_dir = os.path.dirname(flist)

        methods = {}
        with open(flist, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                w = (line.split('#')[0]).split()
                if len(w) < 4:
                    continue
                program = w[1]
                tplt_file = os.path.join(template_dir, w[2])
                kws = w[3:]
                if program not in self.QM_PROGRAMS:
                    continue
                if not os.path.isfile(tplt_file):
                    continue
                methods[w[0]] = QMMethod(program, tplt_file, kws)

        self.methods = methods

    def get_default(self):
        kws = {}
        kws['memory'] = '%d'%self.memory
        kws['numproc'] = '%d'%self.numproc
        kws['disk'] = '%d'%self.disk
        return kws

    def convert_fmt_string(self, fmt, reverse=False):
        '''
        not used
        '''
        sym1 = '<| |> { }'.split()
        sym2 = '{ } <! !>'.split()
        if reverse:
            fmt_dict = dict(zip(sym2, sym1))
        else:
            fmt_dict = dict(zip(sym1, sym2))

    def print_all_methods(self):
        outp = 'Available methods (%d)\n'%(len(self.methods))
        outp += '-'*8 + '\n'
        for theory in self.methods:
            method = self.methods[theory]
            outp += '%s(%s)\n'%(theory, method.program)
        outp += '-'*8 + '\n'
        print(outp)

    def write_qm(self, outf, theory):
        if theory in self.methods:
            method = self.methods[theory]
            if method.program in self.QM_WRITE:
                self.QM_WRITE[method.program](outf, method)
        else:
            self.print_all_methods()


    def fill_missing(self):
        if len(self.group_idx) == 0:
            self.group_idx = [list(range(1, self.top_natoms+1))]

        nmulti_t = [0, 0]
        for igrp, grp in enumerate(self.group_idx):
            if len(self.multiplicity) > igrp:
                nmulti = self.multiplicity[igrp]
            else:
                nmulti = (0, 1)
                self.multiplicity.append(nmulti)
            nmulti_t[0] += nmulti[0]
            nmulti_t[1] += nmulti[1] - 1
        nmulti_t[1] += 1
        self.multiplicity.append(nmulti_t)

    def write_g16(self, outf, method: QMMethod):
        if not isinstance(method, QMMethod): 
            raise TypeError
        self.fill_missing()
        frags = []
        molecule = ''
        ngrps = len(self.group_idx)
        frags = ['%d %d '%tuple(self.multiplicity[-1])]

        for igrp, grp in enumerate(self.group_idx):
            nmulti = self.multiplicity[igrp]
            frags.append('%d %d '%tuple(nmulti))

        frags.append('\n')
        for igrp, grp in enumerate(self.group_idx):
            if ngrps > 1:
                grp_s = '(Fragment=%d)'%(igrp+1)
            else:
                grp_s = ''
            for iatom in grp:
                i = iatom-1
                frags[-1] += '%s%s %12.4f %12.4f %12.4f\n'%(self.top_name[iatom], grp_s, self.coord[i][0], self.coord[i][1], self.coord[i][2])
        molecule = ''.join(frags)
        outf2 = '.'.join(outf.split('.')[:-1]) + '.chk'
        fmts = self.get_default()
        fmts['geometry'] = molecule
        fmts['chkfile'] = outf2
        if len(set(fmts) & set(method.keywords)) == len(set(method.keywords)):
            with open(method.tplt_file) as fin:
                outp = Template(fin.read())
                with open(outf, 'w') as fout:
                    fout.write(outp.safe_substitute(fmts))

    def write_psi4(self, outf, method: QMMethod):
        if not isinstance(method, QMMethod): 
            raise TypeError
        self.fill_missing()
        frags = []
        molecule = ''
        for igrp, grp in enumerate(self.group_idx):
            nmulti = self.multiplicity[igrp]
            frags.append('%d %d\n'%tuple(nmulti))
            for iatom in grp:
                i = iatom - 1
                frags[-1] += '%s %12.4f %12.4f %12.4f\n'%(self.top_name[iatom], self.coord[i][0], self.coord[i][1], self.coord[i][2])
        molecule = '--\n'.join(frags)
        fmts = self.get_default()
        fmts['geometry'] = molecule
        if len(set(fmts) & set(method.keywords)) == len(set(method.keywords)):
            with open(method.tplt_file) as fin:
                outp = Template(fin.read())
                with open(outf, 'w') as fout:
                    fout.write(outp.safe_substitute(fmts))


class Assemble(QMInput):
    def __init__(self):
        super(Assemble, self).__init__()

    def add_mol(self, geo, idx1, idx2, r0=3.0, mode='face'):
        r'''
        mode: 'face' | 'vertical' | 'T-shape' | 'parallel'

              'face': 
              i3            j3
                \          /
                 i1  ...  j1
                /          \
              i2            j2


              'vertical': 
              i3              
                \           .j3
                 i1  ...  j1   
                /           `j2
              i2              


              'T-shape': 
                   i3              
                  /     
                 i1    .j3
                  \ ` j1    
                   i2  `j2         


              'parallel': 
                   i3         
                  /     j3
                 i1    / 
                  \ ` j1
                   i2  \     
                        j2
                               
        '''
        a1 = self.measure([idx1[1], idx1[0], idx1[2]])
        a2 =  geo.measure([idx2[1], idx2[0], idx2[2]])

        d0 = 179.0
        b0 = 1.5
        if mode == 'face':
            r_int = [[idx1[0], r0, idx1[1], 180-0.5*a1, idx1[2], d0], [-1, b0, idx1[0], 180-0.5*a2, idx1[1], 1.0], [-1, b0, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])
        elif mode == 'vertical':
            r_int = [[idx1[0], r0, idx1[1], 180-0.5*a1, idx1[2], d0], [-1, b0, idx1[0], 180-0.5*a2, idx1[1], 90.0], [-1, b0, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])
        elif mode == 'T-shape':
            r_int = [[idx1[0], r0, idx1[1], 90.0, idx1[2], 90.0], [-1, b0, idx1[0], 180-0.5*a2, idx1[1], 90.0-0.5*a1], [-1, b0, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])
        elif mode == 'parallel':
            r_int = [[idx1[0], r0, idx1[1], 90.0, idx1[2], 90.0], [-1, b0, idx1[0], 90.0, idx1[1], 0], [-1, b0, idx1[0], 90.0, idx1[2], 0.0]]
            self.add_mol_int(geo, r_int, idx2[:3])
        else:
            print(self.add_mol.__doc__)

    def add_mol_int(self, geo: GeomFile, r_int, anchor_idx):
        '''
        Call GeomConvert.append_frag to append a fragment
        Then apply geometry operations
        
        r_int: [i1, B1, i2, A1, i3, D1], ...
               i < 0 indicates the index |i| in the new internel coord

        anchor_idx: the indices of atoms in the new mol to be aligned to the internel coord
        '''
        if not isinstance(geo, GeomFile): 
            raise TypeError
        n1 = self.top_natoms
        n2 = geo.top_natoms
        n2a = len(r_int)
        if n2a != len(anchor_idx):
            return
        for i2 in anchor_idx:
            if i2 > n2:
                return
        anchor_i = [K-1 for K in anchor_idx]
        for i2 in range(n2a):
            for ii in range(0, len(r_int[i2]), 2):
                if r_int[i2][ii] < 0:
                    # convert monomer atom index to dimer atom index
                    i_new = (-r_int[i2][ii]) + n1
                    if i_new >= n1+n2a:
                        return
                    r_int[i2][ii] = i_new

        self.append_frag(geo)
        for iframe in range(self.nframes):
            coord = self.frames[iframe]
            rdimer_int = list(coord[:n1]) + r_int
            rdimer_car = int_to_xyz(rdimer_int)
            xyz2 = align_slow(coord[n1:], rdimer_car[n1:],anchor_i,  list(range(n2a)))
            coord[n1:] = xyz2

if __name__ == '__main__':
    #qmfile = QMInput()
    #qmfile.get_template()
    #qmfile.read_input('examples/mol.xyz', ftype='tinker')
    #qmfile.read_input('examples/mol.xyz', ftype='tinker')
    #qmfile.read_input('examples/4_Thiouracil-Water_1.xyz')
    #q2 = QMInput()
    #q2.read_input('examples/4_Thiouracil-Water_1.xyz')
    #qmfile.append_frag(qmfile)
    ##print(qmfile.top_natoms)
    ##print(qmfile.group_idx)
    #
    ##qmfile.write_qm('2.psi4', 'sapt2/adz')
    ##qmfile.group_idx = [list(range(1, 5)), list(range(5, qmfile.top_natoms+1))]
    #qmfile.write_qm('3.psi4', 'sapt2/adz')
    ##qmfile.write_qm('4.com', 'opt/b3lyp')
    ##ang_to_mat((np.pi/3, np.pi/2, np.pi))
    #q1 = QMInput()
    #q2 = QMInput()
    #q1.read_input('examples/4_Thiouracil-Water_1.xyz')
    #q2.read_input('examples/thio_60.xyz')
    #q2.read_input('examples/thio_60_90_120b.xyz')
    #align_slow(q1.coord, q2.coord, [1,2,3,4,5], [1,2,3,4,5])

    q1 = Assemble()
    q1.read_input('examples/4_Thiouracil-Water_1.xyz')
    q2 = Assemble()
    q2.read_input('examples/4_Thiouracil-Water_1.xyz')

    #q1.add_mol_int(q2, [[1, 4.5, 13, 59.0, 2, 179], [-1, 1.5, 1, 170.0, 3, 179.0], [-2, 1.5, 1, 120.0, 3, 0]], [1, 6, 7])
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='face',     r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='parallel', r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='T-shape',  r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='vertical', r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='?', r0=4.0)
    q1.get_template()
    #q1.write_qm('5.psi4', 'sapt2/adz')
    q1.write_qm('6.gjf', 'opt/b3lyp')
