
from geom_io import GeomFile, GeomConvert, debug_w
from utils import align_slow, int_to_xyz
import os
from string import Template
import numpy as np

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
        #self.template_path = os.path.join(self.pkgpath, 'qm_template', 'template.list')
        self.template_path = os.path.join(self.pkgpath, '..', 'dat', 'qm', 'template.list')
        self.QM_WRITE = {'psi4':self.write_psi4, 'g16':self.write_g16}
        self.QM_PROGRAMS = set(self.QM_WRITE)
        self.memory = 20
        self.numproc = 8
        self.disk = 100
        self.methods = {}
        self.psi4_energy = 'scf'
        self.psi4_basis =  'def2-TZVP'
        self.keywords = {}
        self.keywords['memory'] = 20
        self.keywords['disk'] = 100
        self.keywords['numproc'] = 8
        self.keywords['psi4_energy'] = 'scf'
        self.keywords['psi4_basis'] = 'def2-TZVP'

        self.get_template()

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

    def set_keywords(self, kwarg):
        self.keywords.update(kwarg)

    def get_default(self):
        kws = {}
        kws.update(self.keywords)
        kws['nfrag'] = '%d'%(len(self.group_idx))
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

    def write_qm(self, outf, theory, kws={}):
        self.set_keywords(kws)
        if theory in self.methods:
            method = self.methods[theory]
            if method.program in self.QM_WRITE:
                outdir = os.path.dirname(outf)
                if (not os.path.isdir(outdir)) and outdir != '':
                    os.makedirs(outdir)
                self.QM_WRITE[method.program](outf, method)
        else:
            print(theory, 'not supported')
            self.print_all_methods()

    def write_g16(self, outf, method: QMMethod):
        if not isinstance(method, QMMethod): 
            raise TypeError
        self.fill_missing()
        molecule = ''
        ngrps = len(self.group_idx)
        frags = []
        if ngrps > 1:
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
                i = iatom - 1
                idx = self.idx_list[i]
                frags[-1] += '%s%s %12.5f %12.5f %12.5f\n'%(self.top_name[idx], grp_s, self.coord[i][0], self.coord[i][1], self.coord[i][2])
        molecule = ''.join(frags)
        #outf2 = '.'.join(outf.split('.')[:-1]) + '.chk'
        outf2 = '.'.join(os.path.split(outf)[-1].split('.')[:-1]) + '.chk'
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
                idx = self.idx_list[i]
                #i = self.idx_to_rank[iatom]
                frags[-1] += '%s %12.5f %12.5f %12.5f\n'%(self.top_name[idx], self.coord[i][0], self.coord[i][1], self.coord[i][2])
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


if __name__ == '__main__':
    #qmfile = QMInput()
    #qmfile.get_template()
    #qmfile.read_input('scratch/mol.xyz', ftype='tinker')
    #qmfile.read_input('scratch/mol.xyz', ftype='tinker')
    #qmfile.read_input('scratch/4_Thiouracil-Water_1.xyz')
    #q2 = QMInput()
    #q2.read_input('scratch/4_Thiouracil-Water_1.xyz')
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
    #q1.read_input('scratch/4_Thiouracil-Water_1.xyz')
    #q2.read_input('scratch/thio_60.xyz')
    #q2.read_input('scratch/thio_60_90_120b.xyz')
    #align_slow(q1.coord, q2.coord, [1,2,3,4,5], [1,2,3,4,5])

    q1 = Assemble()
    q1.read_input('scratch/4_Thiouracil-Water_1.xyz')
    q2 = Assemble()
    q2.read_input('scratch/4_Thiouracil-Water_1.xyz')

    #q1.add_mol_int(q2, [[1, 4.5, 13, 59.0, 2, 179], [-1, 1.5, 1, 170.0, 3, 179.0], [-2, 1.5, 1, 120.0, 3, 0]], [1, 6, 7])
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='face',     r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='parallel', r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='T-shape',  r0=4.0)
    q1.add_mol(q2, [1, 3, 7], [1, 3, 7], mode='vertical', r0=4.0)
    q1.get_template()
    #q1.write_qm('5.psi4', 'sapt2/adz')
    q1.write_qm('6.gjf', 'opt/b3lyp')
