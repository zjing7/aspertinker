#!/usr/bin/env python
import sys
import os
from tools.chem_constants import ELEMENT_NAME
from chem_utilities import sort_atoms
from geom_io import GeomConvert


ElementName = ELEMENT_NAME

#FORMAT='%3d %4s %14.6f %14.6f %14.6f %4d %s\n'
FORMAT='%4s %14.6f %14.6f %14.6f\n'
def get_tmp_name(pref='tmp'):
    pid = os.getpid()
    for i in range(65536):
        fname = '%s_pid%d.%d'%(pref, pid, i)
        if not os.path.isfile(fname):
            return fname
    return None

def convert_txyz(xyz_in, txyz_in, txyz_out, original_id=False, verbose=1):
    '''
    Convert xyz to txyz
    '''
    pxi = GeomConvert()
    pxi.read_input(xyz_in, ftype='xyz')

    pti = GeomConvert()
    pti.read_input(txyz_in, ftype='tinker')


    if pxi.top_natoms != pti.top_natoms:
        #print("ERROR: Numbers of atoms do not match")
        return

    pto = GeomConvert()

    ftmp = get_tmp_name()
    if ftmp is None:
        print('tmp file exists')
        return
    pti.write_struct(ftmp, ftype='xyz')
    can_xi = sort_atoms(xyz_in) # canonical SMILES
    can_ti = sort_atoms(ftmp, ftype='xyz')
    os.remove(ftmp)

   
    if can_xi[0] != can_ti[0]:
        if verbose >= 1:
            print('ERROR: SMILES strings do not match')
            print('-->%s'%('.'.join(can_xi[0])))
            print('-->%s'%('.'.join(can_ti[0])))
        return None
    nfrags = len(can_xi[1])
    x2t = {}
    t2x = {}
    for i in range(nfrags):
        natoms0 = len(can_xi[1][i])
        for j in range(natoms0):
            idx1 = can_xi[1][i][j]
            idx2 = can_ti[1][i][j]
            x2t[idx1] = idx2
            t2x[idx2] = idx1
    if original_id:
        print("ERROR: retaining original id has not been implemented")
        return
    else:
        pto.assign_geo(pti)
        for idx1 in x2t:
            idx2 = x2t[idx1]
            pto.coord[idx2-1,:] = pxi.coord[idx1-1,:]
        pto.write_struct(txyz_out, ftype='tinker')
    return True




def convert(ftpl, fpsi): 
    fin = open(ftpl, 'r')
    lines0 = fin.readlines()
    fin.close()
    natom = int(lines0[0].split()[0])
    xyz = '%5d\n\n'%natom
    for i in range(1,len(lines0)):
        line0 = lines0[i]
        w0 = line0.split()
        if w0[1] in ElementName:
            ele = w0[1]
        else:
            ele = w0[1][0]
        xyz = xyz + FORMAT%( ele, float(w0[2]), float(w0[3]), float(w0[4]))
    outp = xyz
    fout = open(fpsi, 'w')
    fout.write(outp)
    fout.close()

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        convert_txyz(*sys.argv[1:4])
