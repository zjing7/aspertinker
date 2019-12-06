from openbabel import openbabel
from openbabel import pybel
#import pybel
import sys
import numpy as np

def auto_fragment(xyz_in, **kwargs):
    can_xi = sort_atoms(xyz_in, **kwargs) # canonical SMILES
    return can_xi[1]

def sort_atoms(inpf, ftype=None, reorder_frag=False, from_string=False):
    '''
    inpf: input chemical file name
    reorder_frag: whether to sort the fragments or not

    return: [List(str), List(List(int))] the canonical SMILES string(s) and atom indices corresponding to the canonical orders of fragments

    Note: Only read the first molecule in the file
    '''
    #pybel.ob.OBMessageHandler.SetOutputLevel(pybel.ob.OBMessageHandler(), pybel.ob.obError)
    #openbabel.OBMessageHandler.SetOutputLevel(openbabel.OBMessageHandler(), 3)
    openbabel.obErrorLog.SetOutputLevel(openbabel.obError)
    #openbabel.OBMessageHandler.StopLogging(openbabel.OBMessageHandler())
    #openbabel.OBMessageHandler.StopLogging(openbabel.OBMessageHandler())
    #print('output level', openbabel.OBMessageHandler.GetOutputLevel(openbabel.OBMessageHandler()))
    if ftype is None:
        ftype = openbabel.OBConversion.FormatFromExt(inpf)

    if from_string:
        mymols = list([pybel.readstring(ftype, inpf)])
    else:
        mymols = list(pybel.readfile(ftype, inpf))
    if len(mymols) == 0:
        return [], []

    mymol = list(mymols)[0]
    #smi = mymol.write('smi')
    smi = mymol.write('can')
    sms = smi.split()[0].split('.')
    sms = sorted(list(set(sms)))
    idx_out = []  #all atom
    sm_list = []
    
    natoms = mymol.OBMol.NumAtoms()
    conn = [[] for _k in range(natoms+1)] # connnections
    for atom in mymol:
        bonds = pybel.ob.OBAtomAtomIter(atom.OBAtom)
        for atom2 in bonds:
            atomic = atom2.GetAtomicNum()
            if atomic == 1:
                conn[atom.idx].append(atom2.GetIdx())
    for sm in sms:
        smarts = pybel.Smarts(sm)
        idxs_list = smarts.findall(mymol)
        for idxs in idxs_list:

            if len(idxs) == 0:
                continue
            #idx_list.append(idx)
            sm_list.append(sm)
            idx_out.append([])
            #out_list.append(idxs)
            for idx in idxs:
                idx_out[-1].extend([idx]+sorted(conn[idx]))
            idx_out[-1] = tuple(idx_out[-1])
    rank_size = np.argsort([len(_k) for _k in idx_out], axis=0)
    atom_added = set()
    idx_sel = set()
    for i in reversed(rank_size):
        # filter out sub-fragments that are same as other fragments
        idx = idx_out[i]
        if len(set(idx) & atom_added) == 0:
            atom_added.update(set(idx))
            idx_sel.add(i)

    if reorder_frag:
        idx_sorted = np.argsort(sm_list, axis=0)
    else:
        idx_sorted = np.argsort([_k[0] for _k in idx_out], axis=0)
        
    sm_list = [sm_list[_i] for _i in idx_sorted if _i in idx_sel]
    idx_out = [idx_out[_i] for _i in idx_sorted if _i in idx_sel]

    return sm_list, idx_out

def main():
    inpf = sys.argv[1]
    print(sort_atoms(inpf))
    return

if __name__ == '__main__':
    main()
