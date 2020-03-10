#psi4
#
memory ${memory} gb

molecule {
${geometry}units angstrom
no_reorient  #important for SAPT in psi4, default
symmetry c1  #important for SAPT in psi4, default
}

set {
basis ${psi4_basis}
scf_type DF
freeze_core False
}
${PS}

energy('${psi4_energy}', bsse_type='cp')

