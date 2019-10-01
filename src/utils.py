#!/usr/bin/env python
import numpy as np
import scipy.optimize

def calc_tors(ra, rb, rc, rd):
    ra = np.asarray(ra) # vector a
    rb = np.asarray(rb)
    rc = np.asarray(rc)
    rd = np.asarray(rd)

    rba = rb - ra
    rcb = rc - rb
    rdc = rd - rc
    nabc = np.cross(rba, rcb)   # normal of plane abc
    nbcd = np.cross(rcb, rdc)
    
    cosine = np.dot(nabc, nbcd) / (np.linalg.norm(nabc) * np.linalg.norm(nbcd))
    angle = np.arccos(cosine)/np.pi*180
    sign = np.dot(rcb, np.cross(nabc, nbcd))
    if sign < 0:
        angle = -angle
        pass
    return angle


def coord_trans(xyz3, int3):
    '''
    Given internel coordinate and cartesian coordinates of reference atoms,
    calculate cartesian coordinate
    '''

    ra = xyz3[2]
    rb = xyz3[1]
    rc = xyz3[0]
    rab = rb - ra
    rbc = rc - rb

    B, A, D = int3[:3]
    A = A/180.0*np.pi
    D = D/180.0*np.pi

    nabc = np.cross(rab, rbc)
    nabc /= np.linalg.norm(nabc)

    def int_residual(x, rbc, nabc, cosD):
        '''
        solve x = nbcd using least squares

        # nbcd . rbc == 0
        # nabc . nbcd == cos(D)
        # nbcd . nbcd == 1
        '''
        X = np.asarray(x)
        eq1 = np.dot(X, rbc)
        eq2 = np.dot(nabc, X) - (cosD)
        eq3 = np.dot(X, X) - 1
        return eq1, eq2, eq3

    sols = []
    for x0 in [[1.0, 1.0, 1.1], [-1.0, -1.0, -1.1]]:
        sols.append( scipy.optimize.least_squares(int_residual, x0, verbose=0, kwargs={'rbc':rbc, 'nabc':nabc, 'cosD':np.cos(D)}))

        sol = sols[-1]
        # Check if the solution conforms to the right-handed convention
        nbcd = sol.x
        sign = np.dot(rbc, np.cross(nabc, nbcd))
        if sign * np.sign(D) > 0:
            break

    rbc0 = rbc / np.linalg.norm(rbc)
    rbcT = np.cross(-rbc0, nbcd)
    rd = rc + B * (-rbc0 * np.cos(A) + rbcT * np.sin(A))

    #print(np.sign(D), D, calc_tors(xyz3[0], xyz3[1], xyz3[2], rd))
    return rd


def int_to_xyz(r_int):
    '''
    '''
    # default bond, angle, dihedral
    R0 = np.array([0, 0, 1.0])
    B0 = 1.0
    A0 = 109.5
    D0 = 180.0

    n_atoms = len(r_int)
    r_xyz = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        if len(r_int[i]) == 3:
            r_xyz[i] = r_int[i]
            continue
        if len(r_int[i]) < min(6, (i+1)*2):
            # if not enough arguments, use default
            i1, i2, i3 = i-1, i-2, i-3
            b_i, a_i, d_i = B0, A0, D0
        if i == 0:
            r_xyz[i] = [0, 0, 0]
        elif i == 1:
            r_xyz[i] = [0, 0, r_int[i][1]]
        elif i == 2:
            d_i = D0
            i1, b_i, i2, a_i = r_int[i][:4]
            for ii in (i1, i2):
                if ii-1 >= i:
                    print("ERROR: atom %d referenced in atom %d definition"%(ii, i+1))  
                continue
            R1 = r_xyz[i1-1]
            R2 = r_xyz[i2-1]
            R3 = R1 + np.array([-0.1,0, 0])
            print("Calling trans:", i1, i2, i3)
            r_xyz[i] = coord_trans([R1, R2, R3], [b_i, a_i, d_i])
        else:
            i1, b_i, i2, a_i, i3, d_i = r_int[i][:6]
            for ii in (i1, i2):
                if ii-1 >= i:
                    print("ERROR: atom %d referenced in atom %d definition"%(ii, i+1))  
                continue
            R1 = r_xyz[i1-1]
            R2 = r_xyz[i2-1]
            R3 = r_xyz[i3-1]

            r_xyz[i] = coord_trans([R1, R2, R3], [b_i, a_i, d_i])

    return r_xyz

def ang_to_mat(ang):
    Rx = np.zeros((3, 3))
    Ry = np.zeros((3, 3))
    Rz = np.zeros((3, 3))

    for i0, R in enumerate([Rx, Ry, Rz]):
        j1 = (i0-1)%3
        j2 = (i0+1)%3
        R[i0, i0] = 1
        R[j2, j2] = np.cos(ang[i0])
        R[j1, j1] = np.cos(ang[i0])
        R[j2, j1] = np.sin(ang[i0])
        R[j1, j2] = -np.sin(ang[i0])

    # https://en.wikipedia.org/wiki/Rotation_matrix
    # adapted for row matrics

    return np.dot(np.dot(Rz, Ry), Rx)

def align_slow(R1, R2, sel1=None, sel2=None, wt1=None, wt2=None, rmsd=False):
    '''
    RMSD align using least square
    return aligned coordiates of R1
    '''
    rot_ang0 = [0, 0, 0]

    if sel1 is None:
        sel1 = list(range(len(R1)))
    if sel2 is None:
        sel2 = list(range(len(R2)))
    if wt1 is None or len(wt1) != len(R1):
        wt1 = np.ones(len(R1))
    wt1 = np.asarray(wt1)
    wt1 = wt1/np.sum(wt1)

    if len(sel1) != len(sel2):
        print(sel1, sel2)
        print("ERROR: Numbers of atoms (%d, %d) in selection do not match"%(len(sel1), len(sel2)))
        raise ValueError
        return R1

    def residual(ang, r1, r2, wt):
        mat = ang_to_mat(ang)
        com1 = np.mean(r1, axis=0).reshape(1,3)
        com2 = np.mean(r2, axis=0).reshape(1,3)

        #r1p = np.dot(mat, (r1-com1).T).T + com1
        r1p = np.dot((r1-com1), mat) + com2

        resi = ((r1p - r2)*np.sqrt(wt.reshape(-1,1))).ravel()
        return resi

    sols = []
    sols.append( scipy.optimize.least_squares(residual, rot_ang0, verbose=0, kwargs={'r1':R1[sel1,:], 'r2':R2[sel2,:], 'wt':wt1[sel1]}))
    rot_ang = sols[0].x
    rot_mat = ang_to_mat(rot_ang)

    com1 = np.mean(R1[sel1,:], axis=0)
    com2 = np.mean(R2[sel2,:], axis=0)
    #R1p = np.dot(rot_mat, (R1-com1).T).T + com2
    R1p = np.dot((R1-com1), rot_mat) + com2
    if rmsd:
        natom = len(R1p)
        return np.linalg.norm((R1p-R2)*np.sqrt(wt1.reshape(-1,1)))
    else:
        return R1p

if __name__ == '__main__':
    #xyz0 = np.array([[1.0, -1, 0], [1.0, 0, 0], [0.0, 0.0, 0]])
    #coord_trans(xyz0, [2, 120, 179.0])
    #coord_trans(xyz0, [2, 120, 0])
    #coord_trans(xyz0, [2, 120, 80])
    #coord_trans(xyz0, [2, 120, -80])
    #coord_trans(xyz0, [2, 120, 110])
    #coord_trans(xyz0, [2, 120, -110])
    #r_int = [[], [1, 0.9], [1, 1.1, 2, 110]]
    r_int = [
    [], 
    [1, 1.50715], 
    [1, 1.11661, 2, 110.43],
    [1, 1.11661, 2, 110.63, 3, 120.0],
    [1, 1.11661, 2, 110.63, 3,-120.0],
    [2, 1.50761, 1, 111.83, 3,-179.9],
    [2, 1.11661, 1, 109.43, 6, 121.0],
    [2, 1.11661, 1, 109.43, 6,-121.0],
    [6, 1.11661, 2, 110.43, 1, 179.9],
    ]

    r_int = [
    [0., 0., 0.], 
    [1, 1.50715], 
    [1, 1.11661, 2, 110.43],
    [-0.52319100,  -0.90491000,  -0.39353200],
    [1, 1.11661, 2, 110.63, 3,-120.0],
    [2, 1.50761, 1, 111.83, 3,-179.9],
    [2, 1.11661, 1, 109.43, 6, 121.0],
    [2, 1.11661, 1, 109.43, 6,-121.0],
    [6, 1.11661, 2, 110.43, 1, 179.9],
    ]
    car = int_to_xyz(r_int)
    #print(car)
    for ixyz, xyz in enumerate(car):
        print(ixyz+1, '%8.4f %8.4f %8.4f'%tuple(xyz))

