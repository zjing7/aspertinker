#
import pandas as pd
import numpy as np
import os
import copy
from utils import align_slow, int_to_xyz, ang_to_mat
from chem_constants import *
from chem_utilities import sort_atoms
from collections import defaultdict
from scipy.spatial import distance_matrix

DEBUG_FLAG = True

def debug_w(*args):
    if DEBUG_FLAG:
        print(' '.join([str(K) for K in args]))

class Geom:
    '''Basic geometry object containing topology, trajectory and measurement functions.
    '''
    def __init__(self):
        self.coord = []
        self.frames = []
        self.cells = []
        self.nframes = 0
        self.iframe = None
        self.group_idx = []
        self.multiplicity = []

        # the dicts map atom index to other info
        self.top_natoms = 0
        self.top_name = {}
        self.top_conn = {}
        self.top_type = {}
        self.comments = []
        self.idx_list = []
        self.mass_list = []
        self.idx_to_rank = {}

        # Version 2
        self.verbose = 2
        self.debug = False

        self.groups = pd.DataFrame(columns='Charge Multiplicity'.split())
        self.topology = pd.DataFrame(columns='Name Type Mass Group'.split())
        self.top_conn = defaultdict(list)
        self.comments = []
        self.comment = ''

        self.natoms = 0
        self.nframes = 0
        self.max_natoms = 16
        self.max_nframes = 1
        self.max_size = (1, 16, 3)

        self.frames = np.zeros(tuple(self.max_size))
        self.cells = np.zeros((self.max_size[0], 6))
        self.iframe = None

class GeomObj(Geom):
    '''Geometry object containing basic functions.
    '''
    def __init__(self):
        super(GeomObj, self).__init__()

    def msg_error(self, *args, **kwargs):
        '''Write debug information
        '''
        if self.verbose >= 1:
            print("ERROR:", end=' ')
            print(*args, **kwargs)

    def msg_warning(self, *args, **kwargs):
        '''Write debug information
        '''
        if self.verbose >= 2:
            print("WARNING:", end=' ')
            print(*args, **kwargs)

    def debug_w(self, *args, **kwargs):
        '''Write debug information
        '''
        if self.debug:
            print("DEBUG:", end=' ')
            print(*args, **kwargs)

    def convert_top(self):
        '''Convert between old and new versions.
        '''
        self.natoms = self.top_natoms
        self.topology = pd.DataFrame(index=range(1, self.natoms+1), columns=self.topology.columns)
        for idx in self.topology.index:
            for col, item in zip(('Name', 'Type'), (self.top_name, self.top_type)):
                if idx in item:
                    self.topology.loc[idx, col] = item[idx]
        for igrp, grp in enumerate(self.group_idx):
            for idx in grp:
                self.topology.loc[idx, 'Group'] = igrp
        self.topology.loc[self.topology.index[:len(self.mass_list)], 'Mass'] = self.mass_list

    def reset_frames(self):
        self.max_size = (0, max(16, self.natoms), 3)
        self.frames = np.zeros(tuple(self.max_size))
        self.cells = np.zeros((self.max_size[0], 6))
        self.iframe = None
        self.nframes = 0

    def get_group_index(self):
        group_idxs = []
        for idx in self.groups.index:
            mask = self.topology['Group'] == idx
            group_idxs.append(list(self.topology.index[mask]))
        return group_idxs

    def set_group_index(self, group_idxs):
        self.topology['Group'] = -1
        #self.topology.drop('Group', axis=1, inplace=True)
        self.groups = pd.DataFrame(columns = self.groups.columns)
        for igrp, idxs in enumerate(group_idxs):
            ranks = self.index_to_rank(idxs)
            self.topology.loc[idxs, 'Group'] = int(igrp)
            self.groups.loc[igrp, 'Charge'] = 0
            self.groups.loc[igrp, 'Multiplicity'] = 1

    def index_to_rank(self, idxs):
        '''Convert atom index to rank.

        Rank is the order of the atom, starting from 0, which can be used for 
            indexing coordinates.
        '''
        idx = None
        if isinstance(idxs, int):
            idx = idxs
            idxs = [idx]
        ranks = [self.topology.index.get_loc(_) for _ in idxs if _ in self.topology.index]
        if len(ranks) != len(idxs):
            self.msg_warning("%d out of %d indices were not found."%(len(idxs) - len(ranks), len(idxs)))
            raise ValueError
        if idx is None:
            return ranks
        elif len(ranks) == 1:
            return ranks[0]
        else:
            return None

    def reorder_group(self, new_idxs=[], start=1):
        '''Assign new group indices.

        Assign indices with either new_idxs or range(start, start+natoms)
        '''
        if self.natoms == 0:
            return
        if isinstance(new_idxs, np.ndarray):
            new_idxs = new_idxs.reshape(-1)
        elif isinstance(new_idxs, (tuple, list)):
            pass
        else:
            self.debug_w("WARNING: `new_idxs' should be list-like")
            return
        if len(new_idxs) == 0:
            new_idxs = list(range(start, len(self.groups)+start))
        elif len(new_idxs) != len(self.groups):
            self.debug_w("WARNING: Length of `new_idxs' does not match ngroups")
            return
        idx_map = dict(zip(self.groups.index, new_idxs))
        self.groups.set_index([new_idxs], inplace=True)
        mask = ~(self.topology['Group'].isnull())
        self.topology.loc[mask, 'Group'] = self.topology.loc[mask, 'Group'].apply(lambda t: idx_map[t])

    def sort_index(self):
        '''Sort atoms according to index
        '''
        order_new = np.argsort(self.topology.index)
        if (order_new == np.arange(len(self.topology.index))).all():
            return
        self.topology = self.topology.iloc[order_new, :]
        order_new2 = np.append(order_new, np.arange(len(order_new), self.frames.shape[1]))
        self.frames = self.frames[:, order_new2, :]

    def reorder_index(self, new_idxs=[], start=1, sort=False):
        '''Assign new atom indices.

        Assign indices with either new_idxs or range(start, start+natoms)
        '''
        if self.natoms == 0:
            return
        if isinstance(new_idxs, np.ndarray):
            new_idxs = new_idxs.reshape(-1)
        elif isinstance(new_idxs, (tuple, list)):
            pass
        else:
            self.debug_w("WARNING: `new_idxs' should be list-like")
            return
        if len(new_idxs) == 0:
            if start is None:
                new_idxs = list(self.topology.index)
            else:
                new_idxs = list(range(start, self.natoms+start))
        elif len(new_idxs) != self.natoms:
            self.debug_w("WARNING: Length of `new_idxs' does not match natoms")
            return

        # update indices in top_conn
        idx_map = dict(zip(self.topology.index, new_idxs))
        self.topology.set_index([new_idxs], inplace=True)
        top_conn_new = defaultdict(list)
        for idx in self.top_conn:
            new_idx = idx_map[idx]
            top_conn_new[new_idx].extend( [idx_map[_k] for _k in self.top_conn[idx]])
        self.top_conn = top_conn_new

        if sort:
            self.sort_index()

    def traj_align(self, sels=None, weight=None):
        ''' Align trajectory
        '''
        if self.nframes <= 1:
            return
        if weight == 'mass' and not self.topology['Mass'].isnull().any():
            weight = self.topology['Mass'].to_numpy()
        elif weight == 'heavy' and not self.topology['Mass'].isnull().any():
            weight = self.topology['Mass'].to_numpy(dtype=np.float)
            mask = weight > 3.1
            if sum(mask) >= 3:
                weight[~mask] = 0
        elif weight is None or len(weight) != self.natoms:
            weight = np.ones(self.natoms)
        weight = np.asarray(weight, dtype='float')

        if sels == 'heavy' and not self.topology['Mass'].isnull().any():
            sels = np.arange(self.natoms)[self.topology['Mass'].to_numpy(dtype=np.float) > 3.1]
        if sels is None:
            sels = list(range(0, self.natoms))

        i0 = 0
        for i1 in range(0, self.nframes):
            self.frames[i1] = align_slow(self.frames[i1], self.frames[i0], sel1=sels, sel2=sels, wt1=weight[sels], rmsd=False)

    def traj_rmsd(self, sels=None, outf=None, weight=None):
        ''' Calc trajectory rmsd of groups
        '''
        if sels is None:
            sels = [list(range(1, self.natoms+1))]
        elif sels == 'group':
            self.group_idx = self.get_group_index()
            sels = self.group_idx

        seli = []
        for sel in sels:
            #seli.append([K-1 for K in sel])
            seli.append(self.index_to_rank(sel))

        ngrp = len(sels)

            #weight = self.mass_list
        if weight == 'mass' and not self.topology['Mass'].isnull().any():
            weight = self.topology['Mass']
        elif weight == 'heavy' and not self.topology['Mass'].isnull().any():
            weight = self.topology['Mass'].to_numpy(dtype=np.float)
            mask = weight > 3.1
            if sum(mask) >= 3:
                weight[~mask] = 0
        elif weight is None or len(weight) != self.natoms:
            weight = np.ones(self.natoms)
        weight = np.asarray(weight, dtype='float')

        df_rmsd = pd.DataFrame(columns = ['G%d'%K for K in range(1, ngrp+1)])
        i0 = 0
        for i1 in range(0, self.nframes):
            for isel, sel in enumerate(seli):
                col = 'G%d'%(isel+1)
                rms = align_slow(self.frames[i1][sel], self.frames[i0][sel], wt1=weight[sel], rmsd=True)
                df_rmsd.loc[i1, col] = rms
        if outf is not None:
            df_rmsd.to_csv(outf, float_format='%.4f')
        return df_rmsd
        
    @staticmethod
    def get_ortho(arr1, arr2):
        '''Get orthogonal axes from two non-colinear vectors
        '''
        ax1 = np.array(arr1)
        ax2 = np.array(arr2)
        ax2 -= np.sum(ax1*ax2) * ax1 / np.linalg.norm(ax1)**2.0
        ax1 /= np.linalg.norm(ax1)
        ax2 /= np.linalg.norm(ax2)
        ax3 = np.cross(ax1, ax2)
        return np.array([ax1, ax2, ax3])
    @staticmethod
    def get_principle(arrs):
        R0 = np.mean(arrs, axis=0)
        X = arrs - R0
        w, v = np.linalg.eig(np.dot(X.transpose(), X))
        w_order = np.argsort(-w)
        return v.transpose()[w_order, :]

    def measure(self, ids, iframe=0):
        if len(self.frames) <= iframe:
            return 0
        frame = self.frames[iframe]
        for idx in ids:
            if idx > self.natoms:
                return 0
        ranks = self.index_to_rank(ids)
        xyzs = frame[ranks, :]
        v = 0
        if len(xyzs) == 2:
            r12 = xyzs[1] - xyzs[0]
            dr = r12
            #dr = (r12 + halfcell) % cell - halfcell
            v = np.linalg.norm(dr)
        elif len(xyzs) == 3:
            r21 = xyzs[0] - xyzs[1]
            r23 = xyzs[2] - xyzs[1]
            vcos = np.dot(r21,r23)/np.linalg.norm(r21)/np.linalg.norm(r23)
            v = np.arccos(vcos) / np.pi * 180
        elif len(xyzs) == 4:
            if tuple(xyzs[1]) != tuple(xyzs[2]):
                v = self.calc_tors(*xyzs)
            else:
                v = self.calc_dtoline(*[xyzs[0], xyzs[2], xyzs[3]])
        elif len(xyzs) == 5:
            if np.linalg.norm(xyzs[1] - xyzs[2]) > 0.01: 
                v = self.calc_pseudo(*[xyzs[0], xyzs[1], xyzs[2], xyzs[3], xyzs[4]])
            elif tuple(xyzs[1]) == tuple(xyzs[2]):
                v = self.calc_dtoplane(*[xyzs[0], xyzs[2], xyzs[3], xyzs[4]])
        elif len(xyzs) == 6:
            v = self.calc_angnorm(*xyzs)
        return v

    @staticmethod
    def calc_dtoline(ra, rb, rc):
        "Distance of A to BC"
        ra = np.asarray(ra) # vector a
        rb = np.asarray(rb)
        rc = np.asarray(rc)
        rba = rb - ra
        rcb = rc - rb
        r0cb = rcb / np.linalg.norm(rcb)

        proj = np.dot(rba, r0cb)
        r = rba - proj * r0cb

        d = np.linalg.norm(r)
        return d

    @staticmethod
    def calc_angnorm(ra, rb, rc, rd, re, rf):
        '''
        Angle between norms of two planes
        '''
        ra = np.asarray(ra) # vector a
        rb = np.asarray(rb)
        rc = np.asarray(rc)
        rd = np.asarray(rd)
        re = np.asarray(re)
        rf = np.asarray(rf)

        rba = rb - ra
        rcb = rc - rb
        red = re - rd
        rfe = rf - re
        nabc = np.cross(rba, rcb)   # normal of plane abc
        ndef = np.cross(red, rfe)
        
        cosine = np.dot(nabc, ndef) / (np.linalg.norm(nabc) * np.linalg.norm(ndef))
        angle = np.arccos(cosine)/np.pi*180
        sign = np.dot(rcb, np.cross(nabc, ndef))
        if sign < 0:
            #angle = -angle+360
            pass
        return angle

    @staticmethod
    def calc_dtoplane(ra, rb, rc, rd):
        ra = np.asarray(ra) # vector a
        rb = np.asarray(rb)
        rc = np.asarray(rc)
        rd = np.asarray(rd)

        rba = rb - ra
        rcb = rc - rb
        rdc = rd - rc
        nbcd = np.cross(rcb, rdc)

        d = np.dot(rba, nbcd)/np.linalg.norm(nbcd)
        return d

    @staticmethod
    def calc_pseudo(r1, r2, r3, r4, r5):
        '''
        doi.org/10.1093/nar/gkp608
        Conformational analysis of nucleic acids
        revisited: Curves+
        Nucleic Acids Research, 2009

        r1: C1'
        ...
        r5: O4'
        '''
        # nu1, nu[0]: C1'-C2'-C3'-C4'
        # nu5, nu[4]: O4'-C1'-C2'-C3'
        R = np.array([r1, r2, r3, r4, r5])
        nu = np.zeros(5)

        for i in range(5):
            idx = ((np.arange(0, 5)+i)%5).astype(np.int)
            angle = self.calc_tors(R[idx[0], :], R[idx[1], :], R[idx[2], :], R[idx[3], :])
            nu[i] = (angle)/180*np.pi

        a =  0.4*np.sum(nu*np.cos(0.8*np.arange(5)*np.pi))
        b = -0.4*np.sum(nu*np.sin(0.8*np.arange(5)*np.pi))
        Amp = np.sqrt(a*a+b*b)
        cosine = a/Amp
        angle = np.arccos(cosine)/np.pi*180.0
        if b < 0:
            angle = -angle
        angle = (angle + 90)%360 - 90.0
        return angle
    @staticmethod
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

    @staticmethod
    def list_resize(newsize):
        new_allocated = (newsize >> 3) + (3 if newsize < 9 else 6)
        new_allocated += newsize
        return new_allocated

    def assign_top(self, geo: Geom, update_top=True):
        '''Copy topology from a Geom object.
        '''
        if not isinstance(geo, Geom): 
            raise TypeError

        max_size = list(geo.max_size)
        max_size[0] = 0
        self.frames = np.zeros(tuple(max_size))
        self.cells = np.zeros((max_size[0], 6))
        self.comments = list(geo.comments)

        if True:
            self.topology = geo.topology.copy()
            self.groups = geo.groups.copy()
            self.top_conn = geo.top_conn.copy()
            self.natoms = geo.natoms

    def assign_geo(self, geo: Geom, update_top=True):
        '''Copy coordinates from a Geom object.

        Optionally keep original topology information, if natoms are the same
        '''
        if not isinstance(geo, Geom): 
            raise TypeError

        if self.natoms != geo.natoms:
            update_top = True

        self.natoms = geo.natoms
        self.frames = geo.frames.copy()
        self.cells = geo.cells.copy()
        self.nframes = geo.nframes
        self.iframe = geo.iframe
        self.max_size = tuple(geo.max_size)

        if self.iframe is None:
            self.iframe = 0
        if self.iframe < self.nframes and self.nframes > 0:
            self.coord = self.frames[self.iframe]

        if update_top:
            self.topology = geo.topology.copy()
            self.groups = geo.groups.copy()
            self.top_conn = geo.top_conn.copy()
    def set_max_size(self, i, n):
        '''Set the ith value of max_size to n.
        '''
        if i < len(self.max_size):
            self.max_size = list(self.max_size)
            self.max_size[i] = n
            self.max_size = tuple(self.max_size)

class GeomConvert(GeomObj):
    '''I/O for coordinate files.

    Supported formats: tinker, xyz
    '''
    def __init__(self):
        super(GeomConvert, self).__init__()
        self.version = 2
        pass


    def combine_topology(self, geo):
        ""
        newgeo = GeomObj()
        newgeo.assign_top(geo)

        if len(set(self.topology.index) & set(newgeo.topology.index)) > 0:
            newgeo.reorder_index(start=1+max(self.topology.index))
        if len(set(self.groups.index) & set(newgeo.groups.index)) > 0:
            newgeo.reorder_group(start=1+max(self.groups.index))
        self.topology = pd.concat([self.topology, newgeo.topology])
        self.top_conn.update(newgeo.top_conn)
        self.groups = pd.concat([self.groups, newgeo.groups])
        self.natoms = self.natoms + newgeo.natoms
        self.comments.extend(newgeo.comments)

    def append_atom(self, geo):
        '''Concatenate two structures.

        The numbers of frames should be the same.
        '''
        if self.nframes != geo.nframes:
            self.msg_error("Numbers of frames do not match")
            return
        natoms = self.natoms + geo.natoms
        max_natoms = self.max_size[1]
        if max_natoms < natoms or self.max_size != self.frames.shape:
            max_size = tuple(np.array([self.max_size, self.frames.shape]).max(axis=0))
            self.max_size = max_size
            max_natoms = self.list_resize(natoms)
            #max_size = list(np.max([self.max_size, geo.max_size], axis=0))
            self.set_max_size(1, max_natoms)
            frames = np.zeros(self.max_size)
            frames[:, 0:self.natoms, :] = self.frames
            self.frames = frames
        self.frames[:, self.natoms:natoms,  :] = geo.frames[:, :geo.natoms, :]
        #self.natoms = natoms

        self.combine_topology(geo)

    def append_frame(self, geo):
        '''Append frames.

        The numbers of atoms should be the same.
        '''
        if self.natoms != geo.natoms:
            self.msg_error("Numbers of atoms do not match")
            return
        nframes = self.nframes + geo.nframes
        max_nframes = self.max_size[0]
        if max_nframes < nframes or self.max_size != self.frames.shape:
            max_size = tuple(np.array([self.max_size, self.frames.shape]).max(axis=0))
            self.max_size = max_size

            max_nframes = self.list_resize(nframes)
            self.set_max_size(0, max_nframes)
            frames = np.zeros(self.max_size)
            frames[0:self.nframes, :self.natoms, :] = self.frames[:, :self.natoms, :]
            self.frames = frames
        self.frames[self.nframes:(self.nframes+geo.nframes), :self.natoms,  :] = geo.frames[:geo.nframes, :geo.natoms, :]
        self.nframes = nframes
        self.comments.extend(geo.comments)

    def delete_atom(self, idxs):
        '''Delete coords.
        '''
        keep_atoms = [_ for _ in self.topology.index if _ not in idxs]
        keep_ranks = self.index_to_rank(keep_atoms)
        if len(keep_atoms) == self.natoms:
            self.msg_warning("No atoms deleted")
            return
        self.natoms = len(keep_atoms)
        max_natoms = self.list_resize(self.natoms)
        if max_natoms < self.max_size[1]>>2:
            self.set_max_size(1, max_natoms)
            frames = np.zeros(self.max_size)
            frames[:, :self.natoms, :] = self.frames[:, keep_ranks, :]
            self.frames = frames
        else:
            self.frames = self.frames[:, keep_ranks, :]

        self.topology = self.topology.loc[keep_atoms, :].copy()
        keep_groups = set(self.topology['Group'])
        self.groups = self.groups.loc[self.groups.index.isin(keep_groups), :].copy()
        for idx in idxs:
            if idx in self.top_conn:
                self.top_conn.pop(idx)
        for idx in keep_atoms:
            self.top_conn[idx] = tuple([_ for _ in self.top_conn[idx] if _ in keep_atoms])

    def delete_frame(self, frame_ids):
        '''Delete frames.
        '''
        keep_frames = [_ for _ in range(self.nframes) if _ not in frame_ids]
        if len(keep_frames) == self.nframes:
            self.msg_warning("No frames deleted")
            return
        self.nframes = len(keep_frames)
        max_nframes = self.list_resize(self.nframes)
        if max_nframes < self.max_size[0]>>2:
            self.set_max_size(0, max_nframes)
            frames = np.zeros(self.max_size)
            frames[:self.nframes, :, :] = self.frames[keep_frames, :, :]
            self.frames = frames
        else:
            self.frames = self.frames[keep_frames, :, :]
        self.iframe = min(self.nframes-1, self.iframe)

    def reconstruct(self, frame0, iframe, idxs_base):
        '''Reconstruct coordinates from selected atoms and a template frame
        '''
        N = self.natoms
        R0 = frame0[:N]
        R1 = self.frames[iframe, :N, :].copy()
        distmat0 = distance_matrix(R0, R0)
        ranks = self.index_to_rank(idxs_base)
        # flag whether the atom has been constructed
        idx_f = np.zeros(N, dtype=bool)
        idx_f[ranks] = True
        idx_all = np.arange(N)
        np.random.seed(0)
        R1[~idx_f, :] = np.random.rand(sum(~idx_f), 3)*5

        # [id, atom1, atom2, atom3]
        # id: atom with missing coord
        # atom[1-3]: 3 non-linear closest atoms 
        ids = np.zeros((sum(~idx_f), 4), dtype=int)
        for i in range(sum(~idx_f)):
            imin = np.argsort(distmat0[idx_f][:, ~idx_f].reshape(-1), kind='mergesort')[0]
            # missing atom closest to existing atoms
            idmin = idx_all[~idx_f][imin % sum(~idx_f)]
            id_dist = list(idx_all[idx_f][np.argsort(distmat0[idx_f][:, idmin])])
            id_dist += list(idx_all[~idx_f][np.argsort(distmat0[~idx_f][:, idmin])]    )
            id_anchors = list(id_dist[:2])
            ids[i, 0] = idmin
            ax1 = R0[id_anchors[1]]-R0[id_anchors[0]]
            ax1 /= np.linalg.norm(ax1)
            for i1 in id_dist[2:]:
                ax2 = R0[i1]-R0[id_anchors[0]]
                _cos = np.sum(ax1*ax2)/np.linalg.norm(ax1)/np.linalg.norm(ax2)
                r_ax2 = np.linalg.norm(ax2)*np.sqrt(1-_cos**2.0)
                if r_ax2 > 0.2:
                    id_anchors.append(i1)
                    break
            if len(id_anchors) < 3:
                self.msg_error('Not enough atoms or too many collinear atoms for reconstruction.')
                return
            ids[i, 1:4] = id_anchors
            idx_f[idmin] = True

        coords = np.zeros((len(ids), 3))
        for i in range(len(ids)):
            axes = self.get_ortho(R0[ids[i, 2]] - R0[ids[i, 1]], R0[ids[i, 3]] - R0[ids[i, 1]])
            coord0 = R0[ids[i, 0]] - R0[ids[i, 1]]
            coord1 = np.dot(coord0.reshape(1, 3), np.linalg.inv(axes))
            coords[i, :] = coord1
        for i in range(len(ids)):
            axes = self.get_ortho(R1[ids[i, 2]] - R1[ids[i, 1]], R1[ids[i, 3]] - R1[ids[i, 1]])
            coord1 = coords[i, :]
            coord0 = np.dot(coord1.reshape(1, 3), axes)
            R1[ids[i, 0], :] = R1[ids[i, 1], :] + coord0
        self.frames[iframe, :N, :] = R1

    def guess_mass(self):
        self.mass_list = []
        #for idx in self.idx_list:
        for idx in self.topology.index:
            #name = self.top_name[idx]
            name = self.topology.loc[idx, 'Name']
            mass = 1.008
            for ele in (name[0:2].capitalize(), name[0].upper(), 'H'):
                if ele in ELE_2_MASS:
                    mass = ELE_2_MASS[ele]
                    break
            self.topology.loc[idx, 'Mass'] = mass
            #self.mass_list.append(mass)

    def shift_idx(self, n0, geo: GeomObj):
        ''' Shift inplace

        Superseded by "reoder_idx"
        '''
        return
        for curr_d in geo.top_name, geo.top_type, geo.top_conn:
            d2 = {}
            for iatom in curr_d:
                d2[iatom+n0] = curr_d[iatom]
            curr_d.clear()
            curr_d.update(d2)
        return geo

    def infer_top(self, geo):
        '''Assign topology based on connectivity instead of atom order
        '''
        if self.natoms != geo.natoms:
            self.msg_error('Cannot infer topology from a different molecule.')
            return
        ftype = 'xyz'
        xyz_in = self.write_struct(None, ftype=ftype)
        # template xyz
        txyz_in = geo.write_struct(None, ftype=ftype)
        can_xi = sort_atoms(xyz_in, ftype=ftype, from_string=True, reorder_frag=True)
        can_ti = sort_atoms(txyz_in, ftype=ftype, from_string=True, reorder_frag=True)

        if can_xi[0] != can_ti[0]:
            self.msg_error('ERROR: SMILES strings do not match')
            self.msg_error('-->%s'%('.'.join(can_xi[0])))
            self.msg_error('-->%s'%('.'.join(can_ti[0])))
            return
        geo = copy.deepcopy(geo)
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
        idx_list = sorted(list(t2x))
        t_newidx = [t2x[_k] for _k in idx_list]
        geo.reorder_index(t_newidx, sort=True)
        geo.assign_geo(self, update_top=False)
        self.assign_geo(geo)
        return True

    def append_frag(self, geo: GeomObj, new_frag=True):
        '''Legacy code.
        if new_frag:
            create new fragments
        else:
            combine fragments with existing ones according to group_idx
        '''
        if not isinstance(geo, GeomObj): 
            raise TypeError
        if self.nframes != geo.nframes:
            return
        geo = copy.deepcopy(geo)
        n1 = self.natoms
        n2 = geo.natoms
        self.natoms = n1+n2

        for i in range(n1):
            if len(self.multiplicity) <= i:
                self.multiplicity.append((0, 1))
        for i in range(n2):
            if len(geo.multiplicity) <= i:
                geo.multiplicity.append((0, 1))
        n1grps = len(self.group_idx)
        n2grps = len(geo.group_idx)
        if n1grps == 0:
            self.group_idx = [list(range(1, n1+1))]
            #self.group_idx = [list(self.top_name)]
        if n2grps == 0:
            geo.group_idx = [list(range(1, n2+1))]
            #geo.group_idx = [list(geo.top_name)]
        n1grps = len(self.group_idx)
        n2grps = len(geo.group_idx)
        self.multiplicity = self.multiplicity[:n1grps] + geo.multiplicity[:n2grps]
        for i in range(n2grps):
            if new_frag:
                new_idx = [K+n1 for K in geo.group_idx[i]]
            else:
                new_idx = geo.group_idx[i]
            self.group_idx.append(new_idx)
        

        geo2 = self.shift_idx(self.idx_list[-1], geo)

        self.top_name.update(geo2.top_name)
        self.top_type.update(geo2.top_type)
        self.top_conn.update(geo2.top_conn)

        for iframe in range(self.nframes):
            self.frames[iframe] = np.append(self.frames[iframe], geo.frames[iframe], axis=0)
        self.coord = self.frames[self.iframe]

        self.assign_geo(self)

    def fill_missing(self):
        if len(self.groups) == 0:
            if self.topology['Group'].isnull().all():
                self.topology['Group'] = 0
                idxs = [0]
            else:
                mask = ~((self.topology['Group'].isnull()) | (self.topology['Group']<0))
                idxs = sorted(set(self.topology['Group'][mask]))
            for idx in idxs:
                self.groups.loc[idx, 'Charge'] = 0
                self.groups.loc[idx, 'Multiplicity'] = 1

        self.group_idx = []
        for idx in self.groups.index:
            mask = self.topology['Group'] == idx
            self.group_idx.append(list(self.topology.index[mask]))

        if self.topology['Mass'].isnull().all():
            self.guess_mass()
        return

    def write_traj(self, outf, ftype=None):
        if self.nframes == 0:
            return
        self.write_struct(outf, ftype=ftype, frameids=list(range(self.nframes)))

    def write_struct(self, outf, ftype=None, frameids=[]):
        '''Write structure file.

        If outf is None, return a string.
        '''
        supported_ftypes = {'xyz':self.write_xyz, 'txyz':self.write_tinker, 'tinker':self.write_tinker, 'arc':self.write_tinker}
        if len(frameids) == 0:
            frameids = [self.iframe]
        outp = ''
        if ftype is None or ftype not in supported_ftypes:
            suffix = outf.split('.')[-1]
            ftype = suffix
        if ftype in supported_ftypes:
            if outf is None:
                outp = ''
                for _i in frameids:
                    if _i < 0 or _i >= self.nframes:
                        continue
                    self.coord = self.frames[_i]
                    outp += supported_ftypes[ftype](outf)
                self.coord = self.frames[self.iframe]
                return outp
            else:
                outdir = os.path.dirname(outf)
                if (not os.path.isdir(outdir)) and (outdir != ''):
                    os.makedirs(outdir)
                with open(outf, 'w') as fout:
                    for _i in frameids:
                        if _i < 0 or _i >= self.nframes:
                            continue
                        if _i < len(self.comments):
                            self.comment = self.comments[_i]
                        self.coord = self.frames[_i]
                        outp = supported_ftypes[ftype](outf)
                        fout.write(outp)
                # reset coord 
                self.coord = self.frames[self.iframe]
        else:
            print("WARNING: filetype of %s is not recognized"%(outf))
            return

    def write_xyz(self, outf):
        outp = '%d\n'%(self.natoms)
        outp += '%s\n'%self.comment
        #for i in range(self.natoms):
        for i in range(self.natoms):
            #idx = self.idx_list[i]
            idx = self.topology.index[i]
            name = self.topology.loc[idx, 'Name']
            #name = self.top_name[idx]
            outp += '%4s %12.5f %12.5f %12.5f\n'%(name, self.coord[i,0], self.coord[i,1], self.coord[i,2])
        return outp

        with open(outf, 'w') as fout:
            fout.write(outp)

    def write_tinker(self, outf):
        outp = '%d\n'%(self.natoms)
        #outp += '\n'
        for i in range(self.natoms):
            #idx = self.idx_list[i]
            #name = self.top_name[idx]
            idx = self.topology.index[i]
            name = self.topology.loc[idx, 'Name']
            atype = self.topology.loc[idx, 'Type']
            if np.isnan(atype):
                atype = 0
            if idx in self.top_conn:
                conns = ''.join([' %4d'%K for K in self.top_conn[idx]])
            else:
                conns = ''
            outp += ' %4d %4s %12.5f %12.5f %12.5f %5d%s\n'%(idx, name, self.coord[i,0], self.coord[i,1], self.coord[i,2], atype, conns)
        return outp

        with open(outf, 'w') as fout:
            fout.write(outp)
        pass

    def read_struct(self, *kws, **kwargs):
        '''Read structure file.

        Supported types: xyz, tinker
        '''
        self.read_input(*kws, **kwargs)

    def read_input(self, inpf, ftype=None, ignore_top=False):
        '''Read structure file.

        Supported types: xyz, tinker
        '''
        supported_ftypes = {'xyz':self.read_xyz, 'tinker':self.read_tinker, 'txyz':self.read_tinker, 'arc':self.read_tinker}
        if ftype is None or ftype not in supported_ftypes:
            suffix = inpf.split('.')[-1]
            ftype = suffix
        if ftype in supported_ftypes:
            geo = supported_ftypes[ftype](inpf)
            if ignore_top:
                self.assign_geo(geo, update_top=False)
            else:
                self.assign_geo(geo)
                self.fill_missing()
                self.group_idx = self.get_group_index()
        else:
            print("WARNING: filetype of %s is not recognized"%(inpf))

    def read_tinker_top(self, inpf) -> GeomObj:
        '''
        Read tinker topology
        '''
        geo = GeomObj()
        with open(inpf, 'r') as fin:
            iline = -1
            istart = 1
            for line in fin:
                iline = iline + 1
                w = line.split()
                if iline == 0:
                    geo.natoms = int(w[0])
                    if len(w) > 0:
                        geo.comments.append(' '.join(w[1:]))
                    continue
                elif iline == 1:
                    #if len(w) == 6 and tuple(map(float, w[3:6])) == (90,90,90):
                    if len(w) == 6 and (''.join((''.join(w)).split('.'))).isdigit():
                        istart = 2
                        continue
                    else:
                        istart = 1
                if iline >= istart + geo.natoms:
                    break
                idx = int(w[0])
                geo.topology.loc[idx, ['Name', 'Type']] = w[1], int(w[5])
                #geo.top_name[idx] = w[1]
                #geo.top_type[idx] = int(w[5])
                geo.top_conn[idx] = tuple(map(int, w[6:]))
            geo.topology['Group'] = 0
        return geo

    def read_tinker(self, inpf) -> GeomObj:
        geo = self.read_tinker_top(inpf)
        if geo.natoms == 0:
            return geo

        comments = []
        with open(inpf, 'r') as fin:
            frames = []
            natom = geo.natoms
            iatom = 0

            iline = -1
            istart = 1
            iframe = -1
            cells = []
            cell = np.zeros(6)
            coord = [0 for _ in range(3*natom)]
            idx_read = set()

            for line in fin:
                iline = iline + 1
                w = line.split()
                if iline == 1:
                    #if len(w) == 6 and tuple(map(float, w[3:6])) == (90,90,90):
                    if len(w) == 6 and (''.join((''.join(w)).split('.'))).isdigit():
                        istart = 2
                        cell = np.array(list(map(float, w[0:6])))
                        continue
                    else:
                        istart = 1
                if iline >= 1:
                    #frame number
                    iatom = (iline - istart) % (natom + istart)
                    iframe = (iline - istart)/(natom+istart)
                    if istart == 2 and iatom == natom+istart-1 and len(w) == 6:
                        cell = np.array(tuple(map(float, w[0:6])))
                    if iatom < natom and iline>=1:
                        idx = int(w[0])
                        _xyz = (tuple(map(float, w[2:5])))
                        rank = geo.index_to_rank(idx)
                        #coord[rank, :] = _xyz
                        coord[rank*3:rank*3+3] = _xyz
                        idx_read.add(idx)

                    if iatom == natom - 1:
                        #if len(set(geo.top_name) & set(currframe)) == geo.natoms:
                        if len(set(geo.topology.index) & idx_read) == geo.natoms:
                            frames.append(list(coord))
                            cells.append(cell)
                        cell = np.zeros(6)
                        idx_read = set()

            geo.frames = np.asarray(frames).reshape((-1, natom, 3))
            geo.nframes = len(frames)
            geo.iframe = geo.nframes - 1
            geo.cells = np.asarray(cells)

        self.comments.extend(comments)
        return geo

    def read_xyz(self, inpf) -> GeomObj:
        geo = GeomObj()
        comments = []
        with open(inpf, 'r') as fin:
            frames = []
            lines = fin.readlines()
            if len(lines) == 0:
                return geo
            w = lines[0].split()
            natom = int(w[0])
            coord = [0 for _ in range(3*natom)]
            comments.append(lines[1].strip())
            iatom = 0
            for i in range(2,len(lines)):
                line = lines[i]
                w = line.split()
                if i % (natom + 2) ==1:
                    comments.append(lines[i].strip())
                if i % (natom + 2) <=1:
                    continue
                if len(w) < 4:
                    continue
                iatom += 1
                _name = w[0]
                _xyz = (float(w[1]), float(w[2]), float(w[3]))
                idx = iatom
                rank = iatom - 1
                geo.topology.loc[idx, ['Name', 'Group']] = _name, 0
                coord[rank*3:rank*3+3] = _xyz
                if iatom == natom:
                    #frames.append(np.asarray(coord))
                    frames.append(list(coord))
                    iatom = 0
                    coord = []
            geo.frames = np.asarray(frames).reshape((-1, natom, 3))
            geo.nframes = len(frames)
            geo.iframe = geo.nframes - 1
            geo.natoms = natom

        self.comments.extend(comments)
        return geo

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

        b11 = self.measure([idx1[1], idx1[0]])
        b12 = self.measure([idx1[0], idx1[2]])
        b21 =  geo.measure([idx2[1], idx2[0]])
        b22 =  geo.measure([idx2[0], idx2[2]])

        d0 = 179.0
        b0 = 1.5
        if mode == 'face':
            r_int = [[idx1[0], r0, idx1[1], 180-0.5*a1, idx1[2], d0], [-1, b21, idx1[0], 180-0.5*a2, idx1[1], 1.0], [-1, b22, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])

        elif mode == 'vertical':
            r_int = [[idx1[0], r0, idx1[1], 180-0.5*a1, idx1[2], d0], [-1, b21, idx1[0], 180-0.5*a2, idx1[1], 90.0], [-1, b22, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])

        elif mode == 'vertical-side':
            r_int = [[idx1[0], r0, idx1[1], 180-0.5*a1-30, idx1[2], d0], [-1, b21, idx1[0], 180-0.5*a2, idx1[1], 90.0], [-1, b22, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])

        elif mode == 'T-shape':
            r_int = [[idx1[0], r0, idx1[1], 90.0, idx1[2], 90.0], [-1, b21, idx1[0], 180-0.5*a2, idx1[1], 90.0-0.5*a1], [-1, b22, idx1[0], 180-0.5*a2, -2, 179.0]]
            self.add_mol_int(geo, r_int, idx2[:3])

        elif mode == 'parallel':
            r_int = [[idx1[0], r0, idx1[1], 90.0, idx1[2], 90.0], [-1, b21, idx1[0], 90.0, idx1[1], 0], [-1, b22, idx1[0], 90.0, idx1[2], 0.0]]
            self.add_mol_int(geo, r_int, idx2[:3])
        else:
            print(self.add_mol.__doc__)


    def add_mol_int(self, geo: GeomObj, r_int, anchor_idx):
        '''
        Call GeomConvert.append_frag to append a fragment
        Then apply geometry operations
        
        r_int: [i1, B1, i2, A1, i3, D1], ...
               i < 0 indicates the index |i| in the new internel coord

        anchor_idx: the indices of atoms in the new mol to be aligned to the internel coord
        '''
        if not isinstance(geo, GeomObj): 
            raise TypeError
        n1 = self.natoms
        n2 = geo.natoms
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

    def calc_distance_group(self, idx1, idx2):
        ng1 = len(idx1)
        ng2 = len(idx2)
        ranks1 = self.index_to_rank(idx1)
        ranks2 = self.index_to_rank(idx2)


        dist_mat = np.zeros((ng1, ng2))
        coord = self.frames[self.iframe]
        for i1, id1 in enumerate(ranks1):
            for i2, id2 in enumerate(ranks2):
                d = np.linalg.norm(coord[id2, :] - coord[id1, :])
                dist_mat[i1, i2] = d
        return np.min(dist_mat)

    def find_interface(self, rcutoff=4.5, top=None):
        '''Find the atom indices and unit vectors of the interface.

        returns [indices1, indices2], [x, y, z]
        x is perpendicular to the interaface.
        y is the first principle component of the interface.

        The interface is determined by either rcutoff, or the top n closest-distance atoms
        '''
        self.group_idx = self.get_group_index()
        ngrps = len(self.group_idx)
        if ngrps != 2:
            print('ERROR: only two groups are supported')
            return
        ng1 = len(self.group_idx[0])
        ng2 = len(self.group_idx[1])
        dist_mat = np.ones((ng1, ng2))*100
        coord = self.frames[self.iframe]
        for i1, id1 in enumerate(self.index_to_rank(self.group_idx[0])):
            for i2, id2 in enumerate(self.index_to_rank(self.group_idx[1])):
                d = np.linalg.norm(coord[id2, :] - coord[id1, :])
                dist_mat[i1, i2] = d
        idx_inter = [[], []]
        if isinstance(top, int):
            ids = np.argsort(dist_mat.reshape(-1))[:top]

            idx_inter[0] = [self.group_idx[0][_] for _ in (set(ids//ng2))]
            idx_inter[1] = [self.group_idx[1][_] for _ in (set(ids%ng2))]
            #ids = np.argsort(np.min(dist_mat, axis=1))[:top]
            #idx_inter[0] = [self.group_idx[0][_] for _ in ids]

            #ids = np.argsort(np.min(dist_mat, axis=0))[:top]
            #idx_inter[1] = [self.group_idx[1][_] for _ in ids]
            self.debug_w('Using top n distances. Indices:', idx_inter)
        else:
            for i1, d in enumerate(np.min(dist_mat, axis=1)):
                if d <= rcutoff:
                    idx_inter[0].append(self.group_idx[0][i1])
            for i2, d in enumerate(np.min(dist_mat, axis=0)):
                if d <= rcutoff:
                    idx_inter[1].append(self.group_idx[1][i2])

        if len(idx_inter[0]) + len(idx_inter[1]) < 3:
            self.msg_error('Not enough atoms were found to define the interface')
            return

        ranks = self.index_to_rank(idx_inter[0]), self.index_to_rank(idx_inter[1])
        #rg1 = (np.mean(coord[[_k-1 for _k in idx_inter[0]], :], axis=0))
        #rg2 = (np.mean(coord[[_k-1 for _k in idx_inter[1]], :], axis=0))
        rg1 = (np.mean(coord[ranks[0], :], axis=0))
        rg2 = (np.mean(coord[ranks[1], :], axis=0))
        R0 = 0.5*(rg1+rg2)
        R1 = rg2 - rg1
        R1 /= np.linalg.norm(R1)
        #X1 = coord[[_k-1 for _k in idx_inter[0]+idx_inter[1]], :]
        allranks = list(ranks[0])+list(ranks[1])
        X1 = coord[allranks, :]
        X1 -= np.mean(X1, axis=0)
        Xs = [X1]
        #ws = [R1]
        ws = np.zeros((3, 3))
        ws[0, :] = R1
        for i in range(1, 2):
            Xnew = Xs[-1] - np.dot(np.dot(Xs[0], ws[i-1].reshape(-1, 1)), ws[i-1].reshape(1, -1))
            if self.version > 0:
                w, v = np.linalg.eig(np.dot(Xnew.transpose(), Xnew))
            else:
                w, v = np.linalg.eigh(np.dot(Xnew.transpose(), Xnew))
            w_order = np.argsort(w)
            #wnew = v[:, 0]

            self.debug_w(ws[i-1], w, v)
            if self.version < 2:
                wnew = v[:, 0]
            else:
                wnew = v[:, w_order[-1]]
            Xs.append(Xnew)
            #ws.append(wnew)
            ws[i, :] = np.array(wnew/np.linalg.norm(wnew))
        self.debug_w(Xs)
        #ws.append(np.cross(ws[0], ws[1]))
        ws[2, :] = (np.cross(ws[0, :], ws[1, :]))

        return idx_inter, ws, R0


    def transform(self, idx2, params, R0=[], axes=[], top=3):
        '''Apply rigid-body translation and rotation.

        params: [dx, dy, dz, rx, ry, rz]
        '''
        if len(R0) != 3 or len(params) != 6:
            inters = self.find_interface(top=top)
            if inters is None:
                return
            idxs, ws, R0 = inters 
            axes = ws

        axes = np.array(axes)
        params = np.array(params)
        rotmat = ang_to_mat(params[3:6])
        coord = self.frames[self.iframe]
        R0 = np.array(R0).reshape(1, 3)
        ranks = self.index_to_rank(idx2)
        self.debug_w('param', params)
        self.debug_w('rot mat', rotmat)
        self.debug_w('axes', axes)
        self.debug_w('R0', R0)
        self.debug_w('coord old', coord[ranks, :])
        coord[ranks, :] = np.dot(np.dot(np.dot((coord[ranks, :] - R0), axes.transpose()), rotmat) + params[:3].reshape(1, 3), axes) + R0
        self.debug_w('coord new', coord[ranks, :])
            

        self.frames[self.iframe] = coord

    def add_disp(self, idx2, gx1=[], gx2=[], gy1=[], gy2=[], dr=0, direction=0, mode='manual', rcutoff=4.5, top=5):
        '''
        Add displacement of part of the molecule

        mode
          'manual': manually specify the atom indices that define x-axis and y-axis.
              gx1, gx2, gy1, gy2 need to be provided.
          'distance': automatically find the interface based on cut-off distance, 
              and calculate x-axis to be the direction from the CoM of the first interface 
              to the CoM of the second interface, and the y-axis to be the principle component 
              of the remaining covariance.
              rcutoff needs to be provided.
          'top': automatically find the interface based on nearest atoms

        direction
          0: x
          1: y
          2: z
        '''
        all_dims = ('x', 'y', 'z')
        if direction in all_dims:
            direction = all_dims.index(direction)
        if direction not in (0, 1, 2):
            return
        if len(gx1) == len(gx2) == len(gy1) == len(gy2) == 0:
            mode = 'top'
        if mode == 'distance':
            top = None
        coord = self.frames[self.iframe]
        if mode in ('top', 'distance'):
            inters = self.find_interface(rcutoff=rcutoff, top=top)
            if inters is None:
                return
            idxs, ws, R0 = inters 
            a_x, a_y, a_z = ws[0:3]
        else:
            a_x = np.mean(coord[[K-1 for K in gx2], :], axis=0) - np.mean(coord[[K-1 for K in gx1], :], axis=0)
            a_y = np.mean(coord[[K-1 for K in gy2], :], axis=0) - np.mean(coord[[K-1 for K in gy1], :], axis=0)

            a_x /= np.linalg.norm(a_x)
            a_y = a_y - np.dot(a_y, a_x.transpose())*a_x
            a_y /= np.linalg.norm(a_y)
            a_z = np.cross(a_x, a_y)

        axes = (a_x, a_y, a_z)

        coord[[_k-1 for _k in idx2], :] += axes[direction]*dr
        self.frames[self.iframe] = coord
        self.coord = coord

    def var_dist(self, geo, idx1, idx2, r0=None, r_rel=None):

        if not isinstance(geo, Geom): 
            raise TypeError
        if r0 is None and r_rel is None:
            print("No distance specified")
            return

        n1 = self.natoms
        n2 = geo.natoms

        #self.append_frag(geo)
        self.append_atom(geo)

        for iframe in range(self.nframes):
            coord = self.frames[iframe]
            Rg1 = np.mean(coord[[K-1 for K in idx1],:], axis=0)
            Rg2 = np.mean(coord[[K+n1-1 for K in idx2],:], axis=0)

            _R = Rg2 - Rg1 
            _r = np.linalg.norm(_R)
            _R0 = _R/_r

            if r0 is None:
                dr = _r*(r_rel-1)
            else:
                dr = r0 - _r

            coord[n1:] += (_R0*dr).reshape(1,3)

if __name__ == '__main__':
    geo = GeomConvert()
    #geo.read_input('4_Thiouracil-Water_1.xyz')
    geo.read_input('scratch/mol.xyz', ftype='tinker')
    print(geo.coord)
    print(geo.top_type)
    print(geo.nframes)
    debug_w(1, 2, 3, 'aa', 4)
