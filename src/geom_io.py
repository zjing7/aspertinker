#
import numpy as np
import copy

DEBUG_FLAG = True

def debug_w(*args):
    if DEBUG_FLAG:
        print(' '.join([str(K) for K in args]))

class GeomFile:
    def __init__(self):
        self.coord = []
        self.frames = []
        self.cells = []
        self.nframes = 0
        self.iframe = None
        self.group_idx = []
        self.multiplicity = []
        self.top_natoms = 0
        self.top_name = {}
        self.top_conn = {}
        self.top_type = {}
        self.comments = []
        self.idx_list = []
        self.idx_to_rank = {}

    def measure(self, ids, iframe=0):
        if len(self.frames) <= iframe:
            return 0
        frame = self.frames[iframe]
        for idx in ids:
            if idx > self.top_natoms:
                return 0
        xyzs = frame[[K-1 for K in ids], :]
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

class GeomConvert(GeomFile):
    def __init__(self):
        super(GeomConvert, self).__init__()
        pass

    def assign_geo(self, geo: GeomFile):
        if not isinstance(geo, GeomFile): 
            raise TypeError

        self.frames = geo.frames
        self.cells = geo.cells
        self.nframes = len(self.frames)
        self.iframe = geo.iframe
        if self.iframe is None:
            self.iframe = 0
        if self.iframe < self.nframes:
            self.coord = self.frames[self.iframe]
        self.top_name = geo.top_name
        self.top_type = geo.top_type
        self.top_conn = geo.top_conn
        self.top_natoms = len(self.coord)
        self.idx_list = sorted(list(self.top_name))
        self.idx_to_rank.clear()
        for rank in range(len(self.idx_list)):
            idx = self.idx_list[rank]
            self.idx_to_rank[idx] = rank


    def shift_idx(self, n0, geo: GeomFile):
        '''
        shift inplace
        '''
        for curr_d in geo.top_name, geo.top_type, geo.top_conn:
            d2 = {}
            for iatom in curr_d:
                d2[iatom+n0] = curr_d[iatom]
            curr_d.clear()
            curr_d.update(d2)
        return geo

    def append_frag(self, geo: GeomFile, new_frag=True):
        '''
        if new_frag:
            create new fragments
        else:
            combine fragments with existing ones according to group_idx
        '''
        if not isinstance(geo, GeomFile): 
            raise TypeError
        if self.nframes != geo.nframes:
            return
        geo = copy.deepcopy(geo)
        n1 = self.top_natoms
        n2 = geo.top_natoms
        self.top_natoms = n1+n2

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

    def read_input(self, inpf, ftype=None):
        supported_ftypes = {'xyz':self.read_xyz, 'tinker':self.read_tinker, 'arc':self.read_tinker}
        if ftype == None or ftype not in supported_ftypes:
            suffix = inpf.split('.')[-1]
            ftype = suffix
        if ftype in supported_ftypes:
            geo = supported_ftypes[ftype](inpf)
            self.assign_geo(geo)
        else:
            print("WARNING: filetype of %s is not recognized"%(inpf))

    def read_tinker_top(self, inpf) -> GeomFile:
        '''
        Read tinker topology
        '''
        geo = GeomFile()
        with open(inpf, 'r') as fin:
            iline = -1
            istart = 1
            for line in fin:
                iline = iline + 1
                w = line.split()
                if iline == 0:
                    geo.top_natoms = int(w[0])
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
                if iline >= istart + geo.top_natoms:
                    break
                idx = int(w[0])
                geo.top_name[idx] = w[1]
                geo.top_type[idx] = int(w[5])
                geo.top_conn[idx] = tuple(map(int, w[6:]))
        return geo

    def read_tinker(self, inpf) -> GeomFile:
        geo = self.read_tinker_top(inpf)
        if geo.top_natoms == 0:
            return geo

        with open(inpf, 'r') as fin:
            frames = []
            natom = geo.top_natoms
            iatom = 0

            iline = -1
            istart = 1
            iframe = -1
            coord = []
            cells = []
            cell = []
            currframe = {}
            idx_list = sorted(list(geo.top_name))

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
                        currframe[idx] = _xyz

                    if iatom == natom - 1:
                        if len(set(geo.top_name) & set(currframe)) == geo.top_natoms:
                            coord = []
                            for idx in idx_list:
                                coord.append(currframe[idx])
                            frames.append(np.asarray(coord))
                            cells.append(cell)
                        cell = []
                        currframe = {}

            geo.frames = frames
            geo.nframes = len(frames)
            geo.iframe = geo.nframes - 1
            geo.cells = cells

        return geo

    def read_xyz(self, inpf) -> GeomFile:
        geo = GeomFile()
        with open(inpf, 'r') as fin:
            top_name = {}
            coord = []
            frames = []
            comments = []
            lines = fin.readlines()
            w = lines[0].split()
            natom = int(w[0])
            comments.append(lines[1])
            iatom = 0
            for i in range(2,len(lines)):
                line = lines[i]
                w = line.split()
                if i % (natom + 2) <=1:
                    continue
                if len(w) < 4:
                    continue
                iatom += 1
                _name = w[0]
                _xyz = (float(w[1]), float(w[2]), float(w[3]))
                top_name[iatom] = _name
                coord.append(_xyz)
                if iatom == natom:
                    frames.append(np.asarray(coord))
                    iatom = 0
                    coord = []
            geo.frames = frames
            geo.top_name = top_name
            geo.nframes = len(frames)
            geo.iframe = geo.nframes - 1

        return geo

if __name__ == '__main__':
    geo = GeomConvert()
    #geo.read_input('4_Thiouracil-Water_1.xyz')
    geo.read_input('mol.arc')
    print(geo.coord)
    print(geo.top_type)
    print(geo.nframes)
    debug_w(1, 2, 3, 'aa', 4)
