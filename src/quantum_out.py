#
import numpy as np
import copy
from geom_io import GeomObj, GeomConvert
from quantum_inp import QMInput
import pandas as pd
import re
class QMEntry:
    def __init__(self, name, default_value = None):
        self.name = name 
        self.iline_start = -1
        self.regex_start = None
        self.regex_value = None
        self.value_ilines = [0] # line numbers relative to the start line
        self.default_value = default_value
        self.value = default_value

    def clear(self):
        self.iline_start = -1

    def set_pattern(self, p_value, p_start=None, line_number = [0], convert_func = None):
        self.regex_value = p_value
        self.regex_start = p_start

        self.re_value = re.compile(self.regex_value)
        if p_start is not None:
            self.re_start = re.compile(self.regex_start)

        self.value_ilines = set(line_number)
        if not callable(convert_func):
            convert_func = lambda t: float(t[0])
        self.convert_func = convert_func

    def process_line(self, iline, line):
        match = None
        if self.regex_start is None:
            match = self.re_value.match(line)
        else:
            if self.re_start.match(line):
                self.iline_start = iline
            if iline - self.iline_start in self.value_ilines:
                match = self.re_value.match(line)
        if match:
            #value = self.convert_func(match.group(1))
            if self.name == 'Time':
                pass
            value = self.convert_func(match.groups())
            self.value = value
            return value

class QMResult(GeomConvert):
    def __init__(self):
        super(QMResult, self).__init__()

        self.qm_columns = 'Energy Freq1 Error MaxForce RMSForce MaxDisp RMSDisp'.split()

        self.qm_data = pd.DataFrame(columns = self.qm_columns)
        self.i_coord_start = -100

        self.ftype = None

        self.regex_frame_start = "No man's land"
        self.regex_coord_start = "No man's land"
        self.regex_coord_line  = "No man's land"

        #self.REGEX_FLOAT = '-?\d+(\.\d+)?'
        self.REGEX_FLOAT = '-?\d+(?:\.\d+)?'
        self.entry_format = {}
        self.MAX_VALUE = 1e5
        self.MIN_VALUE = -1e5

    @staticmethod
    def g16_conv(t):
        s = t[0].replace('D', 'E')
        return float(s)

    def set_entry_pattern(self, ftype = 'g16'):
        if self.ftype == ftype:
            return
        self.ftype = ftype
        g16_float = '-?\d+(?:\.\d+)?(?:D.\d+)?'

        all_en = []
        if ftype in ('g16', 'g16std', 'g16cp'):
            all_en = []

            en = QMEntry('Energy', None)
            #en.set_pattern('^ SCF Done: ', '^ SCF Done:\s+\S+\s+=\s+(%s)\s+A\.U\.'%(self.REGEX_FLOAT), [0])
            en.set_pattern('^ SCF Done:\s+\S+\s+=\s+(%s)\s+A\.U\.'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('E2', None)
            #en.set_pattern('^ SCF Done: ', '^ SCF Done:\s+\S+\s+=\s+(%s)\s+A\.U\.'%(self.REGEX_FLOAT), [0])
            en.set_pattern('^ E2 =\s+(%s)'%(g16_float), convert_func=self.g16_conv)
            all_en.append(en)

            en = QMEntry('CCSDT', None)
            #en.set_pattern('^ SCF Done: ', '^ SCF Done:\s+\S+\s+=\s+(%s)\s+A\.U\.'%(self.REGEX_FLOAT), [0])
            en.set_pattern('^ CCSD\(T\)=\s+(%s)'%(g16_float), convert_func=self.g16_conv)
            all_en.append(en)

            en = QMEntry('MaxForce', None)
            en.set_pattern('^ Maximum\s+Force\s+(%s)\s+%s'%(self.REGEX_FLOAT, self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('RMSForce', None)
            en.set_pattern('^ RMS\s+Force\s+(%s)\s+%s'%(self.REGEX_FLOAT, self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('Freq1', None)
            #en.set_pattern('^ Low frequencies ---\s+(%s)\s+'%(self.REGEX_FLOAT), '^ Full mass-weighted force constant matrix', [1])
            en.set_pattern('^ Frequencies --\s+(%s)\s+'%(self.REGEX_FLOAT), '^ Harmonic frequencies ', [6])
            all_en.append(en)

            en = QMEntry('lnQ', None)
            en.set_pattern('^ Total V=0\s+\S+\s+\S+\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('G', None)
            en.set_pattern('^ Sum of electronic and thermal Free Energies=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('H', None)
            en.set_pattern('^ Sum of electronic and thermal Enthalpies=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('Success', False)
            en.set_pattern('^ Normal termination of Gaussian.*at (.*)\.', convert_func = lambda t: True)
            all_en.append(en)

            en = QMEntry('Error', '')
            #en.set_pattern('^ Error termination request processed by (.*).', convert_func = lambda t: t)
            en.set_pattern('^ Error termination via Lnk1e in (\S+)', convert_func = lambda t: t)
            all_en.append(en)

            en = QMEntry('Time', None)
            en.set_pattern('^ Elapsed time:\s+(%s) days\s+(%s) hours\s+(%s) minutes\s+(%s) seconds\.'%(self.REGEX_FLOAT, self.REGEX_FLOAT, self.REGEX_FLOAT, self.REGEX_FLOAT) , \
                    convert_func = lambda t: float(t[0])*24+float(t[1])+float(t[2])/60+float(t[3])/3600)
            all_en.append(en)

        elif ftype == 'cp2k':
            all_en = []

            en = QMEntry('Energy', None)
            en.set_pattern('^.*\s+Total energy:\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('Success', False)
            en.set_pattern('^.*\s+PROGRAM ENDED AT(.*)', convert_func = lambda t: True)
            all_en.append(en)

        elif ftype == 'qchem':
            all_en = []

            en = QMEntry('Energy', None)
            en.set_pattern('^.*\s+total energy =\s+(%s) au'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_Total', None)
            en.set_pattern('^\s+RIMP2\s+total energy =\s+(%s) au'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_SS', None)
            en.set_pattern('^\s+TOTAL SS RI-MP2 ENERGY\s+=\s+(%s) au'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_OS', None)
            en.set_pattern('^\s+TOTAL OS RI-MP2 ENERGY\s+=\s+(%s) au'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('Time', None)
            en.set_pattern('^\s+Total job time:\s+(%s)s\(wall\)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('Success', False)
            en.set_pattern('^\s+\*\s+Thank you very much for using Q-Chem\. (.*)\*', convert_func = lambda t: True)
            all_en.append(en)
            pass
        elif ftype == 'tinker':
            all_en = []

            en = QMEntry('Energy', None)
            en.set_pattern('^ Total Potential Energy :\s+(%s)\s+\S+'%(self.REGEX_FLOAT))
            all_en.append(en)

        elif ftype == 'psi4':
            all_en = []

            en = QMEntry('Energy', None)
            en.set_pattern('^\s+Total Energy =\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('SCS-MP2', None)
            en.set_pattern('^\s+SCS Total Energy\s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_Ref', None)
            en.set_pattern('^\s+Reference Energy  \s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_corl', None)
            en.set_pattern('^\s+MP2 correlation energy  \s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('CCSD_corl', None)
            en.set_pattern('^\s+CCSD correlation energy  \s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('CCSD_ref', None)
            en.set_pattern('^\s+Reference energy    (file100)\s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('T_ener', None)
            en.set_pattern('^\s+\(T\) energy\s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_SS', None)
            en.set_pattern('^\s+Same-Spin Energy  \s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('MP2_OS', None)
            en.set_pattern('^\s+Opposite-Spin Energy  \s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('DF-MP2', None)
            en.set_pattern('^\s+Total Energy \s+= \s+(%s)'%(self.REGEX_FLOAT), '\s+\S+ DF-MP2 Energies \S+', [5, 6, 7, 8, 9, 10])
            #en.set_pattern('^\s+Total Energy   \s+=  \s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('E2B', None)
            en.set_pattern('^\s+2\s+\S+\s+(%s)\s+\S+'%(self.REGEX_FLOAT), '^\s+n-Body\s+Total Energy', [2])
            all_en.append(en)

            en = QMEntry('EDH', None)
            en.set_pattern('^\s+@Final double-hybrid DFT total energy\s+=\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('ECBS', None)
            en.set_pattern('^\s+total\s+CBS\s+(%s)'%(self.REGEX_FLOAT))
            all_en.append(en)

            en = QMEntry('SNS-MP2', None)
            en.set_pattern('^\s+(%s)\s+\S+'%(self.REGEX_FLOAT), '^\s+ SNS-MP2 Interaction Energy ', [1])
            all_en.append(en)

            en = QMEntry('MaxForce', None)
            en.set_pattern('^ Maximum\s+Force\s+(%s)\s+%s'%(self.REGEX_FLOAT, self.REGEX_FLOAT))
            #all_en.append(en)

            en = QMEntry('RMSForce', None)
            en.set_pattern('^ RMS\s+Force\s+(%s)\s+%s'%(self.REGEX_FLOAT, self.REGEX_FLOAT))
            #all_en.append(en)

            en = QMEntry('Freq1', None)
            en.set_pattern('^ Low frequencies ---\s+(%s)\s+'%(self.REGEX_FLOAT), '^ Full mass-weighted force constant matrix', [1])
            #all_en.append(en)

            en = QMEntry('Time', None)
            en.set_pattern('^\s+total time\s+=\s+(%s)\s+seconds'%(self.REGEX_FLOAT), '^Total time', [3])
            all_en.append(en)

            en = QMEntry('Success', False)
            en.set_pattern('^\*\*\* Psi4 exiting successfully(.*)', convert_func = lambda t: True)
            all_en.append(en)

            en = QMEntry('Error', '')
            #en.set_pattern('^ Error termination request processed by (.*).', convert_func = lambda t: t)
            en.set_pattern('^ Error termination via Lnk1e in (\S+)', convert_func = lambda t: t)
            #all_en.append(en)
            pass
        self.entry_format.clear()
        self.qm_columns = []
        for en in all_en:
            self.qm_columns.append(en.name)
            self.entry_format[en.name] = en
        self.qm_data = pd.DataFrame(columns = self.qm_columns)

    def set_coord_pattern(self, ftype = 'g16'):
        '''
        RE for start of coord,
        RE for coord line that returns named groups ('x', 'y', 'z'),
        function that takes (iline_start, iline, natoms) and returns whether this line can be a coord line
        '''
        if ftype == 'g16':
            self.regex_frame_start = '^\s{20,30}Input orientation:'
            self.regex_coord_start = '^\s{20,30}Input orientation:'
            self.regex_coord_line  = '^\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)'
            self.is_coord_line = lambda t: 5 <= (t[0] - t[1]) and (t[0] - t[1]) < t[2]+5
        if ftype == 'g16std':
            self.regex_frame_start = '^\s{20,30}Standard orientation:'
            self.regex_coord_start = '^\s{20,30}Standard orientation:'
            self.regex_coord_line  = '^\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)'
            self.is_coord_line = lambda t: 5 <= (t[0] - t[1]) and (t[0] - t[1]) < t[2]+5
        if ftype == 'g16cp':
            self.regex_frame_start = '^ Counterpoise: doing'
            self.regex_coord_start = '^\s{20,30}Standard orientation:'
            self.regex_coord_line  = '^\s+\d{1,3}\s+\d{1,3}\s+\d{1,3}\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)'
            self.is_coord_line = lambda t: 5 <= (t[0] - t[1]) and (t[0] - t[1]) < t[2]+5
        elif ftype == 'qchem':
            self.regex_frame_start = '^\s+Standard Nuclear Orientation'
            self.regex_coord_start = '^\s+Standard Nuclear Orientation'
            self.regex_coord_line  = '^\s+(?P<i>\d{1,3})\s+(?P<n>\S{1,3})\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)'
            self.is_coord_line = lambda t: 3 <= (t[0] - t[1]) and (t[0] - t[1]) < t[2]+3
        elif ftype == 'psi4':
            self.regex_frame_start = '^\*\*\* tstart\(\) called'
            self.regex_coord_start = '^\s+Center\s+X\s+Y\s+Z\s+'
            self.regex_coord_line  = '^\s+\S{1,7}\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)'
            self.is_coord_line = lambda t: 2 <= (t[0] - t[1]) and (t[0] - t[1]) < t[2]+2

        if self.regex_coord_line is not None:
            self.re_frame_start = re.compile(self.regex_frame_start)
            self.re_coord_start = re.compile(self.regex_coord_start)
            self.re_coord_line = re.compile(self.regex_coord_line)

    def read_qm_top(self, inpf, ftype='g16'):
        self.set_coord_pattern(ftype)
        MAXSIZE=1000000
        iframe = 0
        with open(inpf, 'r') as fin:
            iline = -1
            self.natoms = MAXSIZE
            iatom = self.natoms
            for line in fin:
                iline += 1
                if self.re_frame_start.match(line):
                    iframe += 1
                if self.re_coord_start.match(line):
                    i_coord_start = iline
                    iatom = 0
                if iatom < self.natoms and self.is_coord_line((iline, i_coord_start, self.natoms)):
                    match = self.re_coord_line.match(line)
                    if match:
                        iatom += 1
                        _x, _y, _z = match.group('x', 'y', 'z')
                        idx = iatom
                        if 'i' in match.groupdict():
                            idx = int(match.group('i'))
                        if 'n' in match.groupdict():
                            name = match.group('n')
                            if idx in self.topology.index:
                                break
                            self.topology.loc[idx, 'Name'] = name
            self.natoms = len(self.topology)
            self.reset_frames()
            self.fill_missing()
                                

    def read_qm(self, inpf, ftype='g16'):
        self.set_entry_pattern(ftype)
        self.set_coord_pattern(ftype)

        if self.natoms == 0:
            self.read_qm_top(inpf, ftype=ftype)

        i_coord_start = -100
        iframe = self.nframes - 1
        with open(inpf, 'r') as fin:
            iline = -1
            iatom = self.natoms
            for line in fin:
                iline += 1
                if self.re_frame_start.match(line):
                    iframe += 1
                if self.re_coord_start.match(line):
                    i_coord_start = iline
                    iatom = 0
                    curr_frame = []
                if iatom < self.natoms and self.is_coord_line((iline, i_coord_start, self.natoms)):
                    match = self.re_coord_line.match(line)
                    if match:
                        iatom += 1
                        _x, _y, _z = match.group('x', 'y', 'z')
                                
                        #curr_frame.append([float(_x), float(_y), float(_z)])
                        curr_frame.extend([float(_x), float(_y), float(_z)])
                        #print(_x, _y, _z)
                        if iatom == self.natoms:
                            #self.frames.append(np.asarray(curr_frame))
                            self.frames = np.append(self.frames[:self.nframes, :self.natoms, :], np.array(curr_frame).reshape((-1, self.natoms, 3)), axis=0)
                            self.nframes += 1
                for entry_name in self.qm_columns:
                    en = self.entry_format[entry_name]
                    value = en.process_line(iline, line)
                    if value is not None:
                        self.qm_data.loc[iframe, entry_name] = value
                        #print(iline, entry_name, value)
                        
            self.max_size = self.frames.shape
            self.nframes = len(self.frames)
            self.iframe = iframe
            self.assign_geo(self)

    def find_best_frame(self):
        '''
        Currently find the frame with lowest energy
        Can be adapted for other purpose
        '''
        ndata = len(self.qm_data)
        if ndata == 0:
            return
        sub_idx = self.qm_data.index

        if 'Success' in self.qm_data and sum(self.qm_data['Success'] == True) > 0:
            sub_idx = self.qm_data.index[self.qm_data['Success'] == True]

        if 'Energy' in self.qm_data:
            try:
                iframe = self.qm_data.loc[sub_idx, 'Energy'].idxmin()
            except:
                # the idxmin() method may have a bug when there are multiple NaN
                e_min = self.MAX_VALUE
                i_min = sub_idx[0]
                for iframe in sub_idx:
                    value = self.qm_data.loc[iframe, 'Energy']
                    if np.isreal(value) and np.isfinite(value) and value < e_min:
                        e_min = value
                        i_min = iframe
                iframe = i_min
        else:
            iframe = sub_idx[-1]
        self.iframe = iframe
        self.coord = self.frames[iframe]


if __name__ == '__main__':
    q0 = QMResult()
    q0.read_input('scratch/Acetonitrile-Water_2.xyz', ftype='xyz')
    q0.read_qm('scratch/Acetonitrile-Water_2.out', ftype='g16')
    q0.read_qm('scratch/Acetonitrile-Water_1.out', ftype='g16')

    q0.qm_data.to_csv('7.csv')
    q0.find_best_frame()


    q1 = QMInput()
    q1.get_template()
    q1.assign_geo(q0)
    q1.write_qm('7.gjf', theory='opt/b3lyp')


    #q0.read_input('scratch/Trimethylammonium-Acetate_1.xyz', ftype='xyz')
    #q0.read_qm('scratch/Trimethylammonium-Acetate_1.out', ftype='g16')
    #q0.read_qm('scratch/Trimethylammonium-Acetate_1.out', ftype='xyz')
    #print(q0.nframes)
