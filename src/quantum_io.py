#
import numpy as np
import copy
from geom_io import GeomFile, GeomConvert
import pandas as pd
import re
class QMEntry:
    def __init__(self):
        self.entry_name = ''
        self.iline_start = -1
        self.regex_start = None
        self.regex_value = None
        self.value_ilines = [0] # line numbers relative to the start line

    def set_pattern(self, p_start, p_value, line_number = [0]):
        self.regex_start = p_start
        self.regex_value = p_value

        self.re_start = re.compile(self.regex_start)
        self.re_value = re.compile(self.regex_value)
        self.value_ilines = line_number

    def find_start(self, iline, line):
        pass
    pass

class QMResult(GeomConvert):
    def __init__(self):
        super(QMResult, self).__init__()

        self.qm_columns = 'Energy Freq1 Error MaxForce RMSForce MaxDisp RMSDisp'.split()
        self.qm_defaults = None,  None, 0,    None,    None,    None,   None
        self.info_frames = pd.DataFrame(columns = self.qm_columns)
        self.i_coord_start = -100

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

        self.re_frame_start = re.compile(self.regex_frame_start)
        self.re_coord_start = re.compile(self.regex_coord_start)
        self.re_coord_line = re.compile(self.regex_coord_line)

    def read_qm(self, inpf, ftype='g16'):
        self.set_coord_pattern(ftype)

        i_coord_start = -100
        iframe = self.nframes - 1
        with open(inpf, 'r') as fin:
            iline = -1
            iatom = self.top_natoms
            for line in fin:
                iline += 1
                if self.re_frame_start.match(line):
                    iframe += 1
                if self.re_coord_start.match(line):
                    i_coord_start = iline
                    iatom = 0
                    curr_frame = []
                if iatom < self.top_natoms and self.is_coord_line((iline, i_coord_start, self.top_natoms)):
                    match = self.re_coord_line.match(line)
                    if match:
                        iatom += 1
                        _x, _y, _z = match.group('x', 'y', 'z')
                        curr_frame.append([_x, _y, _z])
                        #print(_x, _y, _z)
                        if iatom == self.top_natoms:
                            self.frames.append(np.asarray(curr_frame))
            self.iframe = iframe
            self.assign_geo(self)



if __name__ == '__main__':
    pass
    q0 = QMResult()
    q0.read_input('examples/Acetonitrile-Water_2.xyz', ftype='xyz')
    q0.read_qm('examples/Acetonitrile-Water_2.out', ftype='g16')
    #print(q0.nframes)
