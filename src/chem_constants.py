
ELEMENT_NAME = \
        ('H' ,'He',                                      # 1st
         'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',        # 2nd
         'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',        # 3rd
         'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn',
                   'Ga','Ge','As','Se','Br','Kr',        # 4th
         'Rb','Sr',                                        'Ag',
                   'In','Sn','Sb','Te','I' ,'Xe',        # 5th
         'Cs','Ba',                                   'Pt','Au','Hg',
                                                         # 6th
         )
ELEMENT_NUMBER = \
        (1,   2,
         3,   4,   5,   6,   7,   8,   9,   10,
         11,  12,  13,  14,  15,  16,  17,  18,
         19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
                   31,  32,  33,  34,  35,  36,
         37,  38,                                          47,
                   49,  50,  51,  52,  53,  54,
         55,  56,                                     78,  79,  80,
         )
NR_2_ELE = dict(zip(ELEMENT_NUMBER, ELEMENT_NAME))
ELE_2_NR = dict(zip(ELEMENT_NAME, ELEMENT_NUMBER))
