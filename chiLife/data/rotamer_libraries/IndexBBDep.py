import pickle
"""Below is the start of the code that should become the new read bbdep library function. Storing seek locations and 
bytearray lengths to read from the file should be a lot faster than the sqlite3 database I am currently using"""

resinames = {'ARG', 'ASN', 'ASP', 'CPR', 'CYD', 'CYH',
             'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU',
             'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
             'TPR', 'TRP', 'TYR', 'VAL', 'R1C'}
index = {}
ActiveResi = b'ARG  -180 -180'
StartResi = b'ARG'
start = 1492  # Starting byte for ALL.bbdep.rotamres.lib
with open('ALL.bbdep.rotamers.lib') as f:
    f.seek(0)
    line = f.readline()
    while line:
        if line[:3] in resinames:

            if line[:14] != ActiveResi:
                length = f.tell() - len(line) - start - 1
                index[ActiveResi] = (start, length)

                ActiveResi = line[:14]
                start = f.tell() - len(line) - 1
        line = f.readline()

start = 0
with open('R1C.lib') as f:
    f.seek(0)
    line = f.readline()
    while line:
        if line[:3] in resinames:

            if line[:14] != ActiveResi:
                length = f.tell() - len(line) - start - 1
                index[ActiveResi] = (start, length)

                ActiveResi = line[:14]
                start = f.tell() - len(line) - 1
        line = f.readline()


with open('RotlibIndexes.pkl', 'wb') as f:
    pickle.dump(index, f)