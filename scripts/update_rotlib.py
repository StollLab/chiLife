#! /usr/bin/env python
from zipfile import ZipFile
from io import BytesIO
import numpy as np
from numpy.lib.format import read_array
import chilife as xl
import oldProteinIC as opic
import sys, os
from importlib import reload
import shutil
import argparse
import zipfile

parser = argparse.ArgumentParser(prog='update_rotlibs',
                                 description='Takes old (incompatible) versions of chiLife rotlibs and creates updated '
                                             '(compatible) versions.',
                                 )
parser.add_argument('files_names', nargs='+', help='Names of the old rotlib files you wish to update')

args = parser.parse_args()


def main():


    for filename in args.files_names:

        if filename.endswith('_rotlib.npz'):
            shutil.copy(filename, filename + '.bku')
            data = get_data(filename)
            np.savez(filename, **data)


        elif filename.endswith('_drotlib.zip'):
            shutil.copy(filename, filename + '.bku')
            libname = filename[:-12]

            with zipfile.ZipFile(filename, 'r') as archive:
                for f in archive.namelist():
                    if 'csts' in f:
                        with archive.open(f) as of:
                            csts = np.load(of)
                            np.save(f'{libname}_csts.npy', csts)
                    elif f[-12] == 'A':
                        with archive.open(f) as of:
                            libA = get_data(of)
                            np.savez(f'{libname}A_rotlib.npz', **libA)
                    elif f[-12] == 'B':
                        with archive.open(f) as of:
                            libB = get_data(of)
                            np.savez(f'{libname}B_rotlib.npz', **libB)

            with zipfile.ZipFile(f'{libname}_drotlib.zip', mode='w') as archive:
                archive.write(f'{libname}A_rotlib.npz')
                archive.write(f'{libname}B_rotlib.npz')
                archive.write(f'{libname}_csts.npy')

            os.remove(f'{libname}A_rotlib.npz')
            os.remove(f'{libname}B_rotlib.npz')
            os.remove(f'{libname}_csts.npy')

        else:
            raise RuntimeError(f'{filename} is not a valid chiLife rotlib. All valid chiLife rotlibs end with '
                               f'`_rotlib.npz` or `_drotlib.zip`.')



def get_data(filename, description = None, comment = None, reference = None):

    with np.load(filename, allow_pickle=True) as f:
        version = f['format_version']

    if version < 1.2:
        data = get_data_lt_1_2(filename)

    elif version == 1.2:
        data = get_data_1_2(filename)

    else:
        data = {}
        with np.load(filename, allow_pickle=True) as f:
            for key in f:
                data[key] = f[key]

    for key, val in zip(('description', 'comment', 'reference'), (description, comment, reference)):
        if (key not in data) or (val is not None):
            data[key] = val

    data['format_version'] = 1.3

    return data

def get_data_lt_1_2(filename):
    sys.modules['chilife.ProteinIC'] = opic
    data = {}
    with np.load(filename, allow_pickle=True) as f:
        for key in f:
            data[key] = f[key]

    IC0 = data['internal_coords'][0]
    bonds = IC0.bonded_pairs
    z_matrix_idxs = IC0.zmat_idxs[1]

    z_matrix = []
    for ic in data['internal_coords']:
        z_matrix.append(ic.zmats[1])

    z_matrix = np.array(z_matrix)
    z_matrix_idxs = np.c_[np.arange(len(z_matrix_idxs)), z_matrix_idxs]
    ic_coords = np.array([ic.coords for ic in data['internal_coords']])

    anames = IC0.atom_names
    atypes = IC0.atom_types
    resnames = np.array([val for val in IC0.resnames.values()])
    resindices = np.array([0 for a in IC0.atoms])
    resnums = IC0.resis.copy()
    segindices = np.array([0])
    segids = ['A']
    uni = xl.make_mda_uni(anames, atypes, resnames, resindices, resnums, segindices, segids)
    uni.load_new(ic_coords)

    del sys.modules['chilife.ProteinIC']
    reload(xl)
    P = xl.MolSysIC(z_matrix, z_matrix_idxs, uni, bonds=bonds)
    data['internal_coords'] = P

    return data

def get_data_1_2(filename):

    with ZipFile(filename, 'r') as fb:
        with fb.open('internal_coords.npy', 'r') as finner:
            ic_bytes = finner.read()

    ic_bytes = ic_bytes.replace(b'chilife.Protein\n', b'chilife.MolSys\n')
    ic_bytes = ic_bytes.replace(b'\nProtein\n', b'\nMolSys\n')
    ic_bytes = ic_bytes.replace(b'chilife.ProteinIC\nProteinIC', b'chilife.MolSysIC\nMolSysIC')

    data = {}
    with np.load(filename, allow_pickle=True) as f:
        for key in f:
            if key != 'internal_coords':
                data[key] = f[key]

    data['internal_coords'] = read_array(BytesIO(ic_bytes), allow_pickle=True)

    return data

main()