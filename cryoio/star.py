"""
Metadata I/O

The STAR format

RELION uses the STAR (Self-defining Text Archiving and Retrieval) format (Hall, Allen and Brown, 1991) for the storage of label-value pairs for all kinds of input and output metadata. The STAR format is an alternative to XML, but it is more readable and occupies less space. The STAR format has been adopted by the crystallographic community in the form of CIF (Crystallographic Information Framework), and Bernard Heymann's BSOFT package was the first to use STAR in the field of 3D-EM. Also Xmipp-3.0 now uses the STAR format.
RELION's implementation of the STAR format has the following rules (partially copied from BSOFT's manual):
The file name must end in a ".star" extension.
Each file must have one or more data blocks. The start of a data block is defined by the keyword "data_" followed by an optional string for identification (e.g., "data_images").
Multiple values associated with one or more labels in a data block can be arranged in a table using the keyword "loop_" followed by the list of labels and columns of values. The values are delimited by whitespace (i.e., blanks, tabs, end-of-lines and carriage returns). The loop must be followed by an empty line to indicate its end.
Label names always starts with an underscore ("_"). Each label may only be used once within each data block.
Data items or values can be numeric or strings of characters. A string is interpreted as a single item when it doesn't contain spaces
Comments are strings which can occur in three places:
File comments: All text before the first "data_" keyword
Data block comments: Strings on their own lines starting with "#" or with ";" as the first character in the line.
Item comments: Strings on the same line as and following tag-value items, also indicated by a leading "#".
Metadata label definitions

RELION has its own set of defined metadata labels. The command relion_refine --print_metadata_labels will print a list of the definitions of all of them.
An example

A STAR file that could be used as input for refinement in RELION that includes CTF information about each particle could look like this:
 data_images
loop_
_rlnImageName
_rlnDefocusU
_rlnDefocusV
_rlnDefocusAngle
_rlnVoltage
_rlnAmplitudeContrast
_rlnSphericalAberration
000001@/lmb/home/scheres/data/VP7/all_images.mrcs 13538 13985 109.45 300 0.15 2
000002@/lmb/home/scheres/data/VP7/all_images.mrcs 13293 13796 109.45 300 0.15 2
000003@/lmb/home/scheres/data/VP7/all_images.mcrs 13626 14085 109.45 300 0.15 2
"""
from copy import deepcopy
import numpy as np

def return_scalar(string):
    """
    Return scalar if input string can transform to a scalar
    Can not deal with too large number (float)
    """
    try:
        scalar = float(string)
        return scalar
    except ValueError:
        return string

def isscalar(string):
    """
    Return True / False if input string can transform to a scalar
    Can not deal with too large number (float)
    """
    try:
        float(string)
        return True
    except ValueError:
        return False

def readSTAR(fname):
    """
    Read a STAR file. Still can not deal with muliple data blocks now
    Return data in dict
    """
    block_prefix = 'data_'
    table_marker = 'loop_'
    label_prefix = '_rln'
    with open(fname, 'r') as f:
        # read name of data block
        while True:
            line = f.readline().split()
            if line:
                if block_prefix in line[0]:
                    block_name = line[0][4:]
                    break
        # read label names
        label_names = list()
        metadata_dict = dict()
        values_in_table = False
        blank_line_break = False
        while True:
            position = f.tell()
            line = f.readline().split()
            if line and not blank_line_break:
                blank_line_break = True
            elif not line and blank_line_break:
                break # break when meet blank line again
            elif not line:
                continue # skip when meet blank line
            if values_in_table is False:
                # read values without table format
                if table_marker in line[0]:
                    values_in_table = True
                elif label_prefix in line[0]:
                    label = line[0][4:]
                    label_names.append(label)
                    value = return_scalar(line[-1])
                    metadata_dict[label] = value
                else:
                    break
            elif values_in_table is True:
                # read label names
                if label_prefix in line[0]:
                    label = line[0][4:]
                    label_names.append(label)
                    metadata_dict[label] = list()
                else:
                    break
        # read values in table
        f.seek(position)
        blank_line_break = False
        if values_in_table:
            while True:
                line = f.readline().split()
                if line and not blank_line_break:
                    blank_line_break = True
                elif not line and blank_line_break:
                    break # break when meet blank line again
                elif not line:
                    continue # skip when meet blank line
                for i, label in enumerate(label_names):
                    value = return_scalar(line[i])
                    metadata_dict[label].append(value)
    return metadata_dict

def writeSTAR(fname, imgs_path=None, block_name=None, **metadata_dict):
    """
    Write a STAR file in relion recommended format. Only write one data block now
    """
    # Class Dict is a mutable object, this function contain codes will revise dict values.
    # Hence, a deep copy is necessary.
    metadata_copy = deepcopy(metadata_dict)
    label_names = metadata_copy.keys()
    label_values = metadata_copy.values()
    label_length = list()
    for value in label_values:
        label_length.append(len(value))
    var_length = np.asarray(label_length).var()
    assert var_length == 0.0
    length = np.mean(label_length, dtype=int)
    label_prefix = '_rln'
    start_idx = 1 # image indice in star file start from 1 (not 0)
    if 'ImageName' in label_names and imgs_path:
        for i, value in enumerate(metadata_copy['ImageName']):
            metadata_copy['ImageName'][i] = str(int(value + start_idx)).zfill(6) \
                                            + '@' + str(imgs_path)
    with open(fname, 'w') as f:
        # write data name
        if block_name:
            f.write('\n' + 'data_' + str(block_name) + '\n' + '\n')
        else:
            f.write('\n' + 'data_ \n' + '\n')
        # write label
        f.write('loop_' + '\n')
        for i, label in enumerate(label_names):
            f.write(label_prefix + label + ' #' + str(i+1) + '\n')
        # write values
        for idx in range(length):
            data = list()
            for value in label_values:
                if isscalar(value[idx]):
                    data.append('{0:>10.6f}'.format(value[idx]))
                else:
                    data.append(value[idx])
            line = ' '.join(data)
            f.write(line + '\n')

def easy_writeSTAR(fname, EAs=None, shifts=None, imgs_path=None, block_name=None):
    """
    keywords for Euler angles and shifts
    AngleRot, AngleTilt, AnglePsi, OriginX, OriginY
    """
    import os
    print(os.path.abspath('.'))
    if not EAs and not shifts:
        raise ValueError('please specify input data: Euler angles or shifts')
    else:
        metadata = dict()
    if EAs:
        if isinstance(EAs, list):
            EAs = np.asarray(EAs)
        assert EAs.shape[1] == 3
        metadata['AngleRot'] = EAs[:, 0]
        metadata['AngleTilt'] = EAs[:, 1]
        metadata['AnglePsi'] = EAs[:, 2]
    if shifts:
        if isinstance(shifts, list):
            shifts = np.asarray(shifts)
        assert shifts.shape[1] == 2
        metadata['OriginX'] = shifts[:, 0]
        metadata['OriginY'] = shifts[:, 1]
    writeSTAR(fname, imgs_path, block_name, **metadata)

# ===== Utility Functions ===== #
def get_indices_from_star(fname):
    """
    Return Euler Angles (phi, theta, psi) from a STAR file or a Metadata dict
    """
    start_idx = 0
    if isinstance(fname, dict):
        metadata_dict = fname
    else:
        metadata_dict = readSTAR(fname)
    try:
        length = len(metadata_dict['ImageName'])
        indices = [i+start_idx for i in range(length)]
    except KeyError:
        raise KeyError("This star file does't exist ImageName label")
    # get_indices_from_star(fname, inc_imgs_path=False)
    # if fname is dict like object, it conflict with 'inc_imgs_path=False'
    # if inc_imgs_path:
    #     line = metadata_dict['ImageName'][0].split('@')
    #     try:
    #         imgs_path = line[1]
    #     except IndexError:
    #         print("This star file probably doesn't exist imgs_path")
    #         imgs_path = ''
    #     return indices, imgs_path
    # else:
    #     return indices
    return indices

def get_imgs_path_from_star(fname):
    """
    Return imgs_path from a STAR file or a Metadata dict
    """
    if isinstance(fname, dict):
        metadata_dict = fname
    else:
        metadata_dict = readSTAR(fname)
    if 'ImageName' in metadata_dict.keys():
        try:
            imgs_path = metadata_dict['ImageName'][0].split('@')[1]
            return imgs_path
        except IndexError:
            return None
    else:
        return None

def get_EAs_from_star(fname):
    """
    Return Euler Angles (phi, theta, psi) from a STAR file or a Metadata dict
    """
    if isinstance(fname, dict):
        metadata_dict = fname
    else:
        metadata_dict = readSTAR(fname)
    try:
        rot = metadata_dict['AngleRot']
    except KeyError:
        raise KeyError("This star file does't exist rotation(phi) angles.")
    try:
        tilt = metadata_dict['AngleTilt']
    except KeyError:
        raise KeyError("This star file does't exist tilt(theta) angles.")
    try:
        psi = metadata_dict['AnglePsi']
    except KeyError:
        raise KeyError("This star file does't exist inplane-rotation(psi) angles.")
    return rot, tilt, psi

def get_shift_from_star(fname):
    """
    Return Euler Angles (phi, theta, psi) from a STAR file or a Metadata dict
    """
    if isinstance(fname, dict):
        metadata_dict = fname
    else:
        metadata_dict = readSTAR(fname)
    try:
        xshift = metadata_dict['OriginX']
    except KeyError:
        raise KeyError("This star file does't exist X shift.")
    try:
        yshift = metadata_dict['OriginY']
    except KeyError:
        raise KeyError("This star file does't exist Y shift")
    return xshift, yshift
