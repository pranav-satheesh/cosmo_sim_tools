""" Illustris Simulation: Public Data Release.
snapshot.py: File I/O related to the snapshot files. """
from __future__ import print_function

import numpy as np
import h5py
import six
from .util import partTypeNum
from .groupcat import gcPath, offsetPath


def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    basePath = basePath.rstrip('/')  # Remove trailing slash if present
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath


def snapPath_groupordered(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    basePath = basePath.rstrip('/')  # Remove trailing slash if present
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap-groupordered_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath


def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart


def loadSubset_groupordered(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath_groupordered(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath_groupordered(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(snapPath_groupordered(basePath, snapNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number (or warn if partial read)
    if origNumToRead != wOffset:
        if wOffset > 0:
            print(f"Warning: Read [{wOffset}] particles, but was expecting [{origNumToRead}]. Some particles may be in missing snapshot chunks.")
        else:
            raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result

def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum_array = np.where(offsetsThisType >= 0)[0]
            if len(fileNum_array) > 0:
                fileNum = int(np.max(fileNum_array))
                # Bounds check: ensure fileNum doesn't exceed available files
                numFiles = subset['snapOffsets'].shape[1]
                fileNum = min(fileNum, numFiles - 1)
            else:
                fileNum = 0
            
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        try:
            f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')
        except FileNotFoundError:
            # Reached the end of available files; remaining particles may not exist
            if numToRead > 0:
                print(f"Warning: Could not read all {origNumToRead} particles. Only read {wOffset}.")
                break
            f.close()
            break

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        # Safety check: ensure fileOff is valid
        if fileOff < 0 or fileOff >= numTypeLocal:
            print(f"Warning: Invalid file offset {fileOff} for file {fileNum} with {numTypeLocal} particles of type {ptNum}. Skipping.")
            f.close()
            fileNum += 1
            fileOff = 0
            continue

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        if numToReadLocal < 0:
            print(f"Warning: Negative read length {numToReadLocal}. Skipping and continuing to next file.")
            f.close()
            fileNum += 1
            fileOff = 0
            continue

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number (or warn if partial read)
    if origNumToRead != wOffset:
        if wOffset > 0:
            print(f"Warning: Read [{wOffset}] particles, but was expecting [{origNumToRead}]. Some particles may be in missing snapshot chunks.")
        else:
            raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result


def getSnapOffsets_groupordered(basePath, snapNum, id, type_field):
    """ OLD: Compute offsets within snapshot for group/subhalo using offset files (assumes FileOffsets/Group hierarchy). """
    r = {}

    # Load header info to get number of files
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        numFiles = int(header['NumFiles'])
    
    # Try old nested hierarchy first (FileOffsets/Group or FileOffsets/Subhalo)
    try:
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            if type_field == 'Group':
                gGroupName = 'FileOffsets/Group'
            elif type_field == 'Subhalo':
                gGroupName = 'FileOffsets/Subhalo'
            else:
                raise ValueError(f"Unknown type_field: {type_field}")
            
            # only groups at id%the number of files
            r['snapOffsets'] = np.array(f[gGroupName][str(id)], dtype='int64')
            fileOff = r['snapOffsets'][0]
            fileNum = r['snapOffsets'][1]
            
            return r['snapOffsets']
    except (KeyError, OSError):
        print(f"Warning: Could not load offsets using old nested hierarchy for {type_field}:{id}. Trying flat format...")
        
        # Try new flat structure
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            if 'FileOffsets' in f:
                offsets_array = np.array(f['FileOffsets'], dtype='int64')
                if type_field == 'Group':
                    type_key = 'GroupID'
                elif type_field == 'Subhalo':
                    type_key = 'SubhaloID'
                else:
                    type_key = type_field
                
                if type_key in f:
                    ids = np.array(f[type_key], dtype='int64')
                    match_idx = np.where(ids == id)[0]
                    if len(match_idx) > 0:
                        offset_idx = match_idx[0]
                        return offsets_array[offset_idx]
        
        raise Exception(f"Could not find offsets for {type_field}:{id}")


def loadSubhalo_groupordered(basePath, snapNum, id, partType, fields=None, subset=None, mdi=None, sq=True, float32=True):
    """Load a subhalo from group-ordered snapshots using explicit particle offsets.

    subset must contain lenType and offsetType arrays of length six and a
    snapOffsets array of shape (6, number_of_chunks), all for the ordered files.
    The caller must supply offsets for id; this function does not infer them.
    Missing offsets raise ValueError rather than returning the whole snapshot.
    """
    if subset is None:
        raise ValueError(
            "Group-ordered subhalo loading requires explicit subset offsets for "
            f"subhalo {id}: lenType, offsetType, and snapOffsets."
        )
    required = {'lenType', 'offsetType', 'snapOffsets'}
    if not isinstance(subset, dict) or not required.issubset(subset):
        raise ValueError("subset must contain lenType, offsetType, and snapOffsets")
    subset = {key: np.asarray(subset[key]) for key in required}
    if (subset['lenType'].shape != (6,) or subset['offsetType'].shape != (6,)
            or subset['snapOffsets'].ndim != 2
            or subset['snapOffsets'].shape[0] != 6
            or subset['snapOffsets'].shape[1] == 0):
        raise ValueError("Expected length-six lenType/offsetType and (6, N) snapOffsets")
    if any(not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)
           for value in subset.values()):
        raise ValueError("Subset lengths and offsets must be nonnegative integers")
    if (np.any(subset['snapOffsets'][:, 0] != 0)
            or np.any(np.diff(subset['snapOffsets'], axis=1) < 0)):
        raise ValueError("snapOffsets must start at zero and increase monotonically")
    return loadSubset_groupordered(
        basePath, snapNum, partType, fields=fields, subset=subset,
        mdi=mdi, sq=sq, float32=float32,
    )


def getSnapOffsets(basePath, snapNum, id, type):
    """Read the catalog offsets needed by ordinary snapshot halo loaders.

    Return lenType, offsetType, and snapOffsets using the existing legacy/public
    offset layouts. Missing metadata raises an error; offsets are never guessed
    from cumulative subhalo lengths, which omit unbound particles.
    """
    if type not in ('Group', 'Subhalo'):
        raise ValueError("type must be 'Group' or 'Subhalo'")
    if not isinstance(id, (int, np.integer)) or id < 0:
        raise ValueError("id must be a nonnegative integer")
    try:
        return getSnapOffsets_old(basePath, snapNum, id, type)
    except (KeyError, OSError) as exc:
        raise ValueError(
            f"Cannot load {type} {id} at snapshot {snapNum}: required catalog "
            "offset metadata is missing or unreadable. Supply the matching "
            "offset files; group-ordered offsets use a different layout."
        ) from exc


def getSnapOffsets_old(basePath, snapNum, id, type):
    """ OLD VERSION - Compute offsets within snapshot for a particular group/subgroup. """
    r = {}

    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/'+type][()]
            r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
            r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset, :]

    # old or new format: load the offset (by type) of this group/subgroup within the snapshot
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            r['offsetType'] = f[type+'/SnapByType'][id, :]
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
            r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

    return r


def loadSubhalo(basePath, snapNum, id, partType, fields=None):
    """Load selected particle fields for one subhalo from ordinary snapshots.

    Resolve catalog offsets with getSnapOffsets and return loadSubset output.
    Missing offset metadata raises an error instead of reading unrelated particles.
    """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadSubhalo_old(basePath, snapNum, id, partType, fields=None):
    """ OLD VERSION - Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets_old(basePath, snapNum, id, "Subhalo")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadHalo(basePath, snapNum, id, partType, fields=None):
    """Load selected particle fields for one FoF halo from ordinary snapshots.

    Resolve catalog offsets with getSnapOffsets and return loadSubset output.
    Missing offset metadata raises an error instead of reading unrelated particles.
    """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadHalo_old(basePath, snapNum, id, partType, fields=None):
    """ OLD VERSION - Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets_old(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)
