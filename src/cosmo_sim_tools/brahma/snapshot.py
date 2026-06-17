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


def loadSubset_groupordered(basePath, snapNum, id, partType, fields=None, subset=None, mdi=None, sq=True, float32=True):
    ''' Load subset of [partType] particles in groups/subhalos.
    Works with new file format (group-ordered snapshots). 
    
    Parameters
    ----------
    basePath : str
        Simulation directory (containing snapdir_###/...) 
    snapNum : int
        Snapshot number
    id : int
        Group/Subhalo ID
    partType : int
        Particle type code (0-5): gas, dm, wind, stars, bh, etc.
    fields : list
        Fields to load (default: all)
    subset : dict
        Subset specification with keys like 'lenType' and 'offsetType'
    mdi : list
        Multi-dimensional index for fields
    sq : bool
        Single query flag
    float32 : bool
        Convert float64 to float32? Default True for memory saving.
    
    Returns
    -------
    dict or np.ndarray : Loaded particle data
    ''' 
    result = {}

    # Default group name for particle type
    gName = "PartType" + str(partType)

    # Determine file counts
    with h5py.File(snapPath(basePath, snapNum, 0), 'r') as f:
        nPartType = f['Header'].attrs['NumPart_ThisFile']
        nPart = f['Header'].attrs['NumPart_Total']

    # If loading via group-ordered subsets, get offset info from group catalog
    if subset is None:
        # Use group catalog to get subset info
        try:
            with h5py.File(gcPath(basePath, snapNum), 'r') as f:
                # Get number of files
                header = dict(f['Header'].attrs.items())
                numFiles = int(header['NumFiles'])
                
                # Compute offsets from scratch using getSnapOffsets
                offsetsThisType = getSnapOffsets(basePath, snapNum, id, 'Group' if 'Groups' in f else 'Subhalo')
                fileNum = offsetsThisType[1]
                fileOff = offsetsThisType[0]
                
                # Get the length of particles of this type
                if 'GroupLen' in f and str(id) in f['GroupLen']:
                    lentype = np.array(f['GroupLen'][str(id)])
                    numToRead = lentype[partType] if partType < len(lentype) else 0
                else:
                    numToRead = nPart[partType]
        except:
            # Fall back to simple case
            fileNum = 0
            fileOff = 0
            numToRead = nPart[partType]
    else:
        fileNum = subset['offsetType'][1]
        fileOff = subset['offsetType'][0]
        numToRead = subset['lenType'][partType]

    result['count'] = numToRead

    if not numToRead:
        return result

    # find a chunk with this particle type
    with h5py.File(snapPath(basePath, snapNum, 0), 'r') as f:
        i = 0
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(partType)+"] does not have field ["+field+"]")

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

        f.close()

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead
    
    # Get total number of files
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        numFiles = int(header['NumFiles'])

    while numToRead:
        try:
            f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')
        except FileNotFoundError:
            # Reached the end of available files; remaining particles may not exist
            if numToRead > 0:
                print(f"Warning: Could not read all {origNumToRead} particles. Only read {wOffset}.")
                break
            break

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][partType]

        # Safety check: ensure fileOff is valid
        if fileOff < 0 or fileOff >= numTypeLocal:
            print(f"Warning: Invalid file offset {fileOff} for file {fileNum} with {numTypeLocal} particles of type {partType}. Skipping.")
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
    
    offset_counter = 0
    snap_counters = np.zeros(6, dtype=np.int64)
    
    for fileNum in range(numFiles):
        with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
            header_this = dict(f['Header'].attrs.items())
            
            # Count groups/subhalos in this file
            if type == "Group":
                count_this = int(header_this.get('Ngroups_ThisFile', 0))
            else:  # Subhalo
                count_this = int(header_this.get('Nsubgroups_ThisFile', 0))
            
            fileOffsets[fileNum] = offset_counter
            snapOffsets[:, fileNum] = snap_counters.copy()
            
            # Update counters
            offset_counter += count_this
            
            # Update particle type counters from NumPart_ThisFile
            if 'NumPart_ThisFile' in header_this:
                snap_counters += np.array(header_this['NumPart_ThisFile'][:6], dtype=np.int64)
    
    groupFileOffsets = fileOffsets.copy()
    r['snapOffsets'] = snapOffsets

    # Calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum_array = np.where(groupFileOffsets >= 0)[0]
    
    if len(fileNum_array) == 0:
        raise ValueError(f"ID {id} not found in any file offset")
    
    fileNum = int(np.max(fileNum_array))
    fileNum = min(fileNum, numFiles - 1)  # Ensure fileNum doesn't exceed numFiles-1
    groupOffset = int(groupFileOffsets[fileNum])

    # Load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset, :]

    # Calculate the offset (by type) of this group/subgroup within the snapshot
    offsetType = np.zeros(6, dtype=np.int64)
    
    # Add offsets from all previous files
    for fn in range(fileNum):
        with h5py.File(gcPath(basePath, snapNum, fn), 'r') as f:
            header_fn = dict(f['Header'].attrs.items())
            if 'NumPart_ThisFile' in header_fn:
                offsetType += np.array(header_fn['NumPart_ThisFile'][:6], dtype=np.int64)
    
    # Add offset within current file for this group/subhalo
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        lentype_all = f[type][type+'LenType'][:]
        for prev_id in range(groupOffset):
            offsetType += lentype_all[prev_id, :]
    
    r['offsetType'] = offsetType

    return r


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
    """ Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
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
    """ Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadHalo_old(basePath, snapNum, id, partType, fields=None):
    """ OLD VERSION - Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets_old(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)
