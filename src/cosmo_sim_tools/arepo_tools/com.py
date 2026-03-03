import multiprocessing
from multiprocessing import Pool
import glob
from cosmo_sim_tools import illustris as il
import numpy as np
from cosmo_sim_tools.arepo_tools.global_props import get_particle_data


def getCOM(output_path, snapnum):
    """
    Calculates the center of mass (COM) and center of mass velocity (COM velocity) 
    for a given snapshot in a simulation.

    Parameters:
    - output_path (str): The path to the output directory.
    - snapnum (int): The snapshot number.

    Returns:
    - dict: A dictionary containing the COM coordinates (x, y, z), COM velocity (vx, vy, vz),
        and total mass of the particles.
    """
    particle_data = get_particle_data(output_path, snapnum, '012345', ['Coordinates', 'Velocities', 'Masses'])
    compos = (particle_data['Coordinates'].T * particle_data['Masses']).T
    totmom = (particle_data['Velocities'].T * particle_data['Masses']).T
    totmass = np.sum(particle_data['Masses'])
    compos = np.sum(compos, axis=0) / totmass
    com_vel = np.sum(totmom, axis=0) / totmass
    return {'x': compos[0], 'y': compos[1], 'z': compos[2], 'vx': com_vel[0], 'vy': com_vel[1], 'vz': com_vel[2], 'mass': totmass}

def COM_vs_time(path, Ncores=20):
    """
    Calculates the center of mass (COM) coordinates (x, y, z) as a function of time for a given simulation.

    Parameters:
    - path (str): The path to the simulation directory.
    - Ncores (int, optional): The number of CPU cores to use for parallel processing. Default is 20.

    Returns:
    - dict: A dictionary containing the time, x, y, and z coordinates of the COM at each snapshot.
    """
    N = len(glob.glob1(path, "snapshot*.hdf5"))

    def getCOM_temp(snap):
        tmp = getCOM(path, snap)
        time = il.snapshot.loadHeader(path, snap).get['Time']
        return time, tmp['x'], tmp['y'], tmp['z']

    p = Pool(Ncores)
    result = np.array(p.map(getCOM_temp, range(N)))
    return {'time': result[:, 0], 'x': result[:, 1], 'y': result[:, 2], 'z': result[:, 3]}



def mediancenter(output_path, snapnum, ptypes='012345'):
    """
    Calculates the median center of particle positions for a given snapshot.

    Args:
        output_path (str): The path to the output directory.
        snapnum (int): The snapshot number.
        ptypes (str, optional): The particle types to consider. Defaults to '012345'.

    Returns:
        array-like: The x, y, and z coordinates of the median center.
    """
    pos = get_particle_data(output_path, snapnum, ptypes, ['Coordinates'])
    x = np.median(pos['Coordinates'][:, 0])
    y = np.median(pos['Coordinates'][:, 1])
    z = np.median(pos['Coordinates'][:, 2])
    return np.array([x, y, z])

def mediancenter_vs_time(output_path, ptypes='012345', Ncpus=20):
    """
    Calculates the median center of particle positions over time.

    Args:
        output_path (str): The path to the output directory.
        ptypes (str, optional): The particle types to consider. Defaults to '012345'.
        Ncpus (int, optional): The number of CPUs to use for parallel processing. Defaults to 20.

    Returns:
        dict: A dictionary containing the time, x, y, and z coordinates of the median center.
    """
    global plotmediansnap
    N = len(glob.glob1(output_path, "snapshot*.hdf5"))

    def plotmediansnap(snapnum):
        """
        Helper function to calculate the median center for a given snapshot.

        Args:
            snapnum (int): The snapshot number.

        Returns:
            tuple: A tuple containing the time, x, y, and z coordinates of the median center.
        """
        time = il.snapshot.loadHeader(output_path, snapnum).get('Time')
        pos = get_particle_data(output_path, snapnum, ptypes, ['Coordinates'])
        x = np.median(pos['Coordinates'][:, 0])
        y = np.median(pos['Coordinates'][:, 1])
        z = np.median(pos['Coordinates'][:, 2])
        return time, x, y, z

    p = Pool(Ncpus)
    result = p.map(plotmediansnap, range(N))
    p.close()
    p.join()
    result = np.array(result)
    return {'time': result[:, 0], 'x': result[:, 1], 'y': result[:, 2], 'z': result[:, 3]}


def rec_com(guess, R, pos, masses, iterations=0):
    """
    Recursive function to calculate the center of mass.

    Args:
        guess (array-like): Initial guess for the center of mass.
        R (float): Radius within which to consider particles.
        pos (array-like): Array of particle positions.
        masses (array-like): Array of particle masses.
        iterations (int, optional): Number of iterations. Defaults to 0.

    Returns:
        array-like: The center of mass coordinates.
    """
    iterations += 1
    pos1 = pos - guess
    r = np.linalg.norm(pos1, axis=1)
    mask = r < R
    pos1 = pos[mask]
    mtmp = masses[mask]
    compos = np.sum((pos1.T * mtmp).T, axis=0) / np.sum(mtmp)

    if iterations == 20 or R < 0.1:
        print(f'Iterations = {iterations}, delta_r = {np.linalg.norm(compos-guess)} R={R}')
        return compos
    elif iterations > 20:
        print('didnt converge after 10 iterations')
        return 0
    else:
        r = np.linalg.norm(pos - compos, axis=1)
        mask = r < R
        r = r[mask]
        mtmp = masses[mask]
        xids = np.argsort(r)
        r = r[xids]
        mtmp = mtmp[xids]
        mtmp = np.cumsum(mtmp)
        for i in range(len(mtmp)):
            if mtmp[i] > mtmp[-1] / 2:
                halfr = r[i]
                break
        return rec_com(compos, halfr, pos, masses, iterations)


def get_iter_com(output_path, snapnum, guess, R):
    """
    Calculates the iterative center of mass for a given snapshot.

    Args:
        output_path (str): The path to the output directory.
        snapnum (int): The snapshot number.
        guess (str or array-like): Initial guess for the center of mass. If 'bhX' (e.g., 'bh0'), the center of mass
                                  of the black hole with ID X will be used.
        R (float): Radius within which to consider particles.

    Returns:
        array-like: The center of mass coordinates.
    """
    pdata = get_particle_data(output_path, snapnum, '012345', ['Coordinates', 'Masses'])
    if isinstance(guess, str) and (guess[:2] == 'bh'):
        bhids = il.snapshot.loadSubset(output_path, 0, 5, 'ParticleIDs')
        bh = il.snapshot.loadSubset(output_path, snapnum, 5, ['Coordinates', 'ParticleIDs'])
        if bh['count'] == 1:
            guess_com = bh['Coordinates'][0]
        else:
            bhnum = int(guess[2:])
            guess_com = bh['Coordinates'][bh['ParticleIDs'] == bhids[bhnum]]
        return rec_com(guess_com, R, pdata['Coordinates'], pdata['Masses'])
    else:
        return rec_com(guess, R, pdata['Coordinates'], pdata['Masses'])


def tmp_get_iter_com(output_path, snapnum, guess, R):
    """
    Temporary function to calculate the iterative center of mass for a given snapshot.

    Args:
        output_path (str): The path to the output directory.
        snapnum (int): The snapshot number.
        guess (str or array-like): Initial guess for the center of mass. If 'bhX' (e.g., 'bh0'), the center of mass
                                  of the black hole with ID X will be used.
        R (float): Radius within which to consider particles.

    Returns:
        tuple: A tuple containing the time, x, y, and z coordinates of the center of mass.
    """
    time = il.snapshot.loadHeader(output_path, snapnum).get('Time')
    cm = get_iter_com(output_path, snapnum, guess, R)
    return time, cm[0], cm[1], cm[2]


def get_iter_com_vs_time(output_path, R, bhnum):
    """
    Calculates the iterative center of mass over time for a given black hole.

    Args:
        output_path (str): The path to the output directory.
        R (float): Radius within which to consider particles.
        bhnum (int): The black hole number.

    Returns:
        dict: A dictionary containing the time, x, y, and z coordinates of the center of mass.
    """
    N = len(glob.glob1(output_path, "snapshot*.hdf5"))
    args = [(output_path, snap, f'bh{bhnum}', R) for snap in range(N)]
    p = Pool(20)
    result = np.array(p.starmap(tmp_get_iter_com, args))
    return {'time': result[:, 0], 'x': result[:, 1], 'y': result[:, 2], 'z': result[:, 3]}

