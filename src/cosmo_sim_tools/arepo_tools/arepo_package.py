
import sys
#sys.path.append('/home/aklantbhowmick/anaconda3/lib/python3.7/site-packages')

#sys.path.append('/home/aklantbhowmick/anaconda3/lib/python3.7/site-packages/scalpy/')
#sys.path.append('/home/aklantbhowmick/anaconda3/envs/nbodykit-env/lib/python3.6/site-packages/')

#%pylab inline
import matplotlib.cm as cm  
import numpy

from cosmo_sim_tools import brahma
import os
#from kdcount import correlate
import scipy
import matplotlib as mpl
import h5py



def get_redshift_from_snapshot(output_path,snap):
    """Find the redshift of the available snapshot nearest to snap.

    Return one redshift using the catalog snapshot correspondence.
    """
    snapshot_space,redshift_space=get_snapshot_redshift_correspondence(output_path)
    diff = abs(snapshot_space-snap)
    return (redshift_space[diff==min(diff)])[0]
    
    

def get_snapshot_redshift_correspondence(output_path,file_format='fof_subfind'):
    """Read snapshot numbers and redshifts from group catalogs.

    Return arrays ordered by snapshot number; file_format selects the catalog layout.
    """
    output_file_names=os.listdir(output_path)
#    print(output_foutput_test_0.32_1.83_new_no_ex_new9_bFOF3_env_seed5.00le_names)
#    print(output_path)
    snapshot_space=[]
    redshift_space=[]
    for name in output_file_names:
        if ('groups' in name):
            try:
                snapshot_number=int(name[7:])
                snapshot_space.append(snapshot_number)
            except:
                aaa=1
                #print("Warning: Ignoring filename:%s"%name)
    snapshot_space=numpy.sort(numpy.array(snapshot_space))
    for snapshot_number in snapshot_space:
            if (file_format=='fof_subfind') | (file_format=='fof_sub_subfind'):
                header=brahma.groupcat.loadHeader(output_path,snapshot_number)
            elif (file_format=='fof'):
                if (snapshot_number >= 100):
                    groups_folder = output_path+'/groups_%d/'%snapshot_number
                    file = 'fof_tab_%d.0.hdf5'%snapshot_number
                elif (snapshot_number >= 10):
                    groups_folder = output_path+'/groups_0%d/'%snapshot_number
                    file = 'fof_tab_0%d.0.hdf5'%snapshot_number
                else:
                    groups_folder = output_path+'/groups_00%d/'%snapshot_number
                    file = 'fof_tab_00%d.0.hdf5'%snapshot_number
                h=h5py.File(groups_folder+file)
                header=dict(h['Header'].attrs.items())
                h.close()
            elif (file_format=='fof_sub'):
                if (snapshot_number >= 100):
                    groups_folder = output_path+'/groups_sub_%d/'%snapshot_number
                    file = 'fof_sub_tab_%d.0.hdf5'%snapshot_number
                elif (snapshot_number >= 10):
                    groups_folder = output_path+'/groups_sub_0%d/'%snapshot_number
                    file = 'fof_sub_tab_0%d.0.hdf5'%snapshot_number
                else:
                    groups_folder = output_path+'/groups_sub_00%d/'%snapshot_number
                    file = 'fof_sub_tab_00%d.0.hdf5'%snapshot_number
                h=h5py.File(groups_folder+file)
                header=dict(h['Header'].attrs.items())
                h.close()
            redshift=header.get('Redshift')   
            redshift_space.append(redshift)   
    return numpy.array(snapshot_space),numpy.array(redshift_space)

def periodic_distance(basePath,position1,position2):
    """Measure the minimum-image distance between two positions.

    Read the periodic box size from basePath; use the same units as the coordinates.
    """
    boxsize=get_box_size(basePath)
    diff1=numpy.abs(position1-position2)
    diff2=numpy.abs(position1-position2+boxsize)
    diff3=numpy.abs(position1-position2-boxsize)
    distance_vec=numpy.amin(numpy.array([diff1,diff2,diff3]),axis=0)
    return numpy.sqrt(numpy.sum(distance_vec*distance_vec))
def get_bootstrap_error(sample,N_bootstrap,MODE,GET_CENTRAL=0):
    """Estimate the scatter of a sample mean or median by resampling.

    MODE selects the statistic; GET_CENTRAL also returns its average over resamples.
    """
    data=[]
    for i in range(0,N_bootstrap):        
        resample=numpy.random.choice(sample,size=sample.shape, replace=True)
        if (MODE=='mean'):
            data.append(numpy.average(resample))
        if (MODE=='median'):
            data.append(numpy.median(resample))
    if (GET_CENTRAL):
        return numpy.average(data),numpy.std(data)
    else:
        return numpy.std(data)


def make_median_with_bootstrap(x_values,y_values,x_min,x_max,nbins,N_bootstrap):
    """Estimate medians and bootstrap errors in evenly spaced x windows.

    Return centers, spacing, bootstrap-averaged medians, errors, and interquartile ranges.
    """
    x_space=numpy.linspace(x_min,x_max,nbins)
    diff=numpy.diff(x_space)[0]
    median_space=[]
    err_space=[]
    IQR_space=[]
    for x in x_space:
        mask=(x_values>(x-diff/2)) & (x_values<(x+diff/2))
        median,err=get_bootstrap_error(y_values[mask],N_bootstrap,'median',GET_CENTRAL=1)
        IQR_space.append(get_IQR(y_values[mask],75))
        median_space.append(median)
        err_space.append(err)
    median_space=numpy.array(median_space)
    err_space=numpy.array(err_space)
    IQR_space=numpy.array(IQR_space)
    return x_space,diff,median_space,err_space,IQR_space

def make_mean_with_bootstrap(x_values,y_values,x_min,x_max,nbins,N_bootstrap):
    """Estimate means and bootstrap errors in evenly spaced x windows.

    Return centers, spacing, bootstrap-averaged means, errors, and interquartile ranges.
    """
    x_space=numpy.linspace(x_min,x_max,nbins)
    diff=numpy.diff(x_space)[0]
    median_space=[]
    err_space=[]
    IQR_space=[]
    for x in x_space:
        mask=(x_values>(x-diff/2)) & (x_values<(x+diff/2))
        median,err=get_bootstrap_error(y_values[mask],N_bootstrap,'mean',GET_CENTRAL=1)
        IQR_space.append(get_IQR(y_values[mask],75))
        median_space.append(median)
        err_space.append(err)
    median_space=numpy.array(median_space)
    err_space=numpy.array(err_space)
    IQR_space=numpy.array(IQR_space)
    return x_space,diff,median_space,err_space,IQR_space



def get_box_size(output_path):
    """Read BoxSize from the first readable group catalog.

    Return the box length in simulation units, or None if no catalog can be read.
    """
    output_file_names=os.listdir(output_path)
    snapshot_space=[]
    redshift_space=[]
    for name in output_file_names:
        try:
            if ('groups' in name):
                snapshot_number=int(name[7:])
                header=brahma.groupcat.loadHeader(output_path,snapshot_number)
                box_size=header.get('BoxSize')   
                return box_size
        except:
            continue
             
        
def get_cosmology(output_path):
    """Read cosmological parameters from a group catalog header.

    Return Omega0, OmegaLambda, and HubbleParam from the first matching catalog.
    """
    output_file_names=os.listdir(output_path)
    snapshot_space=[]
    redshift_space=[]
    for name in output_file_names:
        if ('groups' in name) & ('groups_sub' not in name):
            snapshot_number=int(name[7:])
            header=brahma.groupcat.loadHeader(output_path,snapshot_number)
            om0=header.get('Omega0')
            oml=header.get('OmegaLambda')
            h=header.get('HubbleParam')            
            return om0,oml,h
        
        
def load_snapshot_header(output_path,desired_redshift):
    """Read and print the snapshot header nearest to desired_redshift.

    Return its HDF5 attributes as a dictionary using the Brahma snapshot path.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift)
    with h5py.File(brahma.snapshot.snapPath(output_path, output_snapshot)) as f:
        header = dict(f['Header'].attrs.items())
        print(header)
        return header

def desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=True,file_format='fof_subfind'): 
    """Select the available snapshot closest to a requested redshift.

    Return the actual redshift and snapshot number; list_all prints the selection.
    """
    snapshot_space,redshift_space=get_snapshot_redshift_correspondence(output_path,file_format=file_format)
    redshift_difference=numpy.abs(redshift_space-desired_redshift)
    min_redshift_difference=numpy.amin(redshift_difference)
    output_snapshot=snapshot_space[redshift_difference==min_redshift_difference][0]
    output_redshift=redshift_space[redshift_difference==min_redshift_difference][0]
    if (list_all):
        print("Desired redshift: ",desired_redshift)
        print("Output redshift: ",output_redshift)
        print("Output snapshot: ",output_snapshot)            
    return output_redshift,output_snapshot      
        
def make_cuts(quantities,cut): #selects an array of quantities (argument 1) and makes cuts (argument 2)
    """Apply the same index or boolean selection to multiple arrays.

    Return a list of selected arrays in the input order.
    """
    cutted_quantities=[quantity[cut] for quantity in quantities]
    return cutted_quantities

def get_group_property(output_path,group_property,desired_redshift,list_all=True,file_format='fof_subfind',stack_style='hstack',postprocessed = 0):
    """Load FoF halo fields at the nearest available redshift.

    file_format and postprocessed select the Brahma catalog reader; return fields and redshift.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False,file_format=file_format)
    if (file_format=='fof_subfind'):
        if(postprocessed == 1):
            property = brahma.groupcat.loadHalos_postprocessed(output_path,output_snapshot,fields=group_property)
        else:
            property = brahma.groupcat.loadHalos(output_path,output_snapshot,fields=group_property)
            
    elif (file_format=='fof_sub_subfind'):
        property = brahma.groupcat.loadHalos2(output_path,output_snapshot,fields=group_property)
    elif (file_format=='fof'):
        property = brahma.groupcat.loadHalos3(output_path,output_snapshot,fields=group_property)
    elif (file_format=='fof_sub'):
        property = brahma.groupcat.loadHalos4(output_path,output_snapshot,fields=group_property)
    else:
        print("Error: Unrecognized file format")
    return property,output_redshift
'''
    elif ('fof' in file_format):
        if(file_format=='fof'):
            if (output_snapshot>=100):
                groups_folder = output_path+'/groups_%d/'%output_snapshot
       	    elif(output_snapshot>=10):
       	        groups_folder = output_path+'/groups_0%d/'%output_snapshot
            else:
       	        groups_folder = output_path+'/groups_00%d/'%output_snapshot
        if(file_format=='fof_sub'):
            if (output_snapshot>=100):
                groups_folder = output_path+'/groups_sub_%d/'%output_snapshot
            elif (output_snapshot>=10):
                groups_folder = output_path+'/groups_sub_0%d/'%output_snapshot
       	    else:
       	        groups_folder = output_path+'/groups_sub_00%d/'%output_snapshot
        all_files=os.listdir(groups_folder)
        property=[]
        Nfiles=len(all_files)
        for Nf in range(0,Nfiles):
            if(file_format=='fof'):
                if (output_snapshot>=100):
                    file = 'fof_tab_%d.%d.hdf5'%(output_snapshot,Nf)
                elif (output_snapshot>=10):
                    file = 'fof_tab_0%d.0%d.hdf5'%(output_snapshot,Nf)
                else:
                    file = 'fof_tab_00%d.%d.hdf5'%(output_snapshot,Nf)
                                        
            if(file_format=='fof_sub'):
                if (output_snapshot>=100):
                    file = 'fof_sub_tab_%d.%d.hdf5'%(output_snapshot,Nf)
                elif (output_snapshot>=10):
                    file = 'fof_sub_tab_0%d.%d.hdf5'%(output_snapshot,Nf)
                else:
                    file = 'fof_sub_tab_00%d.%d.hdf5'%(output_snapshot,Nf)
            h=h5py.File(groups_folder+file)
            Group=h.get('Group')
            try:
                property.append(Group.get(group_property)[:])
            except:
                aaa=0
        property=numpy.array(property)
   
        if (stack_style == 'hstack'):
            property=numpy.hstack(property)
        elif (stack_style == 'vstack'):
                property=numpy.vstack(property)
            else:
                print ("Error: Invalid stack style , use 'vstack' or 'hstack'")    
    else:
        print("Error: Unrecognized file format")
    return property,output_redshift
'''


def get_subhalo_property(output_path,subhalo_property,desired_redshift,list_all=True,file_format='fof_subfind',postprocessed=0):
    """Load subhalo fields at the nearest available redshift.

    Select standard, alternate, or postprocessed catalogs; return fields and redshift.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False)  
    if (file_format=='fof_subfind'):
        if(postprocessed==1):
            property = brahma.groupcat.loadSubhalos_postprocessed(output_path,output_snapshot,fields=subhalo_property)
        else:
            property = brahma.groupcat.loadSubhalos(output_path,output_snapshot,fields=subhalo_property)
    elif(file_format=='fof_sub_subfind'):
        property = brahma.groupcat.loadSubhalos2(output_path,output_snapshot,fields=subhalo_property)
        
        #    if (list_all):
#        print('Below are the list of properties')
#        print(subhalos.keys())
    return property,output_redshift

def get_particle_property(output_path,particle_property,p_type,desired_redshift,list_all=True,file_format='fof_subfind'):
    """Load a snapshot field for the requested particle type.

    Return particle data and the actual redshift nearest to desired_redshift.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,file_format=file_format)
#    if (list_all):
#        print('Below are the list of properties for ptype ',p_type)
#        print(brahma.snapshot.loadSubset(output_path,output_snapshot,p_type).keys())
    return brahma.snapshot.loadSubset(output_path,output_snapshot,p_type,fields=particle_property),output_redshift


#def get_effective_zoom_volume(basePath,desired_redshift,levelmax):
#    mpc_to_kpc=1000.
#    Potential,output_redshift=get_particle_property(basePath,'Potential',1,desired_redshift)
#    number_of_DM_particles=len(Potential)
#    simulation_volume=(get_box_size(basePath)/mpc_to_kpc)**3
#    effective_volume=number_of_DM_particles*simulation_volume/(2.**(levelmax))**3
    #print(number_of_DM_particles,(2.**(levelmax))**3)
#    return effective_volume,simulation_volume

def get_effective_zoom_volume(basePath,desired_redshift,HighResGasFractionCut):
    """Estimate gas volumes from particle mass divided by density.

    Return high-resolution gas, total gas, box volumes, and redshift; lengths are converted by 1000.
    """
    mpc_to_kpc=1000.
#    DM_particle_mass=load_snapshot_header(basePath,desired_redshift)['MassTable'][1]
    ptype=0   
    Masses,output_redshift=get_particle_property(basePath,'Masses', ptype, desired_redshift)
    HighResGasMass,output_redshift=get_particle_property(basePath,'HighResGasMass', ptype, desired_redshift)
    Density,output_redshift=get_particle_property(basePath,'Density', ptype, desired_redshift)
    HighResGasFraction=HighResGasMass/Masses
#    print(HighResGasFractionCut)
#    print(HighResGasFraction[HighResGasFraction>HighResGasFractionCut])
    high_res_particle_volume=Masses[HighResGasFraction>HighResGasFractionCut] / Density[HighResGasFraction>HighResGasFractionCut]
#    print(high_res_particle_volume)
    all_particle_volume=Masses / Density
    simulation_volume=(get_box_size(basePath)/mpc_to_kpc)**3
    total_gas_volume=sum(all_particle_volume)/mpc_to_kpc**3
 
    high_res_gas_volume=sum(high_res_particle_volume)/mpc_to_kpc**3
    print("Note: The high resolution gas volume is:",high_res_gas_volume)

    #print(number_of_DM_particles,(2.**(levelmax))**3)
    return high_res_gas_volume,total_gas_volume,simulation_volume,output_redshift
    #return 1,1
    
def mass_counts(HM,Nbins,log_HM_min,log_HM_max,linear=0):
    """Bin masses and estimate counts per unit mass with Poisson errors.

    Return arithmetic bin centers, differential counts, and errors; linear selects the edges.
    """
    def extract(HM_min,HM_max):
        """Summarize values strictly inside the supplied bin edges.

        Return the bin center and the count, weight, or fraction used by the enclosing estimator.
        """
        mask=(HM>HM_min)&(HM<HM_max)
        return (HM_min+HM_max)/2,len(HM[mask])
    if (linear):
        HM_bin=numpy.linspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    else:
        HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
    #return out
    centers=numpy.array(list(zip(*out))[0])
    counts=numpy.array(list(zip(*out))[1])
    dM=centers*numpy.diff(numpy.log(centers))[0]
    HMF=counts/dM
    dHMF=numpy.sqrt(counts)/dM
    return centers,HMF,dHMF


def get_mass_function_for_zoom(basePath,levelmax,desired_redshift,particle_property,logmassmin,logmassmax,nbins,HighResGasFractionCut=0.1,zoom_volume=1):
    
    """Compute a black-hole mass function normalized by the selected volume.

    Scale snapshot masses by 1e10; return mass centers, densities, and Poisson errors.
    """
    if zoom_volume:
        effective_volume,total_gas_volume,simulation_volume,output_redshift=get_effective_zoom_volume(basePath,desired_redshift,HighResGasFractionCut)
    else:
        effective_volume=(get_box_size(basePath)/1000)**3

    print("Efective volume in levelmax %d:"%levelmax,effective_volume)


    p_type=5
    
    
    BHMass,output_redshift=get_particle_property(basePath,particle_property,p_type,desired_redshift)
    #if (particle_property=='BH_Mass'):
    BHMass*=1e10
    
    M,MF,dMF=mass_counts(BHMass,nbins,logmassmin,logmassmax)
    return M,MF/effective_volume,dMF/effective_volume


def luminosity_counts(HM,Nbins,log_HM_min,log_HM_max):
        """Count luminosities in logarithmic bins and normalize per dex.

        Return arithmetic bin centers, counts per dex, and Poisson errors per dex.
        """
        def extract(HM_min,HM_max):
            """Summarize values strictly inside the supplied bin edges.

            Return the bin center and the count, weight, or fraction used by the enclosing estimator.
            """
            mask=(HM>HM_min)&(HM<HM_max)
            #print len(HM[mask])
            return (HM_min+HM_max)/2,len(HM[mask])

        HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
        out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
        centers=numpy.array(list(zip(*out))[0])
        counts=numpy.array(list(zip(*out))[1])
        #print counts
        dlogM=numpy.diff(numpy.log10(centers))[0]
        HMF=counts/dlogM
        dHMF=numpy.sqrt(counts)/dlogM
        return centers,HMF,dHMF

def get_luminosity_function_for_zoom(basePath,levelmax,desired_redshift,Luminosities,logmassmin,logmassmax,nbins,HighResGasFractionCut=0.1,zoom_volume=1):
    """Normalize a supplied luminosity distribution by zoom or box volume.

    Return luminosity centers, number densities per dex, and their Poisson errors.
    """
    if (zoom_volume):
        effective_volume,total_gas_volume,simulation_volume,output_redshift=get_effective_zoom_volume(basePath,desired_redshift,HighResGasFractionCut)
    else:
        effective_volume=(get_box_size(basePath)/1000)**3
    print("Efective volume in levelmax %d:"%levelmax,effective_volume)
    M,MF,dMF=luminosity_counts(Luminosities,nbins,logmassmin,logmassmax)
    return M,MF/effective_volume,dMF/effective_volume



def mass_function(HM,box_size,Nbins,log_HM_min,log_HM_max):
    """Estimate a differential mass function in a periodic box.

    Convert box_size by 1000; return centers, counts per mass per volume, and Poisson errors.
    """
    box_size_Mpc=box_size/1000.
    #print(HM)
    
    def extract(HM_min,HM_max):
        """Summarize values strictly inside the supplied bin edges.

        Return the bin center and the count, weight, or fraction used by the enclosing estimator.
        """
        mask=(HM>HM_min)&(HM<HM_max)
        return (HM_min+HM_max)/2,len(HM[mask])

    HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
    #return out
    centers=numpy.array(list(zip(*out))[0])
    counts=numpy.array(list(zip(*out))[1])
    dM=centers*numpy.diff(numpy.log(centers))[0]
    HMF=counts/dM/box_size_Mpc**3
    dHMF=numpy.sqrt(counts)/dM/box_size_Mpc**3
    return centers,HMF,dHMF

def get_distribution(quantity,Nbins,minimum,maximum,boxsize,min_count):  
    #sdsd
    
    """Histogram positions and estimate low-count boundaries around the mode.

    May unwrap quantity in place; return centers, counts, and the left and right boundaries.
    """
    mask=(quantity<1000)
    
    mask2=(quantity>(boxsize-1000))
    if(len(quantity[mask])>5)&(len(quantity[mask2])>5):
    #if ((min(quantity)<1000)&(max(quantity)>boxsize-1000)):
        print('Warning:Reflection performed')
        quantity[quantity<(boxsize/2)]+=boxsize   
    else:
        print('No Reflection performed')
    def extract(mn,mx):
        """Summarize values strictly inside the supplied bin edges.

        Return the bin center and the count, weight, or fraction used by the enclosing estimator.
        """
        mask=(quantity>mn)&(quantity<mx)
        return (mn+mx)/2,len(quantity[mask])
    bin_edges=numpy.linspace(minimum,maximum,Nbins,endpoint=True)
    out=[extract(bin_edges[i],bin_edges[i+1]) for i in range(0,len(bin_edges)-1)]
    #return out
    centers=numpy.array(list(zip(*out))[0])
    counts=numpy.array(list(zip(*out))[1])
    
    #print("centers:",centers)
    #print("counts:",counts)
    
    mode=numpy.average(centers[counts==max(counts)])
    print(mode)
    left_edge=numpy.amax(centers[(counts<min_count)&(centers<mode)])
    right_edge=numpy.amin(centers[(counts<min_count)&(centers>mode)])
    
    return centers,counts,left_edge,right_edge


def get_probability_density(HM,Nbins,log_HM_min,log_HM_max,linear=0):
    #print(HM)
    
    """Normalize bin counts over the supplied mass or quantity range.

    Return centers, probabilities, errors, normalization width, and total count; linear uses densities.
    """
    def extract(HM_min,HM_max):
        """Summarize values strictly inside the supplied bin edges.

        Return the bin center and the count, weight, or fraction used by the enclosing estimator.
        """
        mask=(HM>HM_min)&(HM<HM_max)
        return (HM_min+HM_max)/2,len(HM[mask])
    if (linear):
        HM_bin=numpy.linspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    else:
        HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
    #return out
    centers=numpy.array(list(zip(*out))[0])
    counts=numpy.array(list(zip(*out))[1])
    counts_sum=numpy.sum(counts)
    
    HMF=counts
    dHMF=numpy.sqrt(counts)
    if (linear):
        norm=numpy.diff(centers)[0]
    else:
        norm=1
    return centers,HMF/norm/counts_sum,dHMF/norm/counts_sum,norm,counts_sum

def get_probability_density_weighed(HM,weigh,Nbins,log_HM_min,log_HM_max,linear=0):
    #print(HM)
    
    """Normalize weighted bin counts over the requested range.

    Return centers, normalized weights, square-root-weight errors, width, and total weight.
    """
    def extract(HM_min,HM_max):
        """Summarize values strictly inside the supplied bin edges.

        Return the bin center and the count, weight, or fraction used by the enclosing estimator.
        """
        mask=(HM>HM_min)&(HM<HM_max)
        return (HM_min+HM_max)/2,sum(weigh[mask])
    if (linear):
        HM_bin=numpy.linspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    else:
        HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
    #return out
    centers=numpy.array(list(zip(*out))[0])
    counts=numpy.array(list(zip(*out))[1])
    counts_sum=numpy.sum(counts)
    
    HMF=counts
    dHMF=numpy.sqrt(counts)
    if (linear):
        norm=numpy.diff(centers)[0]
    else:
        norm=1
    return centers,HMF/norm/counts_sum,dHMF/norm/counts_sum,norm,counts_sum




def BH_mass_function_AGN_fraction(bhmass,bolometric_luminosity,box_size,Nbins,log_HM_min,log_HM_max,log_lbol_cut):
    """Estimate a black-hole mass function and luminosity-selected AGN fraction.

    Return centers, mass-function values and errors, followed by AGN fractions and errors.
    """
    box_size_Mpc=box_size/1000.
    #print(HM)
    HM=bhmass
    def extract(HM_min,HM_max):
        """Summarize values strictly inside the supplied bin edges.

        Return the bin center and the count, weight, or fraction used by the enclosing estimator.
        """
        mask=(HM>HM_min)&(HM<HM_max)
        mask2=mask&(bolometric_luminosity>10**log_lbol_cut)
        return (HM_min+HM_max)/2,len(HM[mask]),float(len(HM[mask2]))/(len(HM[mask])+0.00000001),numpy.sqrt(float(len(HM[mask2])))/(len(HM[mask])+0.00000001)

    HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
    out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
    #return out
    centers=numpy.array(list(zip(*out))[0])
    counts=numpy.array(list(zip(*out))[1])
    AGN_fraction=numpy.array(list(zip(*out))[2])
    d_AGN_fraction=numpy.array(list(zip(*out))[3])
    dM=centers*numpy.diff(numpy.log(centers))[0]
    HMF=counts/dM/box_size_Mpc**3
    dHMF=numpy.sqrt(counts)/dM/box_size_Mpc**3
    return centers,HMF,dHMF,AGN_fraction,d_AGN_fraction




def get_mass_function(category,object_type,desired_redshift,output_path,Nbins,log_mass_min,log_mass_max,list_all=True,dynamical_bh_mass=True):
    """Load group or subhalo masses and calculate their mass function.

    category selects total or component mass; return centers, densities, errors, and redshift.
    """
    box_size=get_box_size(output_path)
      #print (box_)
    if (object_type=='group'):
        if (category=='total'):
            mass,output_redshift=get_group_property(output_path,'GroupMass',desired_redshift,list_all=list_all)
            centers,HMF,dHMF=mass_function(mass*1e10,box_size,Nbins,log_mass_min,log_mass_max)
            return centers,HMF,dHMF,output_redshift
        
    if (object_type=='subhalo'):
        if (category=='total'):
            mass,output_redshift=get_subhalo_property(output_path,'SubhaloMass',desired_redshift,list_all=list_all)
            centers,HMF,dHMF=mass_function(mass*1e10,box_size,Nbins,log_mass_min,log_mass_max)
            return centers,HMF,dHMF,output_redshift
        else:
            if (category=='stellar'):
                p_type=4
            if (category=='bh'):
                p_type=5               
            if (category=='dark'):
                p_type=1          
            if (category=='gas'):
                p_type=0                
                
            if ((p_type==5)&(dynamical_bh_mass==True)):
                mass,output_redshift=get_subhalo_property(output_path,'SubhaloBHMass',desired_redshift,list_all=list_all) 
            else:
                masstype,output_redshift=get_subhalo_property(output_path,'SubhaloMassType',desired_redshift,list_all=list_all)           
                mass=masstype[:,p_type]
            centers,HMF,dHMF=mass_function(mass*1e10,box_size,Nbins,log_mass_min,log_mass_max)
            return centers,HMF,dHMF,output_redshift        

def get_particle_history(z_latest,z_earliest,z_no_of_bins,p_type,p_id_to_be_tracked,desired_property,output_path):
    """Track one particle ID through snapshots near a redshift grid.

    Return property and redshift arrays, skipping failed reads or non-unique ID matches.
    """
    z_space=numpy.linspace(z_latest,z_earliest,z_no_of_bins)
    prperty_history=[]
    z_history=[]
    for z in z_space:
        try:
            p_id_current_z,output_redshift=get_particle_property(output_path,'ParticleIDs',p_type,z,list_all=False)
            extract_target_id=p_id_to_be_tracked==p_id_current_z
            prperty_space,output_redshift=get_particle_property(output_path,desired_property,p_type,z,list_all=False)
            temp=prperty_space[extract_target_id]
            if (len(temp)==1):
                prperty_history.append(temp[0])
                z_history.append(output_redshift)
        except:
            aa=1
    return numpy.array(prperty_history),numpy.array(z_history)

def poiss(rmin,rmax,BOXSIZE):
    """Compute a spherical-shell volume divided by the simulation volume.

    rmin and rmax are in box-length units divided by 1000; BOXSIZE is the original box length.
    """
    p=4./3*scipy.pi*(rmax**3-rmin**3)/(BOXSIZE/1e3)**3
    return p 

    
def correlate_info(data, NBINS, RMIN, RMAX, BOXSIZE, WRAP):
    """Count particle pairs in logarithmic radial bins using kdcount.

    Return radii, data-pair counts, and uniform random expectations; WRAP enables periodicity.
    """
    from kdcount import correlate
    if data is not None:
        if RMAX is None:
            RMAX = BOXSIZE
        
        if WRAP:
            wrap_length = BOXSIZE/1e3
        else:
            wrap_length = None
        
        dataset = correlate.points(data, boxsize = wrap_length)  
        
        binning = correlate.RBinning(numpy.logspace(numpy.log10(RMIN),numpy.log10(RMAX),NBINS+1))
        
#	RR=N**2*numpy.asarray([poiss(rbin[i],rbin[i+1]) for i in range(0,nbins)])
        DD = correlate.paircount(dataset, dataset, binning, np=16)
        DD = DD.sum1
        
#        print 'Done correlating'
        r = binning.centers
        rbin=binning.edges
        N=len(data)
        
        RR=(N**2-N)*numpy.asarray([poiss(rbin[i],rbin[i+1],BOXSIZE) for i in range(0,NBINS)])
    
    
        return r, DD,RR
    else:
        return None, None

    
    
def cross_correlate_info(data1,data2, NBINS, RMIN, RMAX, BOXSIZE, WRAP):
    """Count cross-pairs between two coordinate samples using kdcount.

    Return radii, cross-pair counts, and uniform random expectations for the supplied box.
    """
    from kdcount import correlate
    if data1 is not None:
        if RMAX is None:
            RMAX = BOXSIZE
        
        if WRAP:
            wrap_length = BOXSIZE/1e3
        else:
            wrap_length = None
        
        dataset1 = correlate.points(data1, boxsize = wrap_length) 
        dataset2 = correlate.points(data2, boxsize = wrap_length) 
        
        binning = correlate.RBinning(numpy.logspace(numpy.log10(RMIN),numpy.log10(RMAX),NBINS+1))
        
#	RR=N**2*numpy.asarray([poiss(rbin[i],rbin[i+1]) for i in range(0,nbins)])
        DD = correlate.paircount(dataset1, dataset2, binning, np=16)
        DD = DD.sum1
        
#        print 'Done correlating'
        r = binning.centers
        rbin=binning.edges
        N1=len(data1)
        N2=len(data2)
        
        RR=(N1*N2)*numpy.asarray([poiss(rbin[i],rbin[i+1],BOXSIZE) for i in range(0,NBINS)])
    
        return r, DD,RR
    else:
        return None, None
    
def get_dark_matter_correlation_function(output_path,input_redshift,NBINS, RMIN, RMAX, WRAP,subsample_factor):

    """Estimate the dark-matter autocorrelation from a subsampled snapshot.

    Return radii, DD, RR, correlation, Poisson errors, and the selected redshift.
    """
    BOXSIZE=get_box_size(output_path)
    print("boxsize:",BOXSIZE)
    positions,output_redshift=get_particle_property(output_path,'Coordinates',1,input_redshift)
    positions=positions/1e3
    r,DD,RR=correlate_info(positions[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)
    xi=DD/RR-1
    dxi=numpy.sqrt(DD)/RR
    return r,DD,RR,xi,dxi,output_redshift

def get_dark_matter_correlation_function_zoom(output_path,input_redshift,NBINS, RMIN, RMAX, WRAP,subsample_factor):
    """Combine high- and low-resolution dark-matter pair counts by mass.

    Return radii, weighted DD and RR, correlation, estimated errors, and redshift.
    """
    from kdcount import correlate

    BOXSIZE=get_box_size(output_path)
    #input_redshift=2.0
    positions0,output_redshift=get_particle_property(output_path,'Coordinates',1,input_redshift)
    positions0=positions0/1e3
    header=load_snapshot_header(output_path,input_redshift)
    mass0=header['MassTable'][1]

    low_res_masses,output_redshift=get_particle_property(output_path,'Masses',2,input_redshift)
    low_res_positions,output_redshift=get_particle_property(output_path,'Coordinates',2,input_redshift)

    unique_low_res_masses=numpy.unique(low_res_masses)
    mass1=unique_low_res_masses[0]
    mass2=unique_low_res_masses[1]

    positions1=low_res_positions[low_res_masses==mass1]
    positions2=low_res_positions[low_res_masses==mass2]
    positions1=positions1/1e3
    positions2=positions2/1e3


    #subsample_factor=500
    r,DD00,RR00=correlate_info(positions0[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)
    r,DD11,RR11=correlate_info(positions1[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)
    r,DD22,RR22=correlate_info(positions2[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)

    r,DD01,RR01=cross_correlate_info(positions0[::subsample_factor],positions1[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)
    r,DD02,RR02=cross_correlate_info(positions0[::subsample_factor],positions2[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)
    r,DD12,RR12=cross_correlate_info(positions1[::subsample_factor],positions2[::subsample_factor],NBINS, RMIN, RMAX, BOXSIZE, WRAP)
    #mass0=mass1=mass2=1.
    DD=mass0**2*DD00+mass1**2*DD11+mass2**2*DD22+mass0*mass1*DD01+mass0*mass2*DD02+mass1*mass2*DD12
    DD_error=mass0**2*numpy.sqrt(DD00)+mass1**2*numpy.sqrt(DD11)+mass2**2*numpy.sqrt(DD22)+mass0*mass1*numpy.sqrt(DD01)+mass0*mass2*numpy.sqrt(DD02)+mass1*mass2*numpy.sqrt(DD12)
    RR=mass0**2*RR00+mass1**2*RR11+mass2**2*RR22+mass0*mass1*RR01+mass0*mass2*RR02+mass1*mass2*RR12
    binning = correlate.RBinning(numpy.logspace(numpy.log10(RMIN),numpy.log10(RMAX),NBINS+1))
    rbin=binning.edges
    #RR=(N**2-N)*numpy.asarray([arepo_package.poiss(rbin[i],rbin[i+1],BOXSIZE) for i in range(0,NBINS)])
    
    xi=DD/RR-1
    dxi=DD_error/RR
    return r,DD,RR,xi,dxi,output_redshift



def get_group_lengths_offsets(output_path,p_type,desired_redshift,maximum_index,file_format='fof_subfind',postprocessed=0):
    """Compute cumulative particle offsets from FoF GroupLenType.

    Return lengths through maximum_index, their starting offsets, and the selected redshift.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False)
    group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift,file_format = file_format,list_all=False,postprocessed=postprocessed))
    group_lengths=group_lengths[:,p_type] 
    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,maximum_index+1)])
    return group_lengths[0:maximum_index+1],group_offsets,output_redshift


def get_particle_property_within_postprocessed_groups(output_path,particle_property,p_type,desired_redshift,subhalo_index,group_type='groups',list_all=True,store_all_offsets=1, public_simulation=0,file_format='fof_subfind'):
    """Select particle fields using postprocessed group catalogs and ordered snapshots.

    Return particles and redshift for groups; subhalo mode also returns parent-group particles.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False,file_format=file_format)
    if (public_simulation==0):
        requested_property=brahma.snapshot.loadSubset_groupordered(output_path,output_snapshot,p_type,fields=particle_property)

    if (group_type=='groups'):
        if(public_simulation==0):              
            group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift,list_all=False,file_format=file_format,stack_style='vstack',postprocessed=1))
            group_lengths=group_lengths[:,p_type] 
            if (store_all_offsets==0):
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,subhalo_index+1)]) 
            else:
                if (os.path.exists(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot))):
                    group_offsets=numpy.load(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),allow_pickle = True)
                    print("offsets were already there")
                else:
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
                    numpy.save(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),group_offsets)
                    print("Storing the offsets")
            group_particles=requested_property[group_offsets[subhalo_index]:group_offsets[subhalo_index]+group_lengths[subhalo_index]]
        else:
            group_particles=brahma.snapshot.loadHalo(output_path, output_snapshot, subhalo_index, p_type, fields=particle_property)
        return group_particles,output_redshift

    elif (group_type=='subhalo'):              
        if(public_simulation==0):
            group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift,postprocessed=1))
            group_lengths=group_lengths[:,p_type] 
            if (store_all_offsets==0):
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,subhalo_index+1)]) 
            else:
                if (os.path.exists(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot))):
                    group_offsets=numpy.load(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),allow_pickle = True)
                    print("offsets were already there")
                else:
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
                    numpy.save(output_path+'/offsets_%d_snap%d_postprocessed.npy'%(p_type,output_snapshot),group_offsets)
                    print("Storing the offsets")        
            subhalo_lengths,output_redshift=(get_subhalo_property(output_path,'SubhaloLenType', desired_redshift, postprocessed=1))
            subhalo_lengths=subhalo_lengths[:,p_type] 
            subhalo_indices=numpy.arange(0,len(subhalo_lengths))
            subhalo_group_number,output_redshift=(get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False,postprocessed=1));
            desired_group_number=subhalo_group_number[subhalo_index]  
            subhalo_lengths=subhalo_lengths[subhalo_group_number==desired_group_number]
            subhalo_offsets=numpy.array([sum(subhalo_lengths[0:i]) for i in range(0,len(subhalo_lengths))])
            mask=subhalo_group_number==desired_group_number
            #print(len(mask)),mask
            subhalo_indices=subhalo_indices[mask]  
            subhalo_final_indices=numpy.arange(0,len(subhalo_indices))
            group_particles=requested_property[group_offsets[desired_group_number]:group_offsets[desired_group_number]+group_lengths[desired_group_number]]   
        
            del requested_property

            #subhalo_indices=subhalo_indices[subhalo_group_number==desired_group_number]
            final_index=(subhalo_final_indices[subhalo_indices==subhalo_index])[0]
        
            subhalo_particles=group_particles[subhalo_offsets[final_index]:subhalo_offsets[final_index]+subhalo_lengths[final_index]]
      
            #return subhalo_particles,group_particles,output_redshift     
        else:
            subhalo_group_number,output_redshift=(get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False,postprocessed=1));
            desired_group_number=subhalo_group_number[subhalo_index]
            group_particles=brahma.snapshot.loadHalo(output_path, output_snapshot, desired_group_number, p_type, fields=particle_property)
            subhalo_particles=brahma.snapshot.loadSubhalo(output_path, output_snapshot, subhalo_index, p_type, fields=particle_property)
        return subhalo_particles,group_particles,output_redshift
    else:
        print("Error:Unidentified group type")


def get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift,subhalo_index,group_type='groups',list_all=True,store_all_offsets=1, public_simulation=0,file_format='fof_subfind'):
    """Select particle fields belonging to a FoF group or subhalo.

    Return particles and redshift; subhalo mode also returns parent-group particles. Offsets may be cached.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False,file_format=file_format)
    if (public_simulation==0):
        requested_property=brahma.snapshot.loadSubset(output_path,output_snapshot,p_type,fields=particle_property)

    if (group_type=='groups'):
        if(public_simulation==0):              
            group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift,list_all=False,file_format=file_format,stack_style='vstack'))
            group_lengths=group_lengths[:,p_type] 
            if (store_all_offsets==0):
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,subhalo_index+1)]) 
            else:
                if (os.path.exists(output_path+'/offsets_%d_snap%d.npy'%(p_type,output_snapshot))):
                    group_offsets=numpy.load(output_path+'/offsets_%d_snap%d.npy'%(p_type,output_snapshot),allow_pickle = True)
                    print("offsets were already there")
                else:
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
                    numpy.save(output_path+'/offsets_%d_snap%d.npy'%(p_type,output_snapshot),group_offsets)
                    print("Storing the offsets")
            group_particles=requested_property[group_offsets[subhalo_index]:group_offsets[subhalo_index]+group_lengths[subhalo_index]]
        else:
            group_particles=brahma.snapshot.loadHalo(output_path, output_snapshot, subhalo_index, p_type, fields=particle_property)
        return group_particles,output_redshift

    elif (group_type=='subhalo'):              
        if(public_simulation==0):
            group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift))
            group_lengths=group_lengths[:,p_type] 
            if (store_all_offsets==0):
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,subhalo_index+1)]) 
            else:
                if (os.path.exists(output_path+'/offsets_%d_snap%d.npy'%(p_type,output_snapshot))):
                    group_offsets=numpy.load(output_path+'/offsets_%d_snap%d.npy'%(p_type,output_snapshot),allow_pickle = True)
                    print("offsets were already there")
                else:
                    group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
                    numpy.save(output_path+'/offsets_%d_snap%d.npy'%(p_type,output_snapshot),group_offsets)
                    print("Storing the offsets")        
            subhalo_lengths,output_redshift=(get_subhalo_property(output_path,'SubhaloLenType', desired_redshift))
            subhalo_lengths=subhalo_lengths[:,p_type] 
            subhalo_indices=numpy.arange(0,len(subhalo_lengths))
        
        
            subhalo_group_number,output_redshift=(get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False));
            desired_group_number=subhalo_group_number[subhalo_index]  
            subhalo_lengths=subhalo_lengths[subhalo_group_number==desired_group_number]
            subhalo_offsets=numpy.array([sum(subhalo_lengths[0:i]) for i in range(0,len(subhalo_lengths))])
        
            mask=subhalo_group_number==desired_group_number
            #print(len(mask)),mask
            subhalo_indices=subhalo_indices[mask]  
            subhalo_final_indices=numpy.arange(0,len(subhalo_indices))
            group_particles=requested_property[group_offsets[desired_group_number]:group_offsets[desired_group_number]+group_lengths[desired_group_number]]   
        
            del requested_property

            #subhalo_indices=subhalo_indices[subhalo_group_number==desired_group_number]
            final_index=(subhalo_final_indices[subhalo_indices==subhalo_index])[0]
        
            subhalo_particles=group_particles[subhalo_offsets[final_index]:subhalo_offsets[final_index]+subhalo_lengths[final_index]]
      
            #return subhalo_particles,group_particles,output_redshift     
        else:
            subhalo_group_number,output_redshift=(get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False));
            desired_group_number=subhalo_group_number[subhalo_index]
            group_particles=brahma.snapshot.loadHalo(output_path, output_snapshot, desired_group_number, p_type, fields=particle_property)
            subhalo_particles=brahma.snapshot.loadSubhalo(output_path, output_snapshot, subhalo_index, p_type, fields=particle_property)
        return subhalo_particles,group_particles,output_redshift
    else:
        print("Error:Unidentified group type")
        
        
def reposition(original_position,scaled_halo_centers,boxsize):
    """Shift coordinates relative to a box-scaled halo center and wrap once.

    Return an N-by-3 coordinate array in the input length units.
    """
    x_pos=original_position[:,0]
    y_pos=original_position[:,1]
    z_pos=original_position[:,2]
    
    x_pos=x_pos-(boxsize/2)+scaled_halo_centers[0]*boxsize
    y_pos=y_pos-(boxsize/2)+scaled_halo_centers[1]*boxsize
    z_pos=z_pos-(boxsize/2)+scaled_halo_centers[2]*boxsize

    x_pos[x_pos<0]+=boxsize
    x_pos[x_pos>boxsize]-=boxsize
    z_pos[z_pos<0]+=boxsize
    z_pos[z_pos>boxsize]-=boxsize
    y_pos[y_pos<0]+=boxsize
    y_pos[y_pos>boxsize]-=boxsize
    return numpy.transpose(numpy.array([x_pos,y_pos,z_pos]))
        
        
def make_image(Coordinates,Coordinates_for_COM,plane,obj,boxsize,NBINS,scaled_halo_centers=1.,colormap='Blues_r',opacity=1,about_COM=True,REPOSITION=False,show_image=0):
    """Center or reposition coordinates and optionally draw a projected histogram.

    Return a 3-by-N coordinate array; plane selects xy, yz, or xz when show_image is enabled.
    """
    x_pos=Coordinates[:,0]
    y_pos=Coordinates[:,1]
    z_pos=Coordinates[:,2]

    x_pos_COM=Coordinates_for_COM[:,0]
    y_pos_COM=Coordinates_for_COM[:,1]
    z_pos_COM=Coordinates_for_COM[:,2]
    
    #if (CENTER_OF_MASS):   
    COM_x=numpy.median(x_pos_COM)
    COM_y=numpy.median(y_pos_COM)
    COM_z=numpy.median(z_pos_COM)

    def min_dis(median_position, position,box_size):
        """Choose the signed displacement with smallest absolute periodic distance.

        Compare the direct displacement with its two neighboring-box images.
        """
        pos_1=position-median_position
        pos_2=position-median_position+boxsize
        pos_3=position-median_position-boxsize

        new_position_options=numpy.array([pos_1,pos_2,pos_3])
        get_minimum_distance=numpy.argmin(numpy.abs(new_position_options))
        #print(new_position_options)

        #print(get_minimum_distance)
        return new_position_options[get_minimum_distance]

    vectorized_min_dis = numpy.vectorize(min_dis)
    if about_COM:
        x_pos_wrapped=vectorized_min_dis(COM_x,x_pos,boxsize)
        y_pos_wrapped=vectorized_min_dis(COM_y,y_pos,boxsize)
        z_pos_wrapped=vectorized_min_dis(COM_z,z_pos,boxsize)
    else:
        if (REPOSITION):
            x_pos=x_pos-(boxsize/2)+scaled_halo_centers[0]*boxsize
            y_pos=y_pos-(boxsize/2)+scaled_halo_centers[1]*boxsize
            z_pos=z_pos-(boxsize/2)+scaled_halo_centers[2]*boxsize

            x_pos[x_pos<0]+=boxsize
            x_pos[x_pos>boxsize]-=boxsize
            z_pos[z_pos<0]+=boxsize
            z_pos[z_pos>boxsize]-=boxsize
            y_pos[y_pos<0]+=boxsize
            y_pos[y_pos>boxsize]-=boxsize
        x_pos_wrapped=x_pos
        y_pos_wrapped=y_pos
        z_pos_wrapped=z_pos
        

    #plane='xz'

    if(show_image==1):
        if (plane=='xy'):
            obj.hist2d(x_pos_wrapped,y_pos_wrapped, bins=(NBINS,NBINS), norm=mpl.colors.LogNorm(),cmap=colormap,alpha=opacity);
        if (plane=='yz'):
            obj.hist2d(y_pos_wrapped,z_pos_wrapped, bins=(NBINS,NBINS), norm=mpl.colors.LogNorm(),cmap=colormap,alpha=opacity);
        if (plane=='xz'):
            obj.hist2d(x_pos_wrapped,z_pos_wrapped, bins=(NBINS,NBINS), norm=mpl.colors.LogNorm(),cmap=colormap,alpha=opacity);
        
    return numpy.array([x_pos_wrapped,y_pos_wrapped,z_pos_wrapped])

def sort_X_based_on_Y(X,Y):
    """Sort values in X by the corresponding values in Y.

    Return a NumPy array ordered by ascending Y, with X breaking ties.
    """
    return numpy.array([x for _,x in sorted(zip(Y,X))])

def get_seedremoved_events(output_path):

    """Combine text logs from blackhole_seedsremoved under output_path.

    Return removal and seeding scale factors, BH IDs, file IDs, and the number of malformed logs.
    """
    output_file_names=os.listdir(output_path+'blackhole_seedsremoved/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_seeding_complete=numpy.array([])
    scale_fac_complete=numpy.array([])
    BH_id_complete=numpy.array([],dtype=int)

    #BH_mass2_complete=numpy.array([])

    N_empty=0

    for name in output_file_names[:]:
        data=numpy.loadtxt(output_path+'blackhole_seedsremoved/'+name)

        try:
        #for ii in [1]:
            if (data.shape==(4,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                scale_fac_seeding=numpy.array([data[2]])
                BH_id=numpy.array([data[3].astype(int)])
            else:    
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                scale_fac_seeding=data[:,2]
                BH_id=data[:,3].astype(int)

            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
            scale_fac_seeding_complete=numpy.append(scale_fac_seeding_complete,scale_fac_seeding)
            BH_id_complete=numpy.append(BH_id_complete,BH_id)

        except IndexError:
            N_empty+=1
            aaa=1
            print('Index err:', name)
    
    return scale_fac_complete,scale_fac_seeding_complete,BH_id_complete,file_id_complete,N_empty


def get_seeding_events_debug(output_path):

    """Read the debug black-hole seeding log format.

    Return event and host-property arrays, malformed-log count, and gas/BH smoothing lengths.
    """
    output_file_names=os.listdir(output_path+'blackhole_seeding/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_complete=numpy.array([])
    BH_id_complete=numpy.array([],dtype=int)
    metallicity_complete=numpy.array([])
    SFR_complete=numpy.array([])
    density_complete=numpy.array([])
    FOFDMmass_complete=numpy.array([])
    indexmaxdens_complete=numpy.array([],dtype=int)
    FOFTask_complete=numpy.array([],dtype=int)
    Gas_Hsml_complete=numpy.array([])
    BH_Hsml_complete=numpy.array([])
    #BH_mass2_complete=numpy.array([])

    N_empty=0

    for name in output_file_names[:]:
        try:
            data=numpy.loadtxt(output_path+'blackhole_seeding/'+name)

         
        #for ii in [1]:
            if (data.shape==(11,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                BH_id=numpy.array([data[2].astype(int)])
                density=numpy.array([data[3]])
                metallicity=numpy.array([data[4]])
                SFR=numpy.array([data[5]])
                FOFDMmass=numpy.array([data[6]])
                indexmaxdens=numpy.array([data[7].astype(int)])
                FOFTask=numpy.array([data[8].astype(int)])
                Gas_Hsml=numpy.array([data[9]])
                BH_Hsml=numpy.array([data[10]])
            else:    
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                BH_id=data[:,2].astype(int)
                density=data[:,3]    
                metallicity=data[:,4]
                SFR=data[:,5]
                FOFDMmass=data[:,6]
                indexmaxdens=data[:,7].astype(int)
                FOFTask=data[:,8].astype(int)
                Gas_Hsml=data[:,9]
                BH_Hsml=data[:,10]                
            #print(

            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
            BH_id_complete=numpy.append(BH_id_complete,BH_id)
            #BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
            metallicity_complete=numpy.append(metallicity_complete,metallicity)
            SFR_complete=numpy.append(SFR_complete,SFR) 
            density_complete=numpy.append(density_complete,density)
            FOFDMmass_complete=numpy.append(FOFDMmass_complete,FOFDMmass)
            indexmaxdens_complete=numpy.append(indexmaxdens_complete,indexmaxdens)
            FOFTask_complete=numpy.append(FOFTask_complete,FOFTask)
            Gas_Hsml_complete=numpy.append(Gas_Hsml_complete,Gas_Hsml)
            BH_Hsml_complete=numpy.append(BH_Hsml_complete,BH_Hsml)
        except:
            N_empty+=1
            aaa=1
            print('Index err:', name)
    
    return scale_fac_complete,BH_id_complete,density_complete,metallicity_complete,SFR_complete,FOFDMmass_complete,indexmaxdens_complete,FOFTask_complete, file_id_complete,N_empty,Gas_Hsml_complete,BH_Hsml_complete


def get_seeding_events(output_path):

    """Read black-hole seeding logs including the drawn seed mass.

    Return event and host-property arrays, file IDs, malformed-log count, and seed masses.
    """
    output_file_names=os.listdir(output_path+'blackhole_seeding/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_complete=numpy.array([])
    BH_id_complete=numpy.array([],dtype=int)
    metallicity_complete=numpy.array([])
    SFR_complete=numpy.array([])
    density_complete=numpy.array([])
    FOFDMmass_complete=numpy.array([])
    indexmaxdens_complete=numpy.array([],dtype=int)
    FOFTask_complete=numpy.array([],dtype=int)
    DrawnSeedMass_complete=numpy.array([])
    #Gas_Hsml_complete=numpy.array([])
    #BH_Hsml_complete=numpy.array([])
    #BH_mass2_complete=numpy.array([])

    N_empty=0

    for name in output_file_names[:]:
        data=numpy.loadtxt(output_path+'blackhole_seeding/'+name)

        try:
        #for ii in [1]:
            if (data.shape==(11,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                BH_id=numpy.array([data[2].astype(int)])
                density=numpy.array([data[3]])
                metallicity=numpy.array([data[4]])
                SFR=numpy.array([data[5]])
                FOFDMmass=numpy.array([data[6]])
                indexmaxdens=numpy.array([data[7].astype(int)])
                FOFTask=numpy.array([data[8].astype(int)])
                DrawnSeedMass=numpy.array([data[10]])

            else:    
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                BH_id=data[:,2].astype(int)
                density=data[:,3]    
                metallicity=data[:,4]
                SFR=data[:,5]
                FOFDMmass=data[:,6]
                indexmaxdens=data[:,7].astype(int)
                FOFTask=data[:,8].astype(int)
                DrawnSeedMass=numpy.array([data[:,10]])
              
            #print(

            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
            BH_id_complete=numpy.append(BH_id_complete,BH_id)
            #BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
            metallicity_complete=numpy.append(metallicity_complete,metallicity)
            SFR_complete=numpy.append(SFR_complete,SFR) 
            density_complete=numpy.append(density_complete,density)
            FOFDMmass_complete=numpy.append(FOFDMmass_complete,FOFDMmass)
            indexmaxdens_complete=numpy.append(indexmaxdens_complete,indexmaxdens)
            FOFTask_complete=numpy.append(FOFTask_complete,FOFTask)
            DrawnSeedMass_complete=numpy.append(DrawnSeedMass_complete,DrawnSeedMass)
            #Gas_Hsml_complete=numpy.append(Gas_Hsml_complete,Gas_Hsml)
            #BH_Hsml_complete=numpy.append(BH_Hsml_complete,BH_Hsml)
        except IndexError:
            N_empty+=1
            aaa=1
           # print('Index err:', name)
    
    return scale_fac_complete,BH_id_complete,density_complete,metallicity_complete,SFR_complete,FOFDMmass_complete,indexmaxdens_complete,FOFTask_complete, file_id_complete,N_empty,DrawnSeedMass_complete

def get_seeding_events2(output_path):

    """Read the older extended logs in blackhole_seeding2_backup.

    Return seeding and host-property arrays, including star-forming gas properties and a malformed-log count.
    """
    output_file_names=os.listdir(output_path+'blackhole_seeding2_backup/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_complete=numpy.array([])
    BH_id_complete=numpy.array([],dtype=int)
    metallicity_complete=numpy.array([])
    SFR_complete=numpy.array([])
    density_complete=numpy.array([])
    FOFDMmass_complete=numpy.array([])
    indexmaxdens_complete=numpy.array([],dtype=int)
    FOFStarFormingGasMass_complete=numpy.array([])
    FOFStarFormingGasMetallicity_complete=numpy.array([])
    #BH_mass2_complete=numpy.array([])

    N_empty=0

    for name in output_file_names[:]:
        try:
            data=numpy.loadtxt(output_path+'blackhole_seeding2_backup/'+name)
        except:
            iii=1
        try:
        #for ii in [1]:
            if (data.shape==(11,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                BH_id=numpy.array([data[2].astype(int)])
                density=numpy.array([data[3]])
                metallicity=numpy.array([data[4]])
                SFR=numpy.array([data[5]])
                FOFDMmass=numpy.array([data[6]])
                indexmaxdens=numpy.array([data[7].astype(int)])
                FOFStarFormingGasMass=numpy.array([data[8]]) 
                FOFStarFormingGasMetallicity=numpy.array([data[9]])
            else:    
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                BH_id=data[:,2].astype(int)
                density=data[:,3]    
                metallicity=data[:,4]
                SFR=data[:,5]
                FOFDMmass=data[:,6]
                indexmaxdens=data[:,7].astype(int)
                FOFStarFormingGasMass=data[:,8]
                FOFStarFormingGasMetallicity=data[:,9]
            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
            BH_id_complete=numpy.append(BH_id_complete,BH_id)
            #BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
            metallicity_complete=numpy.append(metallicity_complete,metallicity)
            SFR_complete=numpy.append(SFR_complete,SFR) 
            density_complete=numpy.append(density_complete,density)
            FOFDMmass_complete=numpy.append(FOFDMmass_complete,FOFDMmass)
            indexmaxdens_complete=numpy.append(indexmaxdens_complete,indexmaxdens)
            FOFStarFormingGasMass_complete=numpy.append(FOFStarFormingGasMass_complete,FOFStarFormingGasMass)
            FOFStarFormingGasMetallicity_complete=numpy.append(FOFStarFormingGasMetallicity_complete,FOFStarFormingGasMetallicity)
        except IndexError:
            N_empty+=1
            aaa=1
            #print('Index err:', name)
    
    return scale_fac_complete,BH_id_complete,density_complete,metallicity_complete,SFR_complete,FOFDMmass_complete,indexmaxdens_complete,file_id_complete,FOFStarFormingGasMass_complete,FOFStarFormingGasMetallicity_complete,N_empty



def get_seeding_events3(output_path,GET_ENVIRONMENT=0):

    """Read extended backup seeding logs with stellar and metal-free gas properties.

    Return event arrays and a malformed-log count; GET_ENVIRONMENT appends major-neighbor counts.
    """
    output_file_names=os.listdir(output_path+'blackhole_seeding2_backup/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_complete=numpy.array([])
    BH_id_complete=numpy.array([],dtype=int)
    StellarMetallicity_complete=numpy.array([])
    SFR_complete=numpy.array([])
    StellarMass_complete=numpy.array([])
    FOFDMmass_complete=numpy.array([])
    GasMetallicity_complete=numpy.array([])
    FOFStarFormingGasMass_complete=numpy.array([])
    FOFStarFormingGasMetallicity_complete=numpy.array([])
    #BH_mass2_complete=numpy.array([])
    FOFStarFormingMetalFreeGasMass_complete=numpy.array([])
    if(GET_ENVIRONMENT):
        FOFNumberOfMajorNeighbors_complete=numpy.array([])
    
    N_empty=0

    for name in output_file_names[:]:
        try:
            data=numpy.loadtxt(output_path+'blackhole_seeding2_backup/'+name)


        #for ii in [1]:
            if (data.shape==(11+GET_ENVIRONMENT,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                BH_id=numpy.array([data[2].astype(int)])
                StellarMass=numpy.array([data[3]])
                StellarMetallicity=numpy.array([data[4]])
                SFR=numpy.array([data[5]])
                FOFDMmass=numpy.array([data[6]])
                GasMetallicity=numpy.array([data[7]])
                FOFStarFormingGasMass=numpy.array([data[8]]) 
                FOFStarFormingGasMetallicity=numpy.array([data[9]])
                FOFStarFormingMetalFreeGasMass=numpy.array([data[10]])
                if(GET_ENVIRONMENT):
                    FOFNumberOfMajorNeighbors=numpy.array([data[11]])
            else:    
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                BH_id=data[:,2].astype(int)
                StellarMass=data[:,3]    
                StellarMetallicity=data[:,4]
                SFR=data[:,5]
                FOFDMmass=data[:,6]
                GasMetallicity=data[:,7]
                FOFStarFormingGasMass=data[:,8]
                FOFStarFormingGasMetallicity=data[:,9]
                FOFStarFormingMetalFreeGasMass=data[:,10]
                if(GET_ENVIRONMENT):
                    FOFNumberOfMajorNeighbors=data[:,11]
            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
            BH_id_complete=numpy.append(BH_id_complete,BH_id)
            #BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
            StellarMetallicity_complete=numpy.append(StellarMetallicity_complete,StellarMetallicity)
            SFR_complete=numpy.append(SFR_complete,SFR) 
            StellarMass_complete=numpy.append(StellarMass_complete,StellarMass)
            FOFDMmass_complete=numpy.append(FOFDMmass_complete,FOFDMmass)
            GasMetallicity_complete=numpy.append(GasMetallicity_complete,GasMetallicity)
            FOFStarFormingGasMass_complete=numpy.append(FOFStarFormingGasMass_complete,FOFStarFormingGasMass)
            FOFStarFormingMetalFreeGasMass_complete=numpy.append(FOFStarFormingMetalFreeGasMass_complete,FOFStarFormingMetalFreeGasMass)
            FOFStarFormingGasMetallicity_complete=numpy.append(FOFStarFormingGasMetallicity_complete,FOFStarFormingGasMetallicity)
            if(GET_ENVIRONMENT):
                FOFNumberOfMajorNeighbors_complete=numpy.append(FOFNumberOfMajorNeighbors_complete,FOFNumberOfMajorNeighbors)
        except:
            print(name)
            N_empty+=1
            aaa=1
            #print('Index err:', name)
    if(GET_ENVIRONMENT):
        return scale_fac_complete,BH_id_complete,StellarMass_complete,StellarMetallicity_complete,SFR_complete,FOFDMmass_complete,GasMetallicity_complete,file_id_complete,FOFStarFormingGasMass_complete,FOFStarFormingGasMetallicity_complete,FOFStarFormingMetalFreeGasMass_complete,N_empty,FOFNumberOfMajorNeighbors_complete    
    return scale_fac_complete,BH_id_complete,StellarMass_complete,StellarMetallicity_complete,SFR_complete,FOFDMmass_complete,GasMetallicity_complete,file_id_complete,FOFStarFormingGasMass_complete,FOFStarFormingGasMetallicity_complete,FOFStarFormingMetalFreeGasMass_complete,N_empty



def get_seeding_events4(output_path,GET_ENVIRONMENT=0):

    """Read extended logs in blackhole_seeding2 including drawn seed masses.

    Return event and host-property arrays with neighbor counts and drawn seed masses.
    """
    output_file_names=os.listdir(output_path+'blackhole_seeding2/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_complete=numpy.array([])
    BH_id_complete=numpy.array([],dtype=int)
    StellarMetallicity_complete=numpy.array([])
    SFR_complete=numpy.array([])
    StellarMass_complete=numpy.array([])
    FOFDMmass_complete=numpy.array([])
    GasMetallicity_complete=numpy.array([])
    FOFStarFormingGasMass_complete=numpy.array([])
    FOFStarFormingGasMetallicity_complete=numpy.array([])
    #BH_mass2_complete=numpy.array([])
    FOFStarFormingMetalFreeGasMass_complete=numpy.array([])
    FOFNumberOfMajorNeighbors_complete=numpy.array([])
    FOFDrawnSeedMass_complete=numpy.array([])
    
    N_empty=0

    for name in output_file_names[:]:
        #print(name)
        data=numpy.loadtxt(output_path+'blackhole_seeding2/'+name)

        try:
        #for ii in [1]:
            if (data.shape==(13,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                BH_id=numpy.array([data[2].astype(int)])
                StellarMass=numpy.array([data[3]])
                StellarMetallicity=numpy.array([data[4]])
                SFR=numpy.array([data[5]])
                FOFDMmass=numpy.array([data[6]])
                GasMetallicity=numpy.array([data[7]])
                FOFStarFormingGasMass=numpy.array([data[8]]) 
                FOFStarFormingGasMetallicity=numpy.array([data[9]])
                FOFStarFormingMetalFreeGasMass=numpy.array([data[10]])
                FOFNumberOfMajorNeighbors=numpy.array([data[11]])
                FOFDrawnSeedMass=numpy.array([data[12]])
            else:    
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                BH_id=data[:,2].astype(int)
                StellarMass=data[:,3]    
                StellarMetallicity=data[:,4]
                SFR=data[:,5]
                FOFDMmass=data[:,6]
                GasMetallicity=data[:,7]
                FOFStarFormingGasMass=data[:,8]
                FOFStarFormingGasMetallicity=data[:,9]
                FOFStarFormingMetalFreeGasMass=data[:,10]
                FOFNumberOfMajorNeighbors=data[:,11]
                FOFDrawnSeedMass=data[:,12]
            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
            BH_id_complete=numpy.append(BH_id_complete,BH_id)
            #BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
            StellarMetallicity_complete=numpy.append(StellarMetallicity_complete,StellarMetallicity)
            SFR_complete=numpy.append(SFR_complete,SFR) 
            StellarMass_complete=numpy.append(StellarMass_complete,StellarMass)
            FOFDMmass_complete=numpy.append(FOFDMmass_complete,FOFDMmass)
            GasMetallicity_complete=numpy.append(GasMetallicity_complete,GasMetallicity)
            FOFStarFormingGasMass_complete=numpy.append(FOFStarFormingGasMass_complete,FOFStarFormingGasMass)
            FOFStarFormingMetalFreeGasMass_complete=numpy.append(FOFStarFormingMetalFreeGasMass_complete,FOFStarFormingMetalFreeGasMass)
            FOFStarFormingGasMetallicity_complete=numpy.append(FOFStarFormingGasMetallicity_complete,FOFStarFormingGasMetallicity)

            FOFNumberOfMajorNeighbors_complete=numpy.append(FOFNumberOfMajorNeighbors_complete,FOFNumberOfMajorNeighbors)
            FOFDrawnSeedMass_complete=numpy.append(FOFDrawnSeedMass_complete,FOFDrawnSeedMass)            
        except IndexError:
            N_empty+=1
            aaa=1
    return scale_fac_complete,BH_id_complete,StellarMass_complete,StellarMetallicity_complete,SFR_complete,FOFDMmass_complete,GasMetallicity_complete,file_id_complete,FOFStarFormingGasMass_complete,FOFStarFormingGasMetallicity_complete,FOFStarFormingMetalFreeGasMass_complete,N_empty,FOFNumberOfMajorNeighbors_complete,FOFDrawnSeedMass_complete




def get_phantommerger_events(output_path,get_primary_secondary_indices=0,HDF5=0,SORT_PRIMARY_SECONDARY=0):
    """Read phantom-merger events from text logs or HDF5.

    Return merger types, times, component masses/IDs, file IDs, and malformed-log count; sorting is optional.
    """
    N_empty=0
    if(HDF5):
        print("Note: reading merger events from the post processed hdf5 files")
        hf = h5py.File(output_path+'blackhole_phantommergers.hdf5')
        file_id_complete=hf.get('FileID')[:]
        scale_fac_complete=hf.get('ScaleFactor')[:]
        BH_id1_complete=hf.get('BH_ID1')[:]
        BH_mass1_complete=hf.get('BH_Mass1')[:]
        BH_id2_complete=hf.get('BH_ID2')[:]
        BH_mass2_complete=hf.get('BH_Mass2')[:]
        hf.close()
    else:
        output_file_names=os.listdir(output_path+'blackhole_phantommergers/')
        snapshot_space=[]
        redshift_space=[]

        file_id_complete=numpy.array([],dtype=int)
        scale_fac_complete=numpy.array([])

        BH_id1_complete=numpy.array([],dtype=int)
        BH_mass1_complete=numpy.array([])
        BH_id2_complete=numpy.array([],dtype=int)
        BH_mass2_complete=numpy.array([])
        merger_type_complete=numpy.array([],dtype=int)

        

        for name in output_file_names[:]:
            data=numpy.loadtxt(output_path+'blackhole_phantommergers/'+name)

            try:
                if (data.shape==(7,)):
                    merger_type=numpy.array([data[0].astype(int)])
                    file_id=numpy.array([data[1].astype(int)])
                    scale_fac=numpy.array([data[2]])
                    BH_id1=numpy.array([data[3].astype(int)])
                    BH_mass1=numpy.array([data[4]])
                    BH_id2=numpy.array([data[5].astype(int)])
                    BH_mass2=numpy.array([data[6]])                             
                else:    
                    merger_type=data[:,0].astype(int)
                    file_id=data[:,1].astype(int)
                    scale_fac=data[:,2]
                    BH_id1=data[:,3].astype(int)
                    BH_mass1=data[:,4]
                    BH_id2=data[:,5].astype(int)
                    BH_mass2=data[:,6]
                merger_type_complete=numpy.append(merger_type_complete,merger_type)        
                file_id_complete=numpy.append(file_id_complete,file_id)
                scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
                BH_id1_complete=numpy.append(BH_id1_complete,BH_id1)
                BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
                BH_id2_complete=numpy.append(BH_id2_complete,BH_id2)
                BH_mass2_complete=numpy.append(BH_mass2_complete,BH_mass2) 
            except IndexError:
                N_empty+=1
                aaa=1
    mass_tuple=list(zip(BH_mass1_complete,BH_mass2_complete))
    id_tuple=list(zip(BH_id1_complete,BH_id2_complete))

    #primary_mass=numpy.array([numpy.amax([dat[0],dat[1]]) for dat in mass_tuple])
    #secondary_mass=numpy.array([numpy.amin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_index=numpy.array([numpy.argmax([dat[0],dat[1]]) for dat in mass_tuple])
    secondary_index=numpy.array([numpy.argmin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,primary_index))])
    secondary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,secondary_index))])

    primary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,primary_index))])
    secondary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,secondary_index))])
    
    #scale_fac_complete_sorted=sort_X_based_on_Y(scale_fac_complete,scale_fac_complete)
    #primary_mass_sorted=sort_X_based_on_Y(primary_mass,scale_fac_complete)
    #secondary_mass_sorted=sort_X_based_on_Y(secondary_mass,scale_fac_complete)
    #primary_id_sorted=sort_X_based_on_Y(primary_id,scale_fac_complete)
    #secondary_id_sorted=sort_X_based_on_Y(secondary_id,scale_fac_complete)
    #file_id_complete_sorted=sort_X_based_on_Y(file_id_complete,scale_fac_complete)
    
    scale_fac_complete_sorted=scale_fac_complete
    primary_mass_sorted=primary_mass
    secondary_mass_sorted=secondary_mass
    primary_id_sorted=primary_id
    secondary_id_sorted=secondary_id
    file_id_complete_sorted=file_id_complete
    if SORT_PRIMARY_SECONDARY:
        return merger_type_complete,scale_fac_complete_sorted,primary_mass_sorted,secondary_mass_sorted,primary_id_sorted,secondary_id_sorted,file_id_complete_sorted,N_empty
    else:
        return merger_type_complete,scale_fac_complete_sorted,BH_mass1_complete,BH_mass2_complete,BH_id1_complete,BH_id2_complete,file_id_complete_sorted,N_empty   

def get_merger_events_kin(output_path,get_primary_secondary_indices=0,HDF5=0,SORT_PRIMARY_SECONDARY=0):
    """Read merger events with smoothing lengths and kinetic/potential energies.

    Return event arrays; sorting primary/secondary components returns only the core merger fields.
    """
    N_empty=0
    if(HDF5):
        print("Note: reading merger events from the post processed hdf5 files")
        hf = h5py.File(output_path+'blackhole_mergers.hdf5')
        file_id_complete=hf.get('FileID')[:]
        scale_fac_complete=hf.get('ScaleFactor')[:]
        BH_id1_complete=hf.get('BH_ID1')[:]
        BH_mass1_complete=hf.get('BH_Mass1')[:]
        BH_id2_complete=hf.get('BH_ID2')[:]
        BH_mass2_complete=hf.get('BH_Mass1')[:]
        BH_Hsml1_complete=hf.get('BH_Hsml1')[:]
        BH_Hsml2_complete=hf.get('BH_Hsml2')[:]
        K_Energy_complete=hf.get('K_Energy')[:]
        P_Energy_complete=hf.get('P_Energy')[:]        
        hf.close()
    else:
        output_file_names=os.listdir(output_path+'blackhole_mergers/')
        snapshot_space=[]
        redshift_space=[]

        file_id_complete=numpy.array([],dtype=int)
        scale_fac_complete=numpy.array([])

        BH_id1_complete=numpy.array([],dtype=int)
        BH_mass1_complete=numpy.array([])
        BH_id2_complete=numpy.array([],dtype=int)
        BH_mass2_complete=numpy.array([])
        BH_Hsml1_complete=numpy.array([])
        BH_Hsml2_complete=numpy.array([])
        K_Energy_complete=numpy.array([])
        P_Energy_complete=numpy.array([]) 

        for name in output_file_names[:]:
            #print(name)
            data=numpy.loadtxt(output_path+'blackhole_mergers/'+name)

            try:
                if (data.shape==(10,)):
                    file_id=numpy.array([data[0].astype(int)])
                    scale_fac=numpy.array([data[1]])
                    BH_id1=numpy.array([data[2].astype(int)])
                    BH_mass1=numpy.array([data[3]])
                    BH_id2=numpy.array([data[4].astype(int)])
                    BH_mass2=numpy.array([data[5]])
                    BH_Hsml1=numpy.array([data[6]])
                    BH_Hsml2=numpy.array([data[7]])
                    K_Energy=numpy.array([data[8]])
                    P_Energy=numpy.array([data[9]])
                    
                else:    
                    file_id=data[:,0].astype(int)
                    scale_fac=data[:,1]
                    BH_id1=data[:,2].astype(int)
                    BH_mass1=data[:,3]
                    BH_id2=data[:,4].astype(int)
                    BH_mass2=data[:,5]
                    BH_Hsml1=data[:,6]
                    BH_Hsml2=data[:,7]
                    K_Energy=data[:,8]
                    P_Energy=data[:,9]

                file_id_complete=numpy.append(file_id_complete,file_id)
                scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
                BH_id1_complete=numpy.append(BH_id1_complete,BH_id1)
                BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
                BH_id2_complete=numpy.append(BH_id2_complete,BH_id2)
                BH_mass2_complete=numpy.append(BH_mass2_complete,BH_mass2) 
                BH_Hsml1_complete=numpy.append(BH_Hsml1_complete,BH_Hsml1) 
                BH_Hsml2_complete=numpy.append(BH_Hsml2_complete,BH_Hsml2) 
                K_Energy_complete=numpy.append(K_Energy_complete,K_Energy) 
                P_Energy_complete=numpy.append(P_Energy_complete,P_Energy) 
            except IndexError:
                N_empty+=1
                aaa=1
    mass_tuple=list(zip(BH_mass1_complete,BH_mass2_complete))
    id_tuple=list(zip(BH_id1_complete,BH_id2_complete))

    primary_index=numpy.array([numpy.argmax([dat[0],dat[1]]) for dat in mass_tuple])
    secondary_index=numpy.array([numpy.argmin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,primary_index))])
    secondary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,secondary_index))])

    primary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,primary_index))])
    secondary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,secondary_index))])
    
    if SORT_PRIMARY_SECONDARY:
        return scale_fac_complete,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty
    else:
        return scale_fac_complete,BH_mass1_complete,BH_mass2_complete,BH_id1_complete,BH_id2_complete,file_id_complete,N_empty,BH_Hsml1_complete,BH_Hsml2_complete,K_Energy_complete,P_Energy_complete

def get_merger_events_debug(output_path,get_primary_secondary_indices=0,HDF5=0,SORT_PRIMARY_SECONDARY=0):
    """Read merger logs with component smoothing lengths.

    Return event arrays and malformed-log count; mass-sorted output omits smoothing lengths.
    """
    N_empty=0
    if(HDF5):
        print("Note: reading merger events from the post processed hdf5 files")
        hf = h5py.File(output_path+'blackhole_mergers.hdf5')
        file_id_complete=hf.get('FileID')[:]
        scale_fac_complete=hf.get('ScaleFactor')[:]
        BH_id1_complete=hf.get('BH_ID1')[:]
        BH_mass1_complete=hf.get('BH_Mass1')[:]
        BH_id2_complete=hf.get('BH_ID2')[:]
        BH_mass2_complete=hf.get('BH_Mass1')[:]
        BH_Hsml1_complete=hf.get('BH_Hsml1')[:]
        BH_Hsml2_complete=hf.get('BH_Hsml2')[:]

        
        hf.close()
    else:
        output_file_names=os.listdir(output_path+'blackhole_mergers/')
        snapshot_space=[]
        redshift_space=[]

        file_id_complete=numpy.array([],dtype=int)
        scale_fac_complete=numpy.array([])

        BH_id1_complete=numpy.array([],dtype=int)
        BH_mass1_complete=numpy.array([])
        BH_id2_complete=numpy.array([],dtype=int)
        BH_mass2_complete=numpy.array([])
        BH_Hsml1_complete=numpy.array([])
        BH_Hsml2_complete=numpy.array([])       

        for name in output_file_names[:]:
            #print(name)
            data=numpy.loadtxt(output_path+'blackhole_mergers/'+name)

            try:
                if (data.shape==(8,)):
                    file_id=numpy.array([data[0].astype(int)])
                    scale_fac=numpy.array([data[1]])
                    BH_id1=numpy.array([data[2].astype(int)])
                    BH_mass1=numpy.array([data[3]])
                    BH_id2=numpy.array([data[4].astype(int)])
                    BH_mass2=numpy.array([data[5]])
                    BH_Hsml1=numpy.array([data[6]])
                    BH_Hsml2=numpy.array([data[7]])
                    
                else:    
                    file_id=data[:,0].astype(int)
                    scale_fac=data[:,1]
                    BH_id1=data[:,2].astype(int)
                    BH_mass1=data[:,3]
                    BH_id2=data[:,4].astype(int)
                    BH_mass2=data[:,5]
                    BH_Hsml1=data[:,6]
                    BH_Hsml2=data[:,7]

                file_id_complete=numpy.append(file_id_complete,file_id)
                scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
                BH_id1_complete=numpy.append(BH_id1_complete,BH_id1)
                BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
                BH_id2_complete=numpy.append(BH_id2_complete,BH_id2)
                BH_mass2_complete=numpy.append(BH_mass2_complete,BH_mass2) 
                BH_Hsml1_complete=numpy.append(BH_Hsml1_complete,BH_Hsml1) 
                BH_Hsml2_complete=numpy.append(BH_Hsml2_complete,BH_Hsml2) 
            except IndexError:
                N_empty+=1
                aaa=1
    mass_tuple=list(zip(BH_mass1_complete,BH_mass2_complete))
    id_tuple=list(zip(BH_id1_complete,BH_id2_complete))

    primary_index=numpy.array([numpy.argmax([dat[0],dat[1]]) for dat in mass_tuple])
    secondary_index=numpy.array([numpy.argmin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,primary_index))])
    secondary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,secondary_index))])

    primary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,primary_index))])
    secondary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,secondary_index))])
    
    if SORT_PRIMARY_SECONDARY:
        return scale_fac_complete,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty
    else:
        return scale_fac_complete,BH_mass1_complete,BH_mass2_complete,BH_id1_complete,BH_id2_complete,file_id_complete,N_empty,BH_Hsml1_complete,BH_Hsml2_complete   


def get_merger_events(output_path,get_primary_secondary_indices=0,HDF5=0,SORT_PRIMARY_SECONDARY=0):
    """Combine black-hole merger records from text logs or HDF5.

    Return scale factors, component masses/IDs, file IDs, and malformed-log count; optionally sort by mass.
    """
    N_empty=0
    if(HDF5):
        print("Note: reading merger events from the post processed hdf5 files")
        hf = h5py.File(output_path+'blackhole_mergers.hdf5')
        file_id_complete=hf.get('FileID')[:]
        scale_fac_complete=hf.get('ScaleFactor')[:]
        BH_id1_complete=hf.get('BH_ID1')[:]
        BH_mass1_complete=hf.get('BH_Mass1')[:]
        BH_id2_complete=hf.get('BH_ID2')[:]
        BH_mass2_complete=hf.get('BH_Mass2')[:]
        hf.close()
    else:
        output_file_names=os.listdir(output_path+'blackhole_mergers/')
        snapshot_space=[]
        redshift_space=[]

        file_id_complete=numpy.array([],dtype=int)
        scale_fac_complete=numpy.array([])

        BH_id1_complete=numpy.array([],dtype=int)
        BH_mass1_complete=numpy.array([])
        BH_id2_complete=numpy.array([],dtype=int)
        BH_mass2_complete=numpy.array([])

        

        for name in output_file_names[:]:
            #print(name)
            data=numpy.loadtxt(output_path+'blackhole_mergers/'+name)

            try:
                if (data.shape==(6,)):
                    file_id=numpy.array([data[0].astype(int)])
                    scale_fac=numpy.array([data[1]])
                    BH_id1=numpy.array([data[2].astype(int)])
                    BH_mass1=numpy.array([data[3]])
                    BH_id2=numpy.array([data[4].astype(int)])
                    BH_mass2=numpy.array([data[5]])                             
                else:    
                    file_id=data[:,0].astype(int)
                    scale_fac=data[:,1]
                    BH_id1=data[:,2].astype(int)
                    BH_mass1=data[:,3]
                    BH_id2=data[:,4].astype(int)
                    BH_mass2=data[:,5]

                file_id_complete=numpy.append(file_id_complete,file_id)
                scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)
                BH_id1_complete=numpy.append(BH_id1_complete,BH_id1)
                BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)    
                BH_id2_complete=numpy.append(BH_id2_complete,BH_id2)
                BH_mass2_complete=numpy.append(BH_mass2_complete,BH_mass2) 
            except IndexError:
                N_empty+=1
                aaa=1
    mass_tuple=list(zip(BH_mass1_complete,BH_mass2_complete))
    id_tuple=list(zip(BH_id1_complete,BH_id2_complete))

    #primary_mass=numpy.array([numpy.amax([dat[0],dat[1]]) for dat in mass_tuple])
    #secondary_mass=numpy.array([numpy.amin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_index=numpy.array([numpy.argmax([dat[0],dat[1]]) for dat in mass_tuple])
    secondary_index=numpy.array([numpy.argmin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,primary_index))])
    secondary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,secondary_index))])

    primary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,primary_index))])
    secondary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,secondary_index))])
    if SORT_PRIMARY_SECONDARY:
        return scale_fac_complete,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty
    else:
        return scale_fac_complete,BH_mass1_complete,BH_mass2_complete,BH_id1_complete,BH_id2_complete,file_id_complete,N_empty   


def get_merger_events_hosts(output_path,HDF5=0,SORT_PRIMARY_SECONDARY=0):
    """Read merger records with the host halo mass components.

    Return event, BH, and host-property arrays; optionally order both components by BH mass.
    """
    if (HDF5):
        hf = h5py.File(output_path+'blackhole_mergerhosts.hdf5')
        file_id_complete=hf.get('FileID')[:]
        scale_fac_complete=hf.get('ScaleFactor')[:]

        BH_id1_complete=hf.get('BH_ID1')[:]
        BH_mass1_complete=hf.get('BH_Mass1')[:]
        BH_id2_complete=hf.get('BH_ID2')[:]
        BH_mass2_complete=hf.get('BH_Mass2')[:]

        hosthalomass1_complete=hf.get('HostHaloMass1')[:]
        hosthalostellarmass1_complete=hf.get('HostHaloStellarMass1')[:]
        hosthalogasmass1_complete=hf.get('HostHaloGasMass1')[:]
        hosthalodmmass1_complete=hf.get('HostHaloDMMass1')[:]

        hosthalomass2_complete=hf.get('HostHaloMass2')[:]
        hosthalostellarmass2_complete=hf.get('HostHaloStellarMass2')[:]
        hosthalogasmass2_complete=hf.get('HostHaloGasMass2')[:]
        hosthalodmmass2_complete=hf.get('HostHaloDMMass2')[:]
        hf.close()
    
    else:
        output_file_names=os.listdir(output_path+'blackhole_mergerhosts/')
        snapshot_space=[]
        redshift_space=[]

        file_id_complete=numpy.array([],dtype=int)
        scale_fac_complete=numpy.array([])

        BH_id1_complete=numpy.array([],dtype=int)
        BH_mass1_complete=numpy.array([])
        BH_id2_complete=numpy.array([],dtype=int)
        BH_mass2_complete=numpy.array([])

        hosthalomass1_complete=numpy.array([])
        hosthalomass2_complete=numpy.array([])
        hosthalostellarmass1_complete=numpy.array([])
        hosthalostellarmass2_complete=numpy.array([])
        hosthalogasmass1_complete=numpy.array([])
        hosthalogasmass2_complete=numpy.array([])
        hosthalodmmass1_complete=numpy.array([])
        hosthalodmmass2_complete=numpy.array([])
        N_empty=0
        for name in output_file_names[:]:
            data=numpy.loadtxt(output_path+'blackhole_mergerhosts/'+name)

            try:
                if (data.shape==(14,)):
                    file_id=numpy.array([data[0].astype(int)])
                    scale_fac=numpy.array([data[1]])
                    BH_id1=numpy.array([data[2].astype(int)])
                    BH_mass1=numpy.array([data[3]])
                    hosthalomass1=numpy.array([data[4]])
                    hosthalostellarmass1=numpy.array([data[5]])
                    hosthalogasmass1=numpy.array([data[6]])
                    hosthalodmmass1=numpy.array([data[7]])
                    BH_id2=numpy.array([data[8].astype(int)])
                    BH_mass2=numpy.array([data[9]])
                    hosthalomass2=numpy.array([data[10]])
                    hosthalostellarmass2=numpy.array([data[11]])
                    hosthalogasmass2=numpy.array([data[12]])
                    hosthalodmmass2=numpy.array([data[13]])
                else:
                    file_id=data[:,0].astype(int)
                    scale_fac=data[:,1]
                    BH_id1=data[:,2].astype(int)
                    BH_mass1=data[:,3]
                    hosthalomass1=data[:,4]
                    hosthalostellarmass1=data[:,5]
                    hosthalogasmass1=data[:,6]
                    hosthalodmmass1=data[:,7]
                    BH_id2=data[:,8].astype(int)
                    BH_mass2=data[:,9]
                    hosthalomass2=data[:,10]
                    hosthalostellarmass2=data[:,11]
                    hosthalogasmass2=data[:,12]
                    hosthalodmmass2=data[:,13]

                file_id_complete=numpy.append(file_id_complete,file_id)
                scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)

                hosthalomass1_complete=numpy.append(hosthalomass1_complete,hosthalomass1)
                hosthalostellarmass1_complete=numpy.append(hosthalostellarmass1_complete,hosthalostellarmass1)    
                hosthalogasmass1_complete=numpy.append(hosthalogasmass1_complete,hosthalogasmass1)
                hosthalodmmass1_complete=numpy.append(hosthalodmmass1_complete,hosthalodmmass1)    

                hosthalomass2_complete=numpy.append(hosthalomass2_complete,hosthalomass2)
                hosthalostellarmass2_complete=numpy.append(hosthalostellarmass2_complete,hosthalostellarmass2)    
                hosthalogasmass2_complete=numpy.append(hosthalogasmass2_complete,hosthalogasmass2)
                hosthalodmmass2_complete=numpy.append(hosthalodmmass2_complete,hosthalodmmass2)    

                BH_id1_complete=numpy.append(BH_id1_complete,BH_id1)
                BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)
                BH_id2_complete=numpy.append(BH_id2_complete,BH_id2)
                BH_mass2_complete=numpy.append(BH_mass2_complete,BH_mass2)
            except IndexError:
                N_empty+=1
                aaa=1

    def fetch_primary_secondary(prop1,prop2,primary_index,secondary_index):
        """Reorder paired properties using the primary and secondary component indices.

        Return two arrays that keep host properties aligned with the selected BH ordering.
        """
        prop_tuple=list(zip(prop1,prop2))
        primary_prop=numpy.array([prop_t[index] for (prop_t,index) in list(zip(prop_tuple,primary_index))])
        secondary_prop=numpy.array([prop_t[index] for (prop_t,index) in list(zip(prop_tuple,secondary_index))])
        return primary_prop,secondary_prop

    mass_tuple=list(zip(BH_mass1_complete,BH_mass2_complete))
    id_tuple=list(zip(BH_id1_complete,BH_id2_complete))
    primary_index=numpy.array([numpy.argmax([dat[0],dat[1]]) for dat in mass_tuple])
    secondary_index=numpy.array([numpy.argmin([dat[0],dat[1]]) for dat in mass_tuple])
    BH_id_primary,BH_id_secondary=fetch_primary_secondary(BH_id1_complete, BH_id2_complete, primary_index, secondary_index)
    BH_mass_primary,BH_mass_secondary=fetch_primary_secondary(BH_mass1_complete, BH_mass2_complete, primary_index, secondary_index)
    hosthalomass_primary,hosthalomass_secondary=fetch_primary_secondary(hosthalomass1_complete, hosthalomass2_complete, primary_index, secondary_index)
    hosthalodmmass_primary,hosthalodmmass_secondary=fetch_primary_secondary(hosthalodmmass1_complete, hosthalodmmass2_complete,primary_index, secondary_index)
    hosthalogasmass_primary,hosthalogasmass_secondary=fetch_primary_secondary(hosthalogasmass1_complete, hosthalogasmass2_complete,primary_index, secondary_index)
    hosthalostellarmass_primary,hosthalostellarmass_secondary=fetch_primary_secondary(hosthalostellarmass1_complete, hosthalostellarmass2_complete,primary_index, secondary_index)
    hosthalodmmass_primary,hosthalodmmass_secondary=fetch_primary_secondary(hosthalodmmass1_complete, hosthalodmmass2_complete,primary_index, secondary_index)
    if (SORT_PRIMARY_SECONDARY):
        return scale_fac_complete,BH_id_primary,BH_mass_primary,hosthalomass_primary,hosthalostellarmass_primary,hosthalogasmass_primary,hosthalodmmass_primary,BH_id_secondary,BH_mass_secondary,hosthalomass_secondary,hosthalostellarmass_secondary,hosthalogasmass_secondary,hosthalodmmass_secondary,file_id_complete
    else:
        return scale_fac_complete,BH_id1_complete,BH_mass1_complete,hosthalomass1_complete,hosthalostellarmass1_complete,hosthalogasmass1_complete,hosthalodmmass1_complete,BH_id2_complete,BH_mass2_complete,hosthalomass2_complete,hosthalostellarmass2_complete,hosthalogasmass2_complete,hosthalodmmass2_complete,file_id_complete

                            
def get_merger_events_from_snapshot(output_path,desired_redshift,SORT_PRIMARY_SECONDARY=0,HOSTS=0):
    """Read merger catalog chunks at the snapshot nearest desired_redshift.

    Return merger arrays; HOSTS includes accretion and host fields unless mass-sorted output is requested.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False)
    print(output_snapshot,output_redshift)
    tot_mergers=0
    ID1=numpy.array([],dtype='uint64')
    ID2=numpy.array([],dtype='uint64')
    BH_Mass1=numpy.array([],dtype='float')
    BH_Mass2=numpy.array([],dtype='float')
    if(HOSTS):
        BH_Mdot1=numpy.array([],dtype='float')
        BH_Mdot2=numpy.array([],dtype='float')
        BH_HostHaloMass1=numpy.array([],dtype='float')
        BH_HostHaloMass2=numpy.array([],dtype='float')
        BH_HostStellarMass1=numpy.array([],dtype='float')
        BH_HostStellarMass2=numpy.array([],dtype='float')
        BH_HostGasMass1=numpy.array([],dtype='float')
        BH_HostGasMass2=numpy.array([],dtype='float')
        BH_HostSFR1=numpy.array([],dtype='float')
        BH_HostSFR2=numpy.array([],dtype='float')

    TaskID=numpy.array([],dtype='int')
    Time=numpy.array([],dtype='float')
    for i in range(0,16):
        try:
            if (output_snapshot>=10):
                output_snapshot_str='%d'%output_snapshot
            elif (output_snapshot<=9):
                output_snapshot_str='0%d'%output_snapshot
            #print(output_snapshot_str)
            g=h5py.File(output_path+'mergers_0%s/mergers_tab_0%s.%d.hdf5'%(output_snapshot_str,output_snapshot_str,i))
            header=g['Header'].attrs
            Merger=g.get('Merger')
            #print(list(Merger.keys()))
            #print(list(header))  
            #print("Number of mergers on File %d"%i,header.get('Nmergers_ThisFile'))
            tot_mergers+=header.get('Nmergers_ThisFile')
            ID1=numpy.append(ID1,Merger.get('ID1')[:])
            ID2=numpy.append(ID2,Merger.get('ID2')[:])
            
            BH_Mass1=numpy.append(BH_Mass1,Merger.get('BH_Mass1')[:])
            BH_Mass2=numpy.append(BH_Mass2,Merger.get('BH_Mass2')[:])
            
            if (HOSTS):
                BH_Mdot1=numpy.append(BH_Mdot1,Merger.get('BH_Mdot1')[:])
                BH_Mdot2=numpy.append(BH_Mdot2,Merger.get('BH_Mdot2')[:])

                BH_HostHaloMass1=numpy.append(BH_HostHaloMass1,Merger.get('BH_HostHaloMass1')[:])
                BH_HostHaloMass2=numpy.append(BH_HostHaloMass2,Merger.get('BH_HostHaloMass2')[:])

                BH_HostStellarMass1=numpy.append(BH_HostStellarMass1,Merger.get('BH_HostStellarMass1')[:])
                BH_HostStellarMass2=numpy.append(BH_HostStellarMass2,Merger.get('BH_HostStellarMass2')[:])

                BH_HostGasMass1=numpy.append(BH_HostGasMass1,Merger.get('BH_HostGasMass1')[:])
                BH_HostGasMass2=numpy.append(BH_HostGasMass2,Merger.get('BH_HostGasMass2')[:])

                BH_HostSFR1=numpy.append(BH_HostSFR1,Merger.get('BH_HostSFR1')[:])
                BH_HostSFR2=numpy.append(BH_HostSFR2,Merger.get('BH_HostSFR2')[:])

                
            TaskID=numpy.append(TaskID,Merger.get('TaskID')[:])
            Time=numpy.append(Time,Merger.get('Time')[:])
            #print(Merger.get('Time')[:])
        except TypeError:
            aa=1
    print("Total number of mergers",tot_mergers)

    mass_tuple=list(zip(BH_Mass1,BH_Mass2))
    id_tuple=list(zip(ID1,ID2))

        #primary_mass=numpy.array([numpy.amax([dat[0],dat[1]]) for dat in mass_tuple])
        #secondary_mass=numpy.array([numpy.amin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_index=numpy.array([numpy.argmax([dat[0],dat[1]]) for dat in mass_tuple])
    secondary_index=numpy.array([numpy.argmin([dat[0],dat[1]]) for dat in mass_tuple])

    primary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,primary_index))])
    secondary_mass=numpy.array([mass_t[index] for (mass_t,index) in list(zip(mass_tuple,secondary_index))])

    primary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,primary_index))])
    secondary_id=numpy.array([id_t[index] for (id_t,index) in list(zip(id_tuple,secondary_index))])
    
    #Time_sorted=sort_X_based_on_Y(Time,Time)
    #primary_mass_sorted=sort_X_based_on_Y(primary_mass,Time)
    #secondary_mass_sorted=sort_X_based_on_Y(secondary_mass,Time)
    #primary_id_sorted=sort_X_based_on_Y(primary_id,Time)
    #secondary_id_sorted=sort_X_based_on_Y(secondary_id,Time)
    #TaskID_sorted=sort_X_based_on_Y(TaskID,Time)
    if SORT_PRIMARY_SECONDARY:
        return Time,primary_mass,secondary_mass,primary_id,secondary_id,TaskID,0
    else:
        if (HOSTS):
            return Time,BH_Mass1,BH_Mass2,ID1,ID2,BH_Mdot1,BH_Mdot2,BH_HostHaloMass1,BH_HostHaloMass2,BH_HostStellarMass1,BH_HostStellarMass2,BH_HostGasMass1,BH_HostGasMass2,BH_HostSFR1,BH_HostSFR2,TaskID,0
        else:
            return Time,BH_Mass1,BH_Mass2,ID1,ID2,TaskID,0

        
        
def get_blackhole_history_high_res_all_progenitors(output_path,desired_id,mergers_from_snapshot=0,use_cleaned=0,get_all_blackhole_history=0,HDF5=0,ONLY_PROGENITORS=0,desired_id_redshift=0):
    """Collect detailed histories for a BH and its merger-connected IDs.

    Return IDs, scale factors, masses, accretion rates, densities, sound speeds, and merger times.
    """
    def parse_id_col(BH_ids_as_string):
        """Convert a prefixed BH identifier from a detail log to an integer.

        Strip the first three characters before conversion.
        """
        return numpy.int(BH_ids_as_string[3:])
    vec_parse_id_col=numpy.vectorize(parse_id_col)
    
    output_file_names=os.listdir(output_path+'blackhole_details/')

    BH_ids_for_id=numpy.array([],dtype=int)
    scale_factors_for_id=numpy.array([])
    BH_masses_for_id=numpy.array([])
    BH_mdots_for_id=numpy.array([])
    rhos_for_id=numpy.array([])
    sound_speeds_for_id=numpy.array([])
    
    try:
        total_desired_ids,merging_times=get_progenitors_and_descendants(output_path,desired_id,mergers_from_snapshot=mergers_from_snapshot,HDF5=HDF5,ONLY_PROGENITORS=ONLY_PROGENITORS,desired_id_redshift=desired_id_redshift)
        temp_merging_redshifts=numpy.append(1./merging_times-1,numpy.array([0.]))
    except:
    #if(1==1):
        merging_times=[]
        total_desired_ids=[desired_id]
    ii=0
    
    if (HDF5):
        hf = h5py.File(output_path+'blackhole_details.hdf5')
        BH_ids=hf.get('BH_ID')[:]
        scale_factors=hf.get('ScaleFactor')[:]
        merger_redshifts=1./scale_factors-1.

        BH_masses=hf.get('BH_Mass')[:]
        BH_mdots=hf.get('BH_Mdot')[:]
        rhos=hf.get('Rho')[:]
        sound_speeds=hf.get('cs')[:]  
        hf.close()
        if (len(total_desired_ids)>0):
            final_extract=numpy.array([False]*len(sound_speeds))
            for d_id,z_merger in zip(total_desired_ids,temp_merging_redshifts):
                extract_id=(d_id==BH_ids)
                #print("For id %d, the redshift cut off is %.4f$"%(d_id,z_merger))
                final_extract=final_extract+(extract_id)#&(merger_redshifts>z_merger))
            if (ONLY_PROGENITORS):
                mask_redshift=merger_redshifts>desired_id_redshift-1e-2
                final_extract=final_extract&mask_redshift
                
        if (get_all_blackhole_history==1):
            final_extract=BH_ids==BH_ids
        BH_ids_for_id=BH_ids[final_extract]
        scale_factors_for_id=scale_factors[final_extract]
        BH_masses_for_id=BH_masses[final_extract]
        BH_mdots_for_id=BH_mdots[final_extract]
        rhos_for_id=rhos[final_extract]
        sound_speeds_for_id=sound_speeds[final_extract]    
    else:
        for output_file_name in output_file_names[:]:
            if ('blackhole_details' in output_file_name):
                print(ii)
                ii+=1
                try:
                    if use_cleaned:
                        full_data=numpy.loadtxt(output_path+'blackhole_details_cleaned/'+output_file_name,dtype='str')
                    else:
                        full_data=numpy.loadtxt(output_path+'blackhole_details/'+output_file_name,dtype='str')
                except:

                    try:
                        print("Last row missing in ",output_file_name)
                        full_data=numpy.genfromtxt(output_path+'blackhole_details/'+output_file_name,dtype='str',skip_footer=1)
                    except:
                        print("Failed completely",output_file_name)
                        continue
                try:     
                    BH_ids=vec_parse_id_col(full_data[:,0])
                    scale_factors=(full_data[:,1]).astype('float')
                    BH_masses=(full_data[:,2]).astype('float')
                    BH_mdots=(full_data[:,3]).astype('float')
                    rhos=(full_data[:,4]).astype('float')
                    sound_speeds=(full_data[:,5]).astype('float')

                    final_extract=numpy.array([False]*len(sound_speeds))
                    if (len(total_desired_ids)>0):
                        for d_id in total_desired_ids:
                            extract_id=(d_id==BH_ids)
                            final_extract=final_extract+extract_id
                    #print(len(BH_ids),len(BH_ids[extract_id]))
                    if (get_all_blackhole_history==1):
                        final_extract=BH_ids==BH_ids
                    BH_ids_for_id=numpy.append(BH_ids_for_id,BH_ids[final_extract])
                    scale_factors_for_id=numpy.append(scale_factors_for_id,scale_factors[final_extract])
                    BH_masses_for_id=numpy.append(BH_masses_for_id,BH_masses[final_extract])
                    BH_mdots_for_id=numpy.append(BH_mdots_for_id,BH_mdots[final_extract])
                    rhos_for_id=numpy.append(rhos_for_id,rhos[final_extract])
                    sound_speeds_for_id=numpy.append(sound_speeds_for_id,sound_speeds[final_extract])
                except:
                    print("Warning, this file is not read")
                    aa=1
            
    return BH_ids_for_id,scale_factors_for_id,BH_masses_for_id,BH_mdots_for_id,rhos_for_id,sound_speeds_for_id,merging_times


def get_blackhole_history_high_res_all_progenitors_v2(output_path,desired_id):
    """Collect merger-connected BH histories from the legacy text detail logs.

    Return IDs, scale factors, masses, accretion rates, densities, sound speeds, and merger times.
    """
    def parse_id_col(BH_ids_as_string):
        """Convert a prefixed BH identifier from a detail log to an integer.

        Strip the first three characters before conversion.
        """
        return numpy.int(BH_ids_as_string[3:])
    vec_parse_id_col=numpy.vectorize(parse_id_col)
    
    output_file_names=os.listdir(output_path+'blackhole_details/')

    BH_ids_for_id=numpy.array([],dtype=int)
    scale_factors_for_id=numpy.array([])
    BH_masses_for_id=numpy.array([])
    BH_mdots_for_id=numpy.array([])
    rhos_for_id=numpy.array([])
    sound_speeds_for_id=numpy.array([])
    
    try:
        total_desired_ids,merging_times=get_progenitors_and_descendants(output_path,desired_id)
    except:
    #if(1==1):
        merging_times=[]
        total_desired_ids=[desired_id]
    ii=0
    for output_file_name in output_file_names[:]:
        if ('blackhole_details' in output_file_name):
            print(ii)
            ii+=1
            try:
                full_data=numpy.loadtxt(output_path+'blackhole_details/'+output_file_name,dtype='str')
           
                
                BH_ids=vec_parse_id_col(full_data[:,0])
                scale_factors=(full_data[:,1]).astype('float')
                BH_masses=(full_data[:,2]).astype('float')
                BH_mdots=(full_data[:,3]).astype('float')
                rhos=(full_data[:,4]).astype('float')
                sound_speeds=(full_data[:,5]).astype('float')

                final_extract=numpy.array([False]*len(sound_speeds))
                if (len(total_desired_ids)>0):
                    for d_id in total_desired_ids:
                        extract_id=(d_id==BH_ids)
                        final_extract=final_extract+extract_id
                #print(len(BH_ids),len(BH_ids[extract_id]))
                BH_ids_for_id=numpy.append(BH_ids_for_id,BH_ids[final_extract])
                scale_factors_for_id=numpy.append(scale_factors_for_id,scale_factors[final_extract])
                BH_masses_for_id=numpy.append(BH_masses_for_id,BH_masses[final_extract])
                BH_mdots_for_id=numpy.append(BH_mdots_for_id,BH_mdots[final_extract])
                rhos_for_id=numpy.append(rhos_for_id,rhos[final_extract])
                sound_speeds_for_id=numpy.append(sound_speeds_for_id,sound_speeds[final_extract])
            except ValueError:
                aa=1
            
    return BH_ids_for_id,scale_factors_for_id,BH_masses_for_id,BH_mdots_for_id,rhos_for_id,sound_speeds_for_id,merging_times




        
        
def get_progenitors_and_descendants(output_path,desired_id,MAX_ITERATION=100,mergers_from_snapshot=0,HDF5=0,ONLY_PROGENITORS=0,desired_id_redshift=0):

    """Expand the set of BH IDs connected through merger events.

    Return connected IDs and event scale factors; ONLY_PROGENITORS applies the redshift restriction.
    """
    BH_ids_for_id=numpy.array([],dtype=int)
    scale_factors_for_id=numpy.array([])
    BH_masses_for_id=numpy.array([])
    BH_mdots_for_id=numpy.array([])
    rhos_for_id=numpy.array([])
    sound_speeds_for_id=numpy.array([])
    if (mergers_from_snapshot):
        merging_time,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty=get_merger_events_from_snapshot(output_path,0)
    else:
        merging_time,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty,bbb,ccc=get_merger_events_debug(output_path,HDF5=HDF5)
    event_indices=numpy.arange(0,len(merging_time))
    progenitor_ids=numpy.array([desired_id],dtype=int)
    final_merging_times=numpy.array([])
    final_event_indices=numpy.array([],dtype=int)
    progenitor_ids_before_update=numpy.array([],dtype=int)
    i=0
    while (len(progenitor_ids_before_update)<len(progenitor_ids)):
            
        progenitor_ids_before_update=progenitor_ids+False
        
        extract_events_as_primary_BH=numpy.array([False]*len(secondary_id))
        extract_events_as_secondary_BH=numpy.array([False]*len(secondary_id))
        for ids in progenitor_ids:
            extract_events_as_secondary_BH=extract_events_as_secondary_BH+(secondary_id==ids)
            extract_events_as_primary_BH=extract_events_as_primary_BH+(primary_id==ids)
            if (ONLY_PROGENITORS):
                merger_redshift=1./merging_time-1.
                mask_redshift=merger_redshift>desired_id_redshift-1e-2
                extract_events_as_secondary_BH=extract_events_as_secondary_BH&mask_redshift
                extract_events_as_primary_BH=extract_events_as_primary_BH&mask_redshift
                
        merging_partner_ids=numpy.append(primary_id[extract_events_as_secondary_BH],secondary_id[extract_events_as_primary_BH]) 
        merge_times=numpy.append(merging_time[extract_events_as_secondary_BH],merging_time[extract_events_as_primary_BH])    
        merging_event_indices=numpy.append(event_indices[extract_events_as_secondary_BH],event_indices[extract_events_as_primary_BH])
 
        progenitor_ids=numpy.append(progenitor_ids,merging_partner_ids)
        final_merging_times=numpy.append(final_merging_times,merge_times)
        final_event_indices=numpy.append(final_event_indices,merging_event_indices)         
        progenitor_ids_before_distinct=progenitor_ids+0
        progenitor_ids=numpy.unique(progenitor_ids)
        
        final_event_indices=numpy.unique(final_event_indices)
       
        final_merging_times=numpy.unique(final_merging_times)
        i+=1
        if (i>MAX_ITERATION):
            print("Maximum number of iterations reached! Aborting")
            break
    return progenitor_ids,merging_time[final_event_indices]


def get_merging_event_indices(basePath,desired_id):
    """Find all HDF5 merger events connected to a chosen BH ID.

    Return unique event indices, unique merger scale factors, and connected BH IDs.
    """
    merging_time,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty=get_merger_events(basePath,HDF5=1)
    event_indices=numpy.arange(0,len(merging_time))
    
    progenitor_ids=numpy.array([desired_id],dtype=int)
    final_merging_times=numpy.array([])
    final_event_indices=numpy.array([],dtype=int)
    
    progenitor_ids_before_update=numpy.array([],dtype=int)
    i=0
    while (len(progenitor_ids_before_update)<len(progenitor_ids)):
            
        progenitor_ids_before_update=progenitor_ids+False
        
        extract_events_as_primary_BH=numpy.array([False]*len(secondary_id))
        extract_events_as_secondary_BH=numpy.array([False]*len(secondary_id))
        for ids in progenitor_ids:
            extract_events_as_secondary_BH=extract_events_as_secondary_BH+(secondary_id==ids)
            extract_events_as_primary_BH=extract_events_as_primary_BH+(primary_id==ids)
        merging_partner_ids=numpy.append(primary_id[extract_events_as_secondary_BH],secondary_id[extract_events_as_primary_BH])    
        merging_event_indices=numpy.append(event_indices[extract_events_as_secondary_BH],event_indices[extract_events_as_primary_BH])
        
        merge_times=numpy.append(merging_time[extract_events_as_secondary_BH],merging_time[extract_events_as_primary_BH])    

        progenitor_ids=numpy.append(progenitor_ids,merging_partner_ids)
        final_merging_times=numpy.append(final_merging_times,merge_times) 
        final_event_indices=numpy.append(final_event_indices,merging_event_indices)
        
        progenitor_ids_before_distinct=progenitor_ids+0
        progenitor_ids=numpy.unique(progenitor_ids)
        final_merging_times=numpy.unique(final_merging_times)
        final_event_indices=numpy.unique(final_event_indices)
    return final_event_indices,final_merging_times,progenitor_ids


def generate_group_ids(output_path,desired_redshift,p_type,save_output_path='./',group_type='groups',create=False):    
    """Assign particles to groups or load a cached assignment.

    Return group IDs and write a snapshot-specific NumPy cache when creating assignments.
    """
    global complete_particle_ids
    particle_property='ParticleIDs'
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift)
  
    if not os.path.exists(save_output_path):
        print("Making directory for storing groupids")
        os.makedirs(save_output_path)
    
    if ((os.path.exists(save_output_path+'group_ids_%d.npy'%output_snapshot))&(create==False)):
        print("File exists!! group ids exist already")
        group_ids=numpy.load(save_output_path+'group_ids_%d.npy'%output_snapshot)
        return group_ids
        
    else:
        print("Constructing_group_ids")
        
    
    
    complete_particle_ids,output_redshift=get_particle_property(output_path, particle_property, p_type, desired_redshift)
    #print(len(complete_particle_ids))
    group_ids=numpy.array([-1]*len(complete_particle_ids))
    def find_index(id_to_be_searched):
        #print(numpy.where(complete_particle_ids==id_to_be_searched))
        
        """Find positions of a particle ID in the complete snapshot ID array.

        Return all matching indices for the enclosing group-assignment routine.
        """
        return (numpy.where(complete_particle_ids==id_to_be_searched))[0]
    vec_find_index=numpy.vectorize(find_index)
    
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift,list_all=False)
    requested_property=brahma.snapshot.loadSubset(output_path,output_snapshot,p_type)[particle_property]

    print('Reading group lengths')
    if (group_type=='groups'):              
        group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift,list_all=False))
        group_lengths=group_lengths[:,p_type] 
        #group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])   
        print('Constructing offsets')
        group_offsets=numpy.append(0,(numpy.cumsum(group_lengths))[:-1])
       #return group_particles,output_redshift
    elif (group_type=='subhalo'):     
        
        print("Warning!! This is going to be very slow! Still under development")

        group_lengths,output_redshift=(get_group_property(output_path,'GroupLenType', desired_redshift,list_all=False))
        group_lengths=group_lengths[:,p_type] 
        group_offsets=numpy.array([sum(group_lengths[0:i]) for i in range(0,len(group_lengths))])
        
        subhalo_lengths,output_redshift=(get_subhalo_property(output_path,'SubhaloLenType', desired_redshift,list_all=False))
        subhalo_lengths=subhalo_lengths[:,p_type] 
        subhalo_indices=numpy.arange(0,len(subhalo_lengths))
        
        
        subhalo_group_number,output_redshift=(get_subhalo_property(output_path,'SubhaloGrNr', desired_redshift,list_all=False));

 
        #subhalo_indices=subhalo_indices[subhalo_group_number==desired_group_number]
        #final_index=(subhalo_final_indices[subhalo_indices==subhalo_index])[0]
        
        #group_particles=group_particles[subhalo_offsets[final_index]:subhalo_offsets[final_index]+subhalo_lengths[final_index]]
      
        #return subhalo_particles,group_particles,output_redshift     



    #MAX_ITERATIONS=1000000
    for subhalo_index in range(0,len(group_offsets)): 
        if(subhalo_index%10==0):
            aaa=1
            #print(subhalo_index)
            #print(len(group_ids[group_ids==-1]))
            #print("-------------")
        if (group_type=='groups'):
            #print(len(group_offsets),subhalo_index)
            complete_particle_ids_group_wise=requested_property[group_offsets[subhalo_index]:group_offsets[subhalo_index]+group_lengths[subhalo_index]]
        if (group_type=='subhalo'):     
            desired_group_number=subhalo_group_number[subhalo_index]  
            subhalo_lengths_for_group=subhalo_lengths[subhalo_group_number==desired_group_number]
            subhalo_offsets_for_group=numpy.array([sum(subhalo_lengths[0:i]) for i in range(0,len(subhalo_lengths))])

            mask=subhalo_group_number==desired_group_number
            #print(len(mask)),mask
            subhalo_indices_for_group=subhalo_indices[mask]  
            subhalo_final_indices=numpy.arange(0,len(subhalo_indices_for_group))
            group_particles=requested_property[group_offsets[desired_group_number]:group_offsets[desired_group_number]+group_lengths[desired_group_number]]   
            #print(subhalo_indices_for_group,subhalo_index)
            final_index=(subhalo_final_indices[subhalo_indices_for_group==subhalo_index])[0]
            complete_particle_ids_group_wise=group_particles[subhalo_offsets_for_group[final_index]:subhalo_offsets_for_group[final_index]+subhalo_lengths_for_group[final_index]]

        if (len(complete_particle_ids_group_wise)==0):
            continue
        #print(complete_particle_ids_group_wise)
        indices_to_be_assigned=vec_find_index(complete_particle_ids_group_wise)
        
        group_ids[indices_to_be_assigned]=subhalo_index
        #print(len(group_ids[group_ids==-1]))
        if(len(group_ids[group_ids==-1])==0):
            break
    
    #if (subhalo_index==MAX_ITERATIONS-1):
    #    print("Warning! Maximum number of iterations reached!")
    print(output_redshift,output_snapshot)
    numpy.save(save_output_path+'group_ids_%d.npy'%output_snapshot,group_ids)
    return group_ids
            


def generate_subhalo_ids(output_path,desired_redshift,p_type,save_output_path='./',create=False):
    """Assign particles to the nearest subhalo within their parent FoF group.

    Return IDs and squared periodic distances; load or write snapshot-specific NumPy caches.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift)   
    if ((os.path.exists(save_output_path+'subhalo_ids_%d.npy'%output_snapshot))&(create==False)):
        print("File exists!! subhalo ids exist already")
        subhalo_ids=numpy.load(save_output_path+'subhalo_ids_%d.npy'%output_snapshot)
        distance_from_subhalo_center=numpy.load(save_output_path+'distance_from_subhalo_center_%d.npy'%output_snapshot)
        return subhalo_ids,distance_from_subhalo_center
    
    
    group_ids=generate_group_ids(output_path,desired_redshift,p_type,save_output_path=save_output_path)
    BH_Pos,output_redshift=get_particle_property(output_path,'Coordinates',p_type,desired_redshift,list_all=False)
    ParticleIDs,output_redshift=get_particle_property(output_path,'ParticleIDs',p_type,desired_redshift,list_all=False)
    boxsize=get_box_size(output_path)

    SubhaloGrNr,output_redshift=get_subhalo_property(output_path,'SubhaloGrNr',desired_redshift,list_all=False)
    SubhaloPos,output_redshift=get_subhalo_property(output_path,'SubhaloPos',desired_redshift,list_all=False)
    SubhaloMass,output_redshift=get_subhalo_property(output_path,'SubhaloMass',desired_redshift,list_all=False)
    SubhaloMass=SubhaloMass*1e10
    Subhalo_Indices=numpy.arange(len(SubhaloGrNr))
    mask=SubhaloMass==SubhaloMass
    #mask=SubhaloMass>=1e11
    Subhalo_Indices_cut=Subhalo_Indices[mask]
    SubhaloPos_cut=SubhaloPos[mask]
    SubhaloGrNr_cut=SubhaloGrNr[mask]

    def min_dis(median_position, position,box_size):
            """Choose the signed displacement with smallest absolute periodic distance.

            Compare the direct displacement with its two neighboring-box images.
            """
            pos_1=position-median_position
            pos_2=position-median_position+boxsize
            pos_3=position-median_position-boxsize

            new_position_options=numpy.array([pos_1,pos_2,pos_3])
            get_minimum_distance=numpy.argmin(numpy.abs(new_position_options))
            #print(new_position_options)

            #print(get_minimum_distance)
            return new_position_options[get_minimum_distance]


    def get_subhalo_id(blackhole_info):
        #print(blackhole_info)
        """Find the nearest eligible subhalo in the particle's parent FoF group.

        Return its index and squared periodic distance, or (-1, -1) for no candidates.
        """
        blackhole_group_id=blackhole_info[0]
        blackhole_position=blackhole_info[1]
        extract_ids_within_the_parent_FOF=blackhole_group_id==SubhaloGrNr_cut
        Subhalo_Indices_current_BH=Subhalo_Indices_cut[extract_ids_within_the_parent_FOF]
        #print(Subhalo_Indices_current_BH)
        SubhaloPos_current_BH=SubhaloPos_cut[extract_ids_within_the_parent_FOF]
        x_pos_Subhalo=SubhaloPos_current_BH[:,0]
        y_pos_Subhalo=SubhaloPos_current_BH[:,1]
        z_pos_Subhalo=SubhaloPos_current_BH[:,2]

        vectorized_min_dis = numpy.vectorize(min_dis)
        try:
            x_dis=vectorized_min_dis(blackhole_position[0],x_pos_Subhalo,boxsize)
            y_dis=vectorized_min_dis(blackhole_position[1],y_pos_Subhalo,boxsize)
            z_dis=vectorized_min_dis(blackhole_position[2],z_pos_Subhalo,boxsize)

            distance_sq=x_dis**2+y_dis**2+z_dis**2
            min_distance_sq=numpy.amin(distance_sq)
            subhalo_id=(Subhalo_Indices_current_BH[distance_sq==min_distance_sq])[0]
            return subhalo_id,min_distance_sq
        except ValueError:
            return -1,-1


    blackhole_info_space=list(zip(group_ids,BH_Pos))
    data=[get_subhalo_id(blackhole_info) for blackhole_info in blackhole_info_space]
    subhalo_ids=(numpy.array(data))[:,0]
    distance_from_subhalo_center=(numpy.array(data))[:,1]
    numpy.save(save_output_path+'subhalo_ids_%d.npy'%output_snapshot,subhalo_ids)
    numpy.save(save_output_path+'distance_from_subhalo_center_%d.npy'%output_snapshot,distance_from_subhalo_center)    
    return subhalo_ids,distance_from_subhalo_center

def generate_subhalo_ids_beta(output_path,desired_redshift,p_type,save_output_path='./',create=False):
    """Assign particles to nearby subhalos with nonzero mass in the requested type.

    Return IDs and squared periodic distances within each parent FoF group, using NumPy caches.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(output_path,desired_redshift)   
    if ((os.path.exists(save_output_path+'group_ids_%d.npy'%output_snapshot))&(create==False)):
        print("File exists!! subhalo ids exist already")
        subhalo_ids=numpy.load(save_output_path+'subhalo_ids_%d.npy'%output_snapshot)
        distance_from_subhalo_center=numpy.load(save_output_path+'distance_from_subhalo_center_%d.npy'%output_snapshot)
        return subhalo_ids,distance_from_subhalo_center
    
    
    group_ids=generate_group_ids(output_path,desired_redshift,p_type,save_output_path=save_output_path)
    BH_Pos,output_redshift=get_particle_property(output_path,'Coordinates',p_type,desired_redshift,list_all=False)
    ParticleIDs,output_redshift=get_particle_property(output_path,'ParticleIDs',p_type,desired_redshift,list_all=False)
    boxsize=get_box_size(output_path)

    SubhaloGrNr,output_redshift=get_subhalo_property(output_path,'SubhaloGrNr',desired_redshift,list_all=False)
    SubhaloPos,output_redshift=get_subhalo_property(output_path,'SubhaloPos',desired_redshift,list_all=False)
    SubhaloMass,output_redshift=get_subhalo_property(output_path,'SubhaloMass',desired_redshift,list_all=False)
    SubhaloMassType,output_redshift=get_subhalo_property(output_path,'SubhaloMassType',desired_redshift,list_all=False)
    
    
    SubhaloBHMass=SubhaloMassType[:,p_type]*1e10
    Subhalo_Indices=numpy.arange(len(SubhaloGrNr))
    mask=SubhaloBHMass>0
    #mask=SubhaloMass>=1e11
    Subhalo_Indices_cut=Subhalo_Indices[mask]
    SubhaloPos_cut=SubhaloPos[mask]
    SubhaloGrNr_cut=SubhaloGrNr[mask]

    def min_dis(median_position, position,box_size):
            """Choose the signed displacement with smallest absolute periodic distance.

            Compare the direct displacement with its two neighboring-box images.
            """
            pos_1=position-median_position
            pos_2=position-median_position+boxsize
            pos_3=position-median_position-boxsize

            new_position_options=numpy.array([pos_1,pos_2,pos_3])
            get_minimum_distance=numpy.argmin(numpy.abs(new_position_options))
            #print(new_position_options)

            #print(get_minimum_distance)
            return new_position_options[get_minimum_distance]


    def get_subhalo_id(blackhole_info):
        #print(blackhole_info)
        """Find the nearest eligible subhalo in the particle's parent FoF group.

        Return its index and squared periodic distance, or (-1, -1) for no candidates.
        """
        blackhole_group_id=blackhole_info[0]
        blackhole_position=blackhole_info[1]
        extract_ids_within_the_parent_FOF=blackhole_group_id==SubhaloGrNr_cut
        Subhalo_Indices_current_BH=Subhalo_Indices_cut[extract_ids_within_the_parent_FOF]
        #print(Subhalo_Indices_current_BH)
        SubhaloPos_current_BH=SubhaloPos_cut[extract_ids_within_the_parent_FOF]
        x_pos_Subhalo=SubhaloPos_current_BH[:,0]
        y_pos_Subhalo=SubhaloPos_current_BH[:,1]
        z_pos_Subhalo=SubhaloPos_current_BH[:,2]

        vectorized_min_dis = numpy.vectorize(min_dis)
        try:
            x_dis=vectorized_min_dis(blackhole_position[0],x_pos_Subhalo,boxsize)
            y_dis=vectorized_min_dis(blackhole_position[1],y_pos_Subhalo,boxsize)
            z_dis=vectorized_min_dis(blackhole_position[2],z_pos_Subhalo,boxsize)

            distance_sq=x_dis**2+y_dis**2+z_dis**2
            min_distance_sq=numpy.amin(distance_sq)
            subhalo_id=(Subhalo_Indices_current_BH[distance_sq==min_distance_sq])[0]
            return subhalo_id,min_distance_sq
        except ValueError:
            return -1,-1


    blackhole_info_space=list(zip(group_ids,BH_Pos))
    data=[get_subhalo_id(blackhole_info) for blackhole_info in blackhole_info_space]
    subhalo_ids=(numpy.array(data))[:,0]
    distance_from_subhalo_center=(numpy.array(data))[:,1]
    numpy.save(save_output_path+'subhalo_ids_%d.npy'%output_snapshot,subhalo_ids)
    numpy.save(save_output_path+'distance_from_subhalo_center_%d.npy'%output_snapshot,distance_from_subhalo_center)    
    return subhalo_ids,distance_from_subhalo_center


        
def mean_plot(x,y,xscl,yscl,nbins,manual_bins=False,M_BINS=numpy.arange(0,200)):
    #nbins = 5
    """Bin x and calculate the mean and standard deviation of y.

    Optionally log-transform each input; return centers, means, upper bounds, and lower bounds after filtering.
    """
    if(yscl==True):
        y=numpy.log10(y)
    if(xscl==True):
        x=numpy.log10(x)
    if (manual_bins==False):
        n, _ = numpy.histogram(x, bins=nbins)
        sy, _ = numpy.histogram(x, bins=nbins, weights=y)
        sy2, _ = numpy.histogram(x, bins=nbins, weights=y*y)
        
    else:
        n, _ = numpy.histogram(x, bins=M_BINS)
        sy, _ = numpy.histogram(x, bins=M_BINS, weights=y)
        sy2, _ = numpy.histogram(x, bins=M_BINS, weights=y*y)
    mean = sy / n
    std = numpy.sqrt(sy2/n - mean*mean)
    #std=1/numpy.sqrt(n)
    #plt.plot(x, y, 'bo')
    #plt.errorbar((_[1:] + _[:-1])/2, mean,std, color='blue', label = 'z = 8')
    #mean= savitzky_golay(mean, 11, 3)
    #print (_[1:] + _[:-1])/2
    #print mean
    mask=(std/mean)*100<100000.
    
    x=((_[1:] + _[:-1])/2)[mask]
    y=mean[mask]
    yul=y+std[mask]
    yll=y-std[mask]
    return x,y,yul,yll#,plt.errorbar(x,y,color=colour,linewidth=3,label=labl),plt.fill_between(x,yll, yul,color=colour,alpha=0.5)
 


def median_plot(x,y,xscl,yscl,nbins):
    #nbins = 5
    """Calculate median y values in evenly spaced x bins.

    Optionally log-transform each input and return bin centers and medians in those units.
    """
    if(yscl==True):
        y=numpy.log10(y)
    if(xscl==True):
        x=numpy.log10(x)
    x_space=numpy.linspace(numpy.amin(x),numpy.amax(x),nbins)
    y_space_med=[numpy.median(y[(x>x_space[i])&(x<x_space[i+1])]) for i in range(0,len(x_space)-1)]
    x_space_med=[(x_space[i]+x_space[i+1])/2 for i in range(0,len(x_space)-1)]
    return x_space_med,y_space_med


def mean_plot2(x,y,xscl,yscl,min_x, max_x,nbins):
    #nbins = 5
    """Calculate means and standard deviations within explicit x limits.

    Optionally log-transform each input; return bin centers, means, and standard deviations.
    """
    if(yscl==True):
        y=numpy.log10(y)
    if(xscl==True):
        x=numpy.log10(x)
    x_space=numpy.linspace(min_x,max_x,nbins)
    y_space_med=numpy.array([numpy.mean(y[(x>x_space[i])&(x<x_space[i+1])]) for i in range(0,len(x_space)-1)])
    y_space_std=numpy.array([numpy.std(y[(x>x_space[i])&(x<x_space[i+1])]) for i in range(0,len(x_space)-1)])
    x_space_med=numpy.array([(x_space[i]+x_space[i+1])/2 for i in range(0,len(x_space)-1)])
    return x_space_med,y_space_med,y_space_std

def get_median_with_IQR(x,y,xscl,yscl,minx,maxx,nbins,percentile):
    """Calculate binned medians and symmetric percentile widths.

    Optionally log-transform each input; return centers, medians, and percentile-width arrays.
    """
    if(yscl==True):
        y=numpy.log10(y)
    if(xscl==True):
        x=numpy.log10(x)
    x_space=numpy.linspace(minx,maxx,nbins)
    y_space_med=[numpy.median(y[(x>x_space[i])&(x<x_space[i+1])]) for i in range(0,len(x_space)-1)]
    y_space_IQR=[get_IQR(y[(x>x_space[i])&(x<x_space[i+1])],percentile) for i in range(0,len(x_space)-1)]
    x_space_med=[(x_space[i]+x_space[i+1])/2 for i in range(0,len(x_space)-1)]
    return numpy.array(x_space_med),numpy.array(y_space_med),numpy.array(y_space_IQR)

def get_IQR(dist,percentile):
    """Measure the difference between percentile and its complementary percentile.

    Use percentile=75 for the interquartile range; return zero for an empty distribution.
    """
    if (len(dist)>0):
        return numpy.percentile(dist, percentile) - numpy.percentile(dist, 100.-percentile)
    else:
        return 0
    
    
    
def luminosity_function(HM,box_size,log_HM_min,log_HM_max,Nbins):
        """Estimate a luminosity function per dex and per box volume.

        Return bin centers, number densities, and Poisson errors using box_size cubed as the volume.
        """
        def extract(HM_min,HM_max):
            """Summarize values strictly inside the supplied bin edges.

            Return the bin center and the count, weight, or fraction used by the enclosing estimator.
            """
            mask=(HM>HM_min)&(HM<HM_max)
            #print len(HM[mask])
            return (HM_min+HM_max)/2,len(HM[mask])
        
        HM_bin=numpy.logspace(log_HM_min,log_HM_max,Nbins,endpoint=True)
        out=[extract(HM_bin[i],HM_bin[i+1]) for i in range(0,len(HM_bin)-1)]
        centers=numpy.array(list(zip(*out))[0])
        counts=numpy.array(list(zip(*out))[1])
        #print counts
        dlogM=numpy.diff(numpy.log10(centers))[0]
        HMF=counts/dlogM/box_size**3
        dHMF=numpy.sqrt(counts)/dlogM/box_size**3
        return centers,HMF,dHMF        
    
def get_halo_density_profile(output_path,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,min_edge,max_edge,Nbins,CENTER_AROUND='POTENTIAL_MINIMUM',p_id=0):
    """Bin halo particle masses in logarithmic radial shells with periodic distances.

    Return log-radius centers, shell masses, and densities about the selected potential or BH center.
    """
    from kdcount import correlate
    def min_dis(median_position, position,box_size):
        """Choose the signed displacement with smallest absolute periodic distance.

        Compare the direct displacement with its two neighboring-box images.
        """
        pos_1=position-median_position
        pos_2=position-median_position+boxsize
        pos_3=position-median_position-boxsize
        new_position_options=numpy.array([pos_1,pos_2,pos_3])
        get_minimum_distance=numpy.argmin(numpy.abs(new_position_options))
        return new_position_options[get_minimum_distance]
    boxsize=get_box_size(output_path)
    particle_property='Coordinates'
    group_positions,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
    if (p_type==1):
        MassDM=load_snapshot_header(output_path,desired_redshift_of_selected_halo)['MassTable'][1]
        group_mass=numpy.array([1.]*len(group_positions))*MassDM
    else:
        particle_property='Masses'
        group_mass,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False) 
    particle_property='Potential'
    group_potential,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)

    if (p_type==4):
        particle_property='GFM_StellarFormationTime'       
        GFM_StellarFormationTime,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
        mask=GFM_StellarFormationTime>0
        group_potential=group_potential[mask]
        group_mass=group_mass[mask]
        group_positions=group_positions[mask]
    
    if (CENTER_AROUND=='MOST_MASSIVE_BLACKHOLE'):
        particle_property='ParticleIDs'        
        bh_IDs,output_redshift=get_particle_property_within_groups(output_path,particle_property,5,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)                
        particle_property='Coordinates'        
        bh_positions,output_redshift=get_particle_property_within_groups(output_path,particle_property,5,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)        
        particle_property='Masses'
        bh_masses,output_redshift=get_particle_property_within_groups(output_path,particle_property,5,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False) 
        print("Calculating density around BH with ID:",(bh_IDs[bh_masses==numpy.amax(bh_masses)])[0])
        center=(bh_positions[bh_masses==numpy.amax(bh_masses)])[0]        
    if (CENTER_AROUND=='POTENTIAL_MINIMUM'):
        particle_property='Coordinates'        
        dm_positions,output_redshift=get_particle_property_within_groups(output_path,particle_property,1,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)        
        particle_property='Potential' 
        dm_potential,output_redshift=get_particle_property_within_groups(output_path,particle_property,1,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)        
        center=(dm_positions[dm_potential==numpy.amin(dm_potential)])[0]
    transposed_group_positions=numpy.transpose(group_positions)
    vectorized_min_dis = numpy.vectorize(min_dis)
    x_dis=vectorized_min_dis(center[0],transposed_group_positions[0],boxsize)
    y_dis=vectorized_min_dis(center[1],transposed_group_positions[1],boxsize)
    z_dis=vectorized_min_dis(center[2],transposed_group_positions[2],boxsize)
    log_distances=numpy.log10(numpy.sqrt(x_dis**2+y_dis**2+z_dis**2))

    log_distance_bins=numpy.linspace(min_edge,max_edge,Nbins)
    binning=correlate.RBinning(log_distance_bins)
    bin_edges=binning.edges
    bin_centers=binning.centers
    mass_distribution=[]
    for i in range(0,len(bin_edges)-1):
        left=bin_edges[i]
        right=bin_edges[i+1]
        mask=(log_distances>left)&(log_distances<right)
        mass_inside_bin=numpy.sum(group_mass[mask])
        mass_distribution.append(mass_inside_bin)

    mass_distribution=numpy.array(mass_distribution)
    mass_volumes=4./3*3.14*numpy.diff((10**bin_edges)**3)
    mass_density=mass_distribution/mass_volumes
    #/4./3.14/(10**bin_centers)**2/((numpy.diff(bin_centers))[0])/numpy.log(10)
    return bin_centers,mass_distribution,mass_density


def get_general_profile(output_path,desired_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,min_edge,max_edge,Nbins,CENTER_AROUND='POTENTIAL_MINIMUM',p_id=0,OPERATION='AVERAGE'):
    """Aggregate a halo particle property in logarithmic radial shells.

    Return log-radius centers, edges, and mass-weighted averages or sums selected by OPERATION.
    """
    from kdcount import correlate
    def min_dis(median_position, position,box_size):
        """Choose the signed displacement with smallest absolute periodic distance.

        Compare the direct displacement with its two neighboring-box images.
        """
        pos_1=position-median_position
        pos_2=position-median_position+boxsize
        pos_3=position-median_position-boxsize
        new_position_options=numpy.array([pos_1,pos_2,pos_3])
        get_minimum_distance=numpy.argmin(numpy.abs(new_position_options))
        return new_position_options[get_minimum_distance]
    boxsize=get_box_size(output_path)
    particle_property='Coordinates'
    group_positions,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
    if (p_type==1):
        MassDM=load_snapshot_header(output_path,desired_redshift_of_selected_halo)['MassTable'][1]
        group_mass=numpy.array([1.]*len(group_positions))*MassDM
    else:
        particle_property='Masses'
        group_mass,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False) 

        particle_property=desired_property
        group_desired_property,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False) 
    particle_property='Potential'
    group_potential,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)

    if (p_type==4):
        particle_property='GFM_StellarFormationTime'       
        GFM_StellarFormationTime,output_redshift=get_particle_property_within_groups(output_path,particle_property,p_type,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)
        mask=GFM_StellarFormationTime>0
        group_potential=group_potential[mask]
        group_mass=group_mass[mask]
        group_positions=group_positions[mask]
    
    if (CENTER_AROUND=='MOST_MASSIVE_BLACKHOLE'):
        particle_property='ParticleIDs'        
        bh_IDs,output_redshift=get_particle_property_within_groups(output_path,particle_property,5,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)                
        particle_property='Coordinates'        
        bh_positions,output_redshift=get_particle_property_within_groups(output_path,particle_property,5,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)        
        particle_property='Masses'
        bh_masses,output_redshift=get_particle_property_within_groups(output_path,particle_property,5,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False) 
        print("Calculating density around BH with ID:",(bh_IDs[bh_masses==numpy.amax(bh_masses)])[0])
        center=(bh_positions[bh_masses==numpy.amax(bh_masses)])[0]        
    if (CENTER_AROUND=='POTENTIAL_MINIMUM'):
        particle_property='Coordinates'        
        dm_positions,output_redshift=get_particle_property_within_groups(output_path,particle_property,1,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)        
        particle_property='Potential' 
        dm_potential,output_redshift=get_particle_property_within_groups(output_path,particle_property,1,desired_redshift_of_selected_halo,index_of_selected_halo,group_type='groups',list_all=False)        
        center=(dm_positions[dm_potential==numpy.amin(dm_potential)])[0]
    transposed_group_positions=numpy.transpose(group_positions)
    vectorized_min_dis = numpy.vectorize(min_dis)
    x_dis=vectorized_min_dis(center[0],transposed_group_positions[0],boxsize)
    y_dis=vectorized_min_dis(center[1],transposed_group_positions[1],boxsize)
    z_dis=vectorized_min_dis(center[2],transposed_group_positions[2],boxsize)
    log_distances=numpy.log10(numpy.sqrt(x_dis**2+y_dis**2+z_dis**2))

    log_distance_bins=numpy.linspace(min_edge,max_edge,Nbins)
    binning=correlate.RBinning(log_distance_bins)
    bin_edges=binning.edges
    bin_centers=binning.centers
    mass_distribution=[]
    for i in range(0,len(bin_edges)-1):
        left=bin_edges[i]
        right=bin_edges[i+1]
        mask=(log_distances>left)&(log_distances<right)
        if (OPERATION=='AVERAGE'):
            mass_inside_bin=numpy.sum(group_desired_property[mask]*group_mass[mask])/numpy.sum(group_mass[mask])
        elif (OPERATION=='SUM'):
            mass_inside_bin=numpy.sum(group_desired_property[mask])
        mass_distribution.append(mass_inside_bin)

    mass_distribution=numpy.array(mass_distribution)
    return bin_centers,bin_edges,mass_distribution


                
        
def find_closest_BH(position,All_Coordinates,All_IDs,matching_range):
    """Expand a cube until it contains exactly one candidate BH.

    Return the cube side length and BH ID, or (-1, -1) when no unique match is found.
    """
    length_space=numpy.linspace(0,matching_range,matching_range)
    xpos,ypos,zpos=position
    found=0
    for cube_length in length_space:
        xcoordinates=All_Coordinates[:,0]
        ycoordinates=All_Coordinates[:,1]
        zcoordinates=All_Coordinates[:,2]

        maskx=(xcoordinates<=xpos+cube_length/2)&(xcoordinates>=xpos-cube_length/2)
        masky=(ycoordinates<=ypos+cube_length/2)&(ycoordinates>=ypos-cube_length/2)
        maskz=(zcoordinates<=zpos+cube_length/2)&(zcoordinates>=zpos-cube_length/2)

        mask=(maskx&masky)&maskz
        if (len(All_IDs[mask])==1):
            found=1
            break
    if found:
        return cube_length,All_IDs[mask][0]
    else:        #print(position)
        return -1,-1
    
def intersectnd(A,B):
    """Find common pairs of integer values in two arrays.

    Return unique shared pairs using string representations for the intersection.
    """
    A_string=[(str(a[0])+'_'+str(a[1]))for a in A]
    B_string =[(str(b[0])+'_'+str(b[1]))for b in B]
    return (numpy.array([AB.split('_') for AB in numpy.intersect1d(A_string,B_string)])).astype(int)

def match_the_blackholes(basePath1,basePath2, desired_redshift,matching_range):
    """Match BHs between simulations using mutually consistent spatial searches.

    Return mutual ID pairs, directional matches, and both original ID arrays.
    """
    p_type=5
    Coordinates1,output_redshift=get_particle_property(basePath1,'Coordinates',p_type,desired_redshift,list_all=False)
    ID1,output_redshift=get_particle_property(basePath1,'ParticleIDs',p_type,desired_redshift,list_all=False)    
    print("No of black holes",len(Coordinates1))
    
    Coordinates2,output_redshift=get_particle_property(basePath2,'Coordinates',p_type,desired_redshift,list_all=False)
    ID2,output_redshift=get_particle_property(basePath2,'ParticleIDs',p_type,desired_redshift,list_all=False)    
    print("No of black holes",len(Coordinates2))
    


    combined_data=numpy.transpose(numpy.array([find_closest_BH(position,Coordinates2,ID2,matching_range) for position in Coordinates1]))
    distances_1to2=combined_data[0]
    matchedIDs_1to2=combined_data[1].astype(int)
    combined_data=numpy.transpose(numpy.array([find_closest_BH(position,Coordinates1,ID1,matching_range) for position in Coordinates2]))
    distances_2to1=combined_data[0]
    matchedIDs_2to1=combined_data[1].astype(int)
    
    matched_ID_tuple_1to2=numpy.transpose(numpy.array([ID1,matchedIDs_1to2]))
    matched_ID_tuple_2to1=numpy.transpose(numpy.array([matchedIDs_2to1,ID2]))
    
    matched_pairs=intersectnd(matched_ID_tuple_1to2,matched_ID_tuple_2to1)
    print("no. of matched pairs:", len(matched_pairs)) 
          
    return matched_pairs,matchedIDs_1to2,matchedIDs_2to1,ID1,ID2     
        
    
def get_sublink_progenitors(basePath,subhalo_index,desired_redshift):    
    """Build a recursive subhalo progenitor tree from tree_extended.hdf5.

    Return a root node containing subhalo indices, snapshots, and two progenitor links.
    """
    class Subhalo:
        def __init__(self):
            """Initialize an empty tree node with sentinel values.

            Traversal fills in the object identity and progenitor links.
            """
            self.Index=-1
            self.MostMassiveProgenitor = -1
            self.NextMostMassiveProgenitor = -1
            self.Snap=-1 
    #-----------------------------------------------------------------------------------------------------------------------
    #----------------------------------This function fills up the progenitor tree------------------------------------------               
    def function_fill_progenitor_tree(subhalo_index,currentsubhalo,current_subhalo_ID):
        #print(current_subhalo_ID)
        """Populate the current node and recursively attach its progenitors.

        Update the tree in place using the event or tree arrays from the enclosing scope.
        """
        fetch_subhalo=current_subhalo_ID==SubhaloID_Tree
        currentsubhalo.Index=SubfindID_Tree[fetch_subhalo][0]
        currentsubhalo.Snap=SnapNum_Tree[fetch_subhalo][0]
        MostMassiveProgenitorID=FirstProgenitorID_Tree[fetch_subhalo][0]
        NextMostMassiveProgenitorID=NextProgenitorID_Tree[fetch_subhalo][0]

        MostMassiveProgenitorIndex=SubfindID_Tree[MostMassiveProgenitorID==SubhaloID_Tree]
        NextMostMassiveProgenitorIndex=SubfindID_Tree[NextMostMassiveProgenitorID==SubhaloID_Tree]


        currentsubhalo.MostMassiveProgenitor=Subhalo()
        currentsubhalo.NextMostMassiveProgenitor=Subhalo()


        if (MostMassiveProgenitorID!=-1):
            function_fill_progenitor_tree(MostMassiveProgenitorIndex,currentsubhalo.MostMassiveProgenitor,MostMassiveProgenitorID)
        if (NextMostMassiveProgenitorID!=-1):
            function_fill_progenitor_tree(NextMostMassiveProgenitorIndex,currentsubhalo.NextMostMassiveProgenitor,NextMostMassiveProgenitorID)    
#        print("Found no more progenitors: returning to previous recursion")
        return 

    tree=h5py.File(basePath+'/postprocessing/tree_extended.hdf5','r')
    save_output_path='/home/aklantbhowmick/Aklant/arepo_code_development/progenitor_outputs/'

    output_redshift,output_snapshot=desired_redshift_to_output_redshift(basePath,desired_redshift)
    print("The root subhalo is at the following redshift and snapshot:",output_redshift,output_snapshot)
    SubfindID=tree.get('SubfindID')[:]
    SubhaloID=tree.get('SubhaloID')[:]
    FirstProgenitorID=tree.get('FirstProgenitorID')[:]
    NextProgenitorID=tree.get('NextProgenitorID')[:]
    TreeID=tree.get('TreeID')[:]
    SnapNum=tree.get('SnapNum')[:]
    
    find_original_subhalo=(SubfindID==subhalo_index)&(SnapNum==output_snapshot)
    if (len(SubfindID[find_original_subhalo])!=1):
        print("Warning: The number of selected subhaloes must be 1")
    Target_TreeID=TreeID[find_original_subhalo]
    extract_tree=Target_TreeID==TreeID
    SubhaloID_Tree=SubhaloID[extract_tree]
    SnapNum_Tree=SnapNum[extract_tree]
    FirstProgenitorID_Tree=FirstProgenitorID[extract_tree]
    NextProgenitorID_Tree=NextProgenitorID[extract_tree]
    SubfindID_Tree=SubfindID[extract_tree]

    find_subhalo_on_tree=(SubfindID_Tree==subhalo_index)&(SnapNum_Tree==output_snapshot)
    RootSubhaloID=SubhaloID_Tree[find_subhalo_on_tree][0]

    sys.setrecursionlimit(1000)
        
    rootsubhalo = Subhalo()
    function_fill_progenitor_tree(subhalo_index,rootsubhalo,RootSubhaloID)
    return rootsubhalo

def get_sublink_descendants(basePath,subhalo_index,desired_redshift,TNG=0,path_to_TNG_trees='.'):  
    """Follow a subhalo descendant chain through an extended tree file.

    Return subhalo indices and snapshots; TNG selects the supplied external tree directory.
    """
    output_redshift,output_snapshot=desired_redshift_to_output_redshift(basePath,desired_redshift)
#	    save_output_path='/home/aklantbhowmick/Aklant/arepo_code_development/descendant_outputs/'
    if (TNG==0):
        tree=h5py.File(basePath+'/postprocessing/tree_extended.hdf5','r')
    if (TNG==1):
        tree=h5py.File(path_to_TNG_trees+'/tree_extended.hdf5','r')

    SubfindID=tree.get('SubfindID')[:]
    SubhaloID=tree.get('SubhaloID')[:]
    DescendantID=tree.get('DescendantID')[:]
    TreeID=tree.get('TreeID')[:]
    SnapNum=tree.get('SnapNum')[:]

    find_original_subhalo=(SubfindID==subhalo_index)&(SnapNum==output_snapshot)
    if (len(SubfindID[find_original_subhalo])!=1):
        print("Warning: The number of selected subhaloes must be 1")
    Target_TreeID=TreeID[find_original_subhalo]
    extract_tree=Target_TreeID==TreeID
    SubhaloID_Tree=SubhaloID[extract_tree]
    SnapNum_Tree=SnapNum[extract_tree]
    DescendantID_Tree=DescendantID[extract_tree]
    SubfindID_Tree=SubfindID[extract_tree]
    find_subhalo_on_tree=(SubfindID_Tree==subhalo_index)&(SnapNum_Tree==output_snapshot)

    SubfindID_descendants=[SubfindID[find_original_subhalo][0]]
    SnapNum_descendants=[SnapNum[find_original_subhalo][0]]
    
    MAX_ITERATIONS=100
    i=0
    Last_Snap_Num=200

    while (i<MAX_ITERATIONS):
        Subhalo_ID_tracking=DescendantID_Tree[find_subhalo_on_tree]
        if (Subhalo_ID_tracking[0]==[-1]):
            break
        fetch_subhalo=SubhaloID_Tree==Subhalo_ID_tracking 
        SubfindID_descendants.append(SubfindID_Tree[fetch_subhalo][0])
        SnapNum_descendants.append(SnapNum_Tree[fetch_subhalo][0])
        find_subhalo_on_tree=fetch_subhalo   
        #print(i)
        i+=1
    return SubfindID_descendants,SnapNum_descendants

def get_sublink_progenitors_most_massive_branch(basePath,root_subhalo_index,root_redshift):
    """Follow the first-progenitor links from a selected subhalo.

    Return subhalo-index and snapshot arrays for up to 100 nodes, including the root.
    """
    rootsubhalo=get_sublink_progenitors(basePath,root_subhalo_index,root_redshift)
    currentsubhalo=rootsubhalo
    Progenitor_SubhaloIndices=[]
    Progenitor_Snaps=[]
    i=0
    while (i<100):
        Index=currentsubhalo.Index
        Snap=currentsubhalo.Snap
        currentsubhalo=currentsubhalo.MostMassiveProgenitor
        if (Index==-1):
            break
        else:
            Progenitor_Snaps.append(Snap)
            Progenitor_SubhaloIndices.append(Index)
        i+=1
    Progenitor_Snaps=numpy.array(Progenitor_Snaps)
    Progenitor_SubhaloIndices=numpy.array(Progenitor_SubhaloIndices)
    return Progenitor_SubhaloIndices,Progenitor_Snaps


def trace_a_halo(basePath,halo_index_to_be_traced,initial_redshift,final_redshift):
    """Trace a halo through its first catalog subhalo to another redshift.

    Use descendants or the most-massive progenitor branch and return the destination FoF index.
    """
    SubhaloGrNr,o=get_subhalo_property(basePath,'SubhaloGrNr',initial_redshift)
    SubhaloMass,o=get_subhalo_property(basePath,'SubhaloMass',initial_redshift)
    SubhaloIndex=numpy.arange(0,len(SubhaloMass))
    subhalo_index_to_be_traced=SubhaloIndex[SubhaloGrNr==halo_index_to_be_traced][0]
    if (final_redshift<=initial_redshift):
        subhalo_index_tree,snap_tree=get_sublink_descendants(basePath,subhalo_index_to_be_traced,initial_redshift)
    else:
        subhalo_index_tree,snap_tree=get_sublink_progenitors_most_massive_branch(basePath,subhalo_index_to_be_traced,initial_redshift)


        
    SubhaloGrNr,o=get_subhalo_property(basePath,'SubhaloGrNr',final_redshift)
    snap_list,redshift_list=get_snapshot_redshift_correspondence(basePath)
    final_snap=snap_list[redshift_list==o][0]

    subhalo_index_tree=numpy.array(subhalo_index_tree)
    final_subhalo_index=subhalo_index_tree[snap_tree==final_snap][0]

    final_halo_index_to_be_traced=SubhaloGrNr[final_subhalo_index]
    return final_halo_index_to_be_traced


def convert_merger_events_to_hdf5(basePath, dont_save_in_basePath=0, save_output_path='.'):
    """Combine text merger logs into blackhole_mergers.hdf5.

    Write component IDs, masses, smoothing lengths, and event metadata, replacing the destination file.
    """
    output_file_names = os.listdir(basePath + 'blackhole_mergers/')
    snapshot_space = []
    redshift_space = []

    file_id_complete = numpy.array([], dtype=int)
    scale_fac_complete = numpy.array([])

    BH_id1_complete = numpy.array([], dtype=int)
    BH_mass1_complete = numpy.array([])
    BH_id2_complete = numpy.array([], dtype=int)
    BH_mass2_complete = numpy.array([])
    BH_Hsml1_complete = numpy.array([])
    BH_Hsml2_complete = numpy.array([])

    for name in output_file_names[:]:
        data = numpy.loadtxt(basePath + 'blackhole_mergers/' + name)

        try:
            if data.shape == (8,):
                file_id = numpy.array([data[0].astype(int)])
                scale_fac = numpy.array([data[1]])
                BH_id1 = numpy.array([data[2].astype(int)])
                BH_mass1 = numpy.array([data[3]])
                BH_id2 = numpy.array([data[4].astype(int)])
                BH_mass2 = numpy.array([data[5]])
                BH_Hsml1 = numpy.array([data[6]])
                BH_Hsml2 = numpy.array([data[7]])
            else:
                file_id = data[:, 0].astype(int)
                scale_fac = data[:, 1]
                BH_id1 = data[:, 2].astype(int)
                BH_mass1 = data[:, 3]
                BH_id2 = data[:, 4].astype(int)
                BH_mass2 = data[:, 5]
                BH_Hsml1 = data[:, 6]
                BH_Hsml2 = data[:, 7]

            file_id_complete = numpy.append(file_id_complete, file_id)
            scale_fac_complete = numpy.append(scale_fac_complete, scale_fac)
            BH_id1_complete = numpy.append(BH_id1_complete, BH_id1)
            BH_mass1_complete = numpy.append(BH_mass1_complete, BH_mass1)
            BH_id2_complete = numpy.append(BH_id2_complete, BH_id2)
            BH_mass2_complete = numpy.append(BH_mass2_complete, BH_mass2)
            BH_Hsml1_complete = numpy.append(BH_Hsml1_complete, BH_Hsml1)
            BH_Hsml2_complete = numpy.append(BH_Hsml2_complete, BH_Hsml2)
        except Exception as e:
            print(f"Error processing file {name}: {e}")

    if dont_save_in_basePath:
        hf = h5py.File(save_output_path + '/blackhole_mergers.hdf5', 'w')
    else:
        hf = h5py.File(basePath + '/blackhole_mergers.hdf5', 'w')
    hf.create_dataset('FileID', data=file_id_complete)
    hf.create_dataset('ScaleFactor', data=scale_fac_complete)
    hf.create_dataset('BH_ID1', data=BH_id1_complete)
    hf.create_dataset('BH_Mass1', data=BH_mass1_complete)
    hf.create_dataset('BH_ID2', data=BH_id2_complete)
    hf.create_dataset('BH_Mass2', data=BH_mass2_complete)
    hf.create_dataset('BH_Hsml1', data=BH_Hsml1_complete)
    hf.create_dataset('BH_Hsml2', data=BH_Hsml2_complete)
    hf.close()
    
    
def convert_merger_hosts_to_hdf5(basePath):
    """Combine merger-host text logs into blackhole_mergerhosts.hdf5.

    Write component BH and host masses with event metadata, replacing the destination file.
    """
    output_file_names=os.listdir(basePath+'/blackhole_mergerhosts/')
    snapshot_space=[]
    redshift_space=[]

    file_id_complete=numpy.array([],dtype=int)
    scale_fac_complete=numpy.array([])

    BH_id1_complete=numpy.array([],dtype=int)
    BH_mass1_complete=numpy.array([])
    BH_id2_complete=numpy.array([],dtype=int)
    BH_mass2_complete=numpy.array([])

    hosthalomass1_complete=numpy.array([])
    hosthalomass2_complete=numpy.array([])
    hosthalostellarmass1_complete=numpy.array([])
    hosthalostellarmass2_complete=numpy.array([])
    hosthalogasmass1_complete=numpy.array([])
    hosthalogasmass2_complete=numpy.array([])
    hosthalodmmass1_complete=numpy.array([])
    hosthalodmmass2_complete=numpy.array([])
    N_empty=0
    for name in output_file_names[:]:
        data=numpy.loadtxt(basePath+'/blackhole_mergerhosts/'+name)

        try:
            if (data.shape==(14,)):
                file_id=numpy.array([data[0].astype(int)])
                scale_fac=numpy.array([data[1]])
                BH_id1=numpy.array([data[2].astype(int)])
                BH_mass1=numpy.array([data[3]])
                hosthalomass1=numpy.array([data[4]])
                hosthalostellarmass1=numpy.array([data[5]])
                hosthalogasmass1=numpy.array([data[6]])
                hosthalodmmass1=numpy.array([data[7]])
                BH_id2=numpy.array([data[8].astype(int)])
                BH_mass2=numpy.array([data[9]])
                hosthalomass2=numpy.array([data[10]])
                hosthalostellarmass2=numpy.array([data[11]])
                hosthalogasmass2=numpy.array([data[12]])
                hosthalodmmass2=numpy.array([data[13]])
            else:
                file_id=data[:,0].astype(int)
                scale_fac=data[:,1]
                BH_id1=data[:,2].astype(int)
                BH_mass1=data[:,3]
                hosthalomass1=data[:,4]
                hosthalostellarmass1=data[:,5]
                hosthalogasmass1=data[:,6]
                hosthalodmmass1=data[:,7]
                BH_id2=data[:,8].astype(int)
                BH_mass2=data[:,9]
                hosthalomass2=data[:,10]
                hosthalostellarmass2=data[:,11]
                hosthalogasmass2=data[:,12]
                hosthalodmmass2=data[:,13]

            file_id_complete=numpy.append(file_id_complete,file_id)
            scale_fac_complete=numpy.append(scale_fac_complete,scale_fac)

            hosthalomass1_complete=numpy.append(hosthalomass1_complete,hosthalomass1)
            hosthalostellarmass1_complete=numpy.append(hosthalostellarmass1_complete,hosthalostellarmass1)    
            hosthalogasmass1_complete=numpy.append(hosthalogasmass1_complete,hosthalogasmass1)
            hosthalodmmass1_complete=numpy.append(hosthalodmmass1_complete,hosthalodmmass1)    

            hosthalomass2_complete=numpy.append(hosthalomass2_complete,hosthalomass2)
            hosthalostellarmass2_complete=numpy.append(hosthalostellarmass2_complete,hosthalostellarmass2)    
            hosthalogasmass2_complete=numpy.append(hosthalogasmass2_complete,hosthalogasmass2)
            hosthalodmmass2_complete=numpy.append(hosthalodmmass2_complete,hosthalodmmass2)    
 
            BH_id1_complete=numpy.append(BH_id1_complete,BH_id1)
            BH_mass1_complete=numpy.append(BH_mass1_complete,BH_mass1)
            BH_id2_complete=numpy.append(BH_id2_complete,BH_id2)
            BH_mass2_complete=numpy.append(BH_mass2_complete,BH_mass2)



        except IndexError:
            N_empty+=1
            aaa=1
            
            
    hf = h5py.File(basePath+'blackhole_mergerhosts.hdf5','w')
    hf.create_dataset('FileID',data=file_id_complete)
    hf.create_dataset('ScaleFactor',data=scale_fac_complete)
    
    hf.create_dataset('BH_ID1',data=BH_id1_complete)
    hf.create_dataset('BH_Mass1',data=BH_mass1_complete)
    hf.create_dataset('BH_ID2',data=BH_id2_complete)
    hf.create_dataset('BH_Mass2',data=BH_mass2_complete)
    
    hf.create_dataset('HostHaloMass1',data=hosthalomass1_complete)
    hf.create_dataset('HostHaloStellarMass1',data=hosthalostellarmass1_complete)
    hf.create_dataset('HostHaloGasMass1',data=hosthalogasmass1_complete)
    hf.create_dataset('HostHaloDMMass1',data=hosthalodmmass1_complete)
    
    hf.create_dataset('HostHaloMass2',data=hosthalomass2_complete)
    hf.create_dataset('HostHaloStellarMass2',data=hosthalostellarmass2_complete)
    hf.create_dataset('HostHaloGasMass2',data=hosthalogasmass2_complete)
    hf.create_dataset('HostHaloDMMass2',data=hosthalodmmass2_complete)   
    hf.close()
    
    
def convert_details_to_hdf5(basePath,DFD=0,KIN=0):
    """Combine BH detail logs into blackhole_details.hdf5.

    DFD and KIN select extended log layouts; write history and phase-space fields, replacing the file.
    """
    def parse_id_col(BH_ids_as_string):
        """Convert a prefixed BH identifier from a detail log to an integer.

        Strip the first three characters before conversion.
        """
        return numpy.int(BH_ids_as_string[3:])
    vec_parse_id_col=numpy.vectorize(parse_id_col)
    output_file_names=os.listdir(basePath+'blackhole_details/')
    BH_ids_for_id=numpy.array([],dtype=int)
    scale_factors_for_id=numpy.array([])
    BH_masses_for_id=numpy.array([])
    BH_mdots_for_id=numpy.array([])
    rhos_for_id=numpy.array([])
    sound_speeds_for_id=numpy.array([])
    if(KIN):
        xpos_for_id = numpy.array([])
        ypos_for_id = numpy.array([])
        zpos_for_id = numpy.array([])
        xvel_for_id = numpy.array([])
        yvel_for_id = numpy.array([])
        zvel_for_id = numpy.array([])

        xacc_for_id = numpy.array([])
        yacc_for_id = numpy.array([])
        zacc_for_id = numpy.array([])
        if(DFD):
            xDFDacc_for_id = numpy.array([])
            yDFDacc_for_id = numpy.array([])
            zDFDacc_for_id = numpy.array([])
        
    ii=0
    for output_file_name in output_file_names[:]:
        if ('blackhole_details' in output_file_name):
            print(ii)
            ii+=1
            try:
                full_data=numpy.loadtxt(basePath+'blackhole_details/'+output_file_name,dtype='str')
                BH_ids=vec_parse_id_col(full_data[:,0])
                scale_factors=(full_data[:,1]).astype('float')
                BH_masses=(full_data[:,2]).astype('float')
                BH_mdots=(full_data[:,3]).astype('float')
                rhos=(full_data[:,4]).astype('float')
                sound_speeds=(full_data[:,5]).astype('float')
                if(KIN):
                    xpos=(full_data[:,6]).astype('float')
                    ypos=(full_data[:,7]).astype('float')
                    zpos=(full_data[:,8]).astype('float')

                    xvel=(full_data[:,9]).astype('float')
                    yvel=(full_data[:,10]).astype('float')
                    zvel=(full_data[:,11]).astype('float')
                    
                    xacc=(full_data[:,12]).astype('float')
                    yacc=(full_data[:,13]).astype('float')
                    zacc=(full_data[:,14]).astype('float')
                    if(DFD):
                        xDFDacc=(full_data[:,15]).astype('float')
                        yDFDacc=(full_data[:,16]).astype('float')
                        zDFDacc=(full_data[:,17]).astype('float')

                BH_ids_for_id=numpy.append(BH_ids_for_id,BH_ids)
                scale_factors_for_id=numpy.append(scale_factors_for_id,scale_factors)
                BH_masses_for_id=numpy.append(BH_masses_for_id,BH_masses)
                BH_mdots_for_id=numpy.append(BH_mdots_for_id,BH_mdots)
                rhos_for_id=numpy.append(rhos_for_id,rhos)
                sound_speeds_for_id=numpy.append(sound_speeds_for_id,sound_speeds)
                if(KIN):
                    xpos_for_id = numpy.append(xpos_for_id,xpos)
                    ypos_for_id = numpy.append(ypos_for_id, ypos)
                    zpos_for_id = numpy.append(zpos_for_id, zpos)
                    xvel_for_id = numpy.append(xvel_for_id, xvel)
                    yvel_for_id = numpy.append(yvel_for_id, yvel)
                    zvel_for_id = numpy.append(zvel_for_id, zvel)
                    xacc_for_id = numpy.append(xacc_for_id, xacc)
                    yacc_for_id = numpy.append(yacc_for_id, yacc)
                    zacc_for_id = numpy.append(zacc_for_id, zacc)
                    if(DFD):
                        xDFDacc_for_id = numpy.append(xDFDacc_for_id, xDFDacc)
                        yDFDacc_for_id = numpy.append(yDFDacc_for_id, yDFDacc)
                        zDFDacc_for_id = numpy.append(zDFDacc_for_id, zDFDacc)
            except ValueError:
                aa=1            
    
    hf = h5py.File(basePath+'blackhole_details.hdf5','w')
    hf.create_dataset('BH_ID',data=BH_ids_for_id)
    hf.create_dataset('ScaleFactor',data=scale_factors_for_id)
    
    hf.create_dataset('BH_Mass',data=BH_masses_for_id)
    hf.create_dataset('BH_Mdot',data=BH_mdots_for_id)
    hf.create_dataset('Rho',data=rhos_for_id)
    hf.create_dataset('cs',data=sound_speeds_for_id)
    if(KIN):
       hf.create_dataset('xpos',data=xpos_for_id)
       hf.create_dataset('ypos',data=ypos_for_id)
       hf.create_dataset('zpos', data=zpos_for_id)
       hf.create_dataset('xvel', data=xvel_for_id)
       hf.create_dataset('yvel', data=yvel_for_id)
       hf.create_dataset('zvel', data=zvel_for_id)
       hf.create_dataset('xacc', data=xacc_for_id)
       hf.create_dataset('yacc', data=yacc_for_id)
       hf.create_dataset('zacc', data=zacc_for_id)
       if(DFD):
            hf.create_dataset('xDFDacc', data=xDFDacc_for_id)
            hf.create_dataset('yDFDacc', data=yDFDacc_for_id)
            hf.create_dataset('zDFDacc', data=zDFDacc_for_id)
            

    hf.close()
    
    
    
def convert_details_to_hdf5_previous(basePath,DFD=0,KIN=0):
    """Convert the earlier BH detail log layout to HDF5.

    Write history and phase-space arrays to blackhole_details.hdf5, replacing the destination file.
    """
    def parse_id_col(BH_ids_as_string):
        """Convert a prefixed BH identifier from a detail log to an integer.

        Strip the first three characters before conversion.
        """
        return numpy.int(BH_ids_as_string[3:])
    vec_parse_id_col=numpy.vectorize(parse_id_col)
    output_file_names=os.listdir(basePath+'blackhole_details/')
    BH_ids_for_id=numpy.array([],dtype=int)
    scale_factors_for_id=numpy.array([])
    BH_masses_for_id=numpy.array([])
    BH_mdots_for_id=numpy.array([])
    rhos_for_id=numpy.array([])
    sound_speeds_for_id=numpy.array([])
    if(KIN):
        xpos_for_id = numpy.array([])
        ypos_for_id = numpy.array([])
        zpos_for_id = numpy.array([])
        xvel_for_id = numpy.array([])
        yvel_for_id = numpy.array([])
        zvel_for_id = numpy.array([])


        
    ii=0
    for output_file_name in output_file_names[:]:
        if ('blackhole_details' in output_file_name):
            print(ii)
            ii+=1
            try:
                full_data=numpy.loadtxt(basePath+'blackhole_details/'+output_file_name,dtype='str')
                BH_ids=vec_parse_id_col(full_data[:,0])
                scale_factors=(full_data[:,1]).astype('float')
                BH_masses=(full_data[:,2]).astype('float')
                BH_mdots=(full_data[:,3]).astype('float')
                rhos=(full_data[:,4]).astype('float')
                sound_speeds=(full_data[:,5]).astype('float')
                if(KIN):
                    xpos=(full_data[:,6]).astype('float')
                    ypos=(full_data[:,7]).astype('float')
                    zpos=(full_data[:,8]).astype('float')

                    xvel=(full_data[:,9]).astype('float')
                    yvel=(full_data[:,10]).astype('float')
                    zvel=(full_data[:,11]).astype('float')
                    


                BH_ids_for_id=numpy.append(BH_ids_for_id,BH_ids)
                scale_factors_for_id=numpy.append(scale_factors_for_id,scale_factors)
                BH_masses_for_id=numpy.append(BH_masses_for_id,BH_masses)
                BH_mdots_for_id=numpy.append(BH_mdots_for_id,BH_mdots)
                rhos_for_id=numpy.append(rhos_for_id,rhos)
                sound_speeds_for_id=numpy.append(sound_speeds_for_id,sound_speeds)
                if(KIN):
                    xpos_for_id = numpy.append(xpos_for_id,xpos)
                    ypos_for_id = numpy.append(ypos_for_id, ypos)
                    zpos_for_id = numpy.append(zpos_for_id, zpos)
                    xvel_for_id = numpy.append(xvel_for_id, xvel)
                    yvel_for_id = numpy.append(yvel_for_id, yvel)
                    zvel_for_id = numpy.append(zvel_for_id, zvel)
            except ValueError:
                aa=1            
    
    hf = h5py.File(basePath+'blackhole_details.hdf5','w')
    hf.create_dataset('BH_ID',data=BH_ids_for_id)
    hf.create_dataset('ScaleFactor',data=scale_factors_for_id)
    
    hf.create_dataset('BH_Mass',data=BH_masses_for_id)
    hf.create_dataset('BH_Mdot',data=BH_mdots_for_id)
    hf.create_dataset('Rho',data=rhos_for_id)
    hf.create_dataset('cs',data=sound_speeds_for_id)
    if(KIN):
       hf.create_dataset('xpos',data=xpos_for_id)
       hf.create_dataset('ypos',data=ypos_for_id)
       hf.create_dataset('zpos', data=zpos_for_id)
       hf.create_dataset('xvel', data=xvel_for_id)
       hf.create_dataset('yvel', data=yvel_for_id)
       hf.create_dataset('zvel', data=zvel_for_id)
    hf.close()

def get_blackhole_progenitors(basePath,blackhole_index,desired_redshift,redshift_step): 
    """Build a recursive BH progenitor tree from snapshot merger records.

    Return the root node, merger count, and included event indices; update global traversal state.
    """
    global N_mergers
    global indices_of_included_events
    N_mergers=0
    indices_of_included_events=[-1]
    class Blackhole:
        def __init__(self):
            """Initialize an empty tree node with sentinel values.

            Traversal fills in the object identity and progenitor links.
            """
            self.BHID=-1
            self.BHMassAtLastMerger=-1.
            self.PrimaryProgenitor = -1
            self.SecondaryProgenitor = -1
            self.LastMergedAtRedshift=-1 
    #-----------------------------------------------------------------------------------------------------------------------
    #----------------------------------This function fills up the progenitor tree------------------------------------------               
    def function_fill_progenitor_tree(blackhole_ID,currentblackhole,minimum_redshift,redshift_step):
        """Populate the current node and recursively attach its progenitors.

        Update the tree in place using the event or tree arrays from the enclosing scope.
        """
        global N_mergers
        
        currentblackhole.BHID=blackhole_ID

        
        extract_last_merger_candidate=((primary_id==currentblackhole.BHID)|(secondary_id==currentblackhole.BHID))&(merging_redshifts>minimum_redshift)
 
        possible_merging_redshifts=merging_redshifts[extract_last_merger_candidate]
        possible_event_indices=event_indices[extract_last_merger_candidate]
        #print(minimum_redshift)
        possible_primary_id=primary_id[extract_last_merger_candidate]
        possible_secondary_id=secondary_id[extract_last_merger_candidate]
        possible_primary_mass=primary_mass[extract_last_merger_candidate]
        possible_secondary_mass=secondary_mass[extract_last_merger_candidate]     
        if (len(possible_merging_redshifts)==0):
            return
        remove_included_events=numpy.array([indices not in indices_of_included_events for indices in possible_event_indices])
        possible_merging_redshifts=possible_merging_redshifts[remove_included_events]
        possible_event_indices=possible_event_indices[remove_included_events]
        possible_primary_id=possible_primary_id[remove_included_events]
        possible_secondary_id=possible_secondary_id[remove_included_events]
        possible_primary_mass=possible_primary_mass[remove_included_events]
        possible_secondary_mass=possible_secondary_mass[remove_included_events]
        if (len(possible_merging_redshifts)==0):
            return  
        N_mergers+=1
        extract_most_recent_merger=possible_merging_redshifts==numpy.amin(possible_merging_redshifts)
        
        currentblackhole.LastMergedAtRedshift=possible_merging_redshifts[extract_most_recent_merger][0] 
        if (len(possible_merging_redshifts[extract_most_recent_merger])>1):
            print("Warning: Multiple mergers at given sync point")
            next_minimum_redshift=minimum_redshift
        else:
            next_minimum_redshift=currentblackhole.LastMergedAtRedshift
        indices_of_included_events.append(possible_event_indices[extract_most_recent_merger][0])

        primary_progenitor_ID=possible_primary_id[extract_most_recent_merger][0]
        secondary_progenitor_ID=possible_secondary_id[extract_most_recent_merger][0]

        if (primary_progenitor_ID==currentblackhole.BHID):
            currentblackhole.BHMassAtLastMerger=possible_primary_mass[extract_most_recent_merger][0]

        if (secondary_progenitor_ID==currentblackhole.BHID):
            currentblackhole.BHMassAtLastMerger=possible_secondary_mass[extract_most_recent_merger][0]

        currentblackhole.PrimaryProgenitor=Blackhole()
        currentblackhole.SecondaryProgenitor=Blackhole()

        function_fill_progenitor_tree(primary_progenitor_ID,currentblackhole.PrimaryProgenitor,next_minimum_redshift,redshift_step)
        function_fill_progenitor_tree(secondary_progenitor_ID,currentblackhole.SecondaryProgenitor,next_minimum_redshift,redshift_step)    


    p_type=5
    ParticleIDs,output_redshift=get_particle_property(basePath,'ParticleIDs',p_type,desired_redshift)
    BH_Mass,output_redshift=get_particle_property(basePath,'BH_Mass',p_type,desired_redshift)
    blackhole_ID=ParticleIDs[blackhole_index]   
    #merging_time,primary_mass,secondary_mass,primary_id,secondary_id,file_id_complete,N_empty=arepo_package.get_merger_events(basePath,HDF5=1,SORT_PRIMARY_SECONDARY=1)
    #merging_time,primary_id,primary_mass,hosthalomass_primary,hosthalostellarmass_primary,hosthalogasmass_primary,hosthalodmmass_primary,secondary_id,secondary_mass,hosthalomass_secondary,hosthalostellarmass_secondary,hosthalogasmass_secondary,hosthalodmmass_secondary,file_id_complete=get_merger_events_hosts(basePath,HDF5=1,SORT_PRIMARY_SECONDARY=0)
    
    
    merging_time,primary_mass,secondary_mass,primary_id,secondary_id,BH_Mdot1,BH_Mdot2,hosthalomass_primary,hosthalomass_secondary,hosthalostellarmass_primary,hosthalostellarmass_secondary,hosthalogasmass_primary,hosthalogasmass_secondary,BH_HostSFR1,BH_HostSFR2,file_id_complete,NNN=get_merger_events_from_snapshot(basePath,desired_redshift,HOSTS=1)
    #merging_time,secondary_id,secondary_mass,hosthalomass_primary,hosthalostellarmass_primary,hosthalogasmass_primary,hosthalodmmass_primary,primary_id,primary_mass,hosthalomass_secondary,hosthalostellarmass_secondary,hosthalogasmass_secondary,hosthalodmmass_secondary,file_id_complete=arepo_package.get_merger_events_hosts(basePath,HDF5=1)

   
    merging_redshifts=1./merging_time-1.
    event_indices=numpy.arange(0,len(merging_redshifts))
    #print(merging_redshifts)



    sys.setrecursionlimit(1000000)
        
    rootblackhole = Blackhole()
    function_fill_progenitor_tree(blackhole_ID,rootblackhole,output_redshift-redshift_step,redshift_step)
    return rootblackhole,N_mergers,indices_of_included_events





    

   


    

        
        
        
        
        
        







