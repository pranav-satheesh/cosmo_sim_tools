import multiprocessing
from multiprocessing import Pool
import glob
from cosmo_sim_tools import illustris as il
import numpy as np

def calc_stuff_isolated(path,snap):
        '''Calculate basic global properties of isolated galaxy simulations for given snapshot'''
        header = il.snapshot.loadHeader(path,snap)
        gasprop=il.snapshot.loadSubset(path,snap,0,['Coordinates','Masses','StarFormationRate'])
        totgasmass=np.sum(gasprop['Masses'])
        totstarmass=np.sum(il.snapshot.loadSubset(path,snap,4,'Masses'))
        initstarmass=np.sum(il.snapshot.loadSubset(path,snap,4,'GFM_InitialMass'))
        totsfr=np.sum(gasprop['StarFormationRate'])
        if totstarmass=={'count': 0}:
            totstarmass=0
            initstarmass=0

        try:
            bh=il.snapshot.loadSubset(path,snap,5,fields=['BH_Mdot','BH_MdotEddington','BH_Mass','BH_Density','BH_Hsml','BH_MdotBondi','Masses'])
            if bh['count'] != 0:
                mdotBondi=bh['BH_MdotBondi'][0]*10.22
                bhmdot=bh['BH_Mdot'][0]*10.22
                eddlim=bh['BH_MdotEddington'][0]*10.22
                bhmass=bh['BH_Mass'][0]
                bhtotmass=bh['Masses'][0]
                bhdensity=bh['BH_Density'][0]
                bhhsml=bh['BH_Hsml'][0]
                if eddlim!=0:
                    eddratio=bhmdot/eddlim
                else:
                    eddratio=0
            else:
                mdotBondi=-1
                bhmdot=-1
                eddlim=-1
                bhmass=-1
                bhdensity=-1
                bhhsml=-1
                eddratio=-1
                bhtotmass=-1
        except:
            mdotBondi=-1
            bhmdot=-1
            eddlim=-1
            bhmass=-1
            bhdensity=-1
            bhhsml=-1
            eddratio=-1
            bhtotmass=-1
            
        
            

        time=header.get('Time')
        gasfraction=totgasmass/(totgasmass+totstarmass)
        
        return [time,totgasmass,totstarmass,initstarmass,totsfr,gasfraction,bhmdot,eddratio,bhmass,bhdensity,bhhsml,mdotBondi,bhtotmass]


def isolated_gal(path):
    '''Calculate basic global properties of isolated galaxy simulations for every snapshot'''
    N=len(glob.glob1(path,"snapshot*.hdf5"))
    cpus = multiprocessing.cpu_count()
    p=Pool(cpus)
    args = [(path,snap) for snap in range(N)]
    result=np.array(p.starmap(calc_stuff_isolated,args))
    data = {}
    data['Time']=result[:,0]
    data['GasMass']=result[:,1]
    data['StellarMass']=result[:,2]
    data['InitialStellarMass']=result[:,3]
    data['StarFormationRate']=result[:,4]
    data['GasFraction']=result[:,5]
    data['BH_Mdot']=result[:,6]
    data['EddingtonRatio']=result[:,7]
    data['BH_Mass']=result[:,8]
    data['BH_Density']=result[:,9]
    data['BH_Hsml']=result[:,10]
    data['ModBHMdot']=0.715/10.22*data['BH_Mdot']/(data['BH_Mass']**2)*data['BH_Mass'][0]**2
    data['ModBHMass']=np.array([data['BH_Mass'][0]+np.trapz(y=data['ModBHMdot'][0:i+1],dx=0.005) for i in np.arange(len(data['ModBHMdot']))])
    data['MdotBondi']=result[:,11]
    data['BH_TotMass']=result[:,12]
    return data

def calc_stuff_merger(path,snap):
        '''Calculate basic global properties of galaxy merger simulations for given snapshot'''
        bhids=il.snapshot.loadSubset(path,0,5,fields=['ParticleIDs'])
        header = il.snapshot.loadHeader(path,snap)
        gasprop=il.snapshot.loadSubset(path,snap,0,['Coordinates','Masses','StarFormationRate'])
        totgasmass=np.sum(gasprop['Masses'])
        totstarmass=np.sum(il.snapshot.loadSubset(path,snap,4,'Masses'))
        initstarmass=np.sum(il.snapshot.loadSubset(path,snap,4,'GFM_InitialMass'))
        totsfr=np.sum(gasprop['StarFormationRate'])
        if totstarmass=={'count': 0}:
            totstarmass=0
            initstarmass=0
            
        try:
            
            bh=il.snapshot.loadSubset(path,snap,5,fields=['ParticleIDs','BH_Mdot','BH_MdotEddington','BH_Mass','BH_Density','Masses','Coordinates','BH_Hsml'])
            mask1=(bh['ParticleIDs']==bhids[0])
            mask2=(bh['ParticleIDs']==bhids[1])

            if len(bh['BH_Mdot'][mask1])==1:
                bhmdot1=bh['BH_Mdot'][mask1][0]*10.22
                eddlim1=bh['BH_MdotEddington'][mask1][0]*10.22
                bh1mass=bh['BH_Mass'][mask1][0]
                bh1density=bh['BH_Density'][mask1][0]
                bh1hsml=bh['BH_Hsml'][mask1][0]
                #bh1totmass=bh['Masses'][mask1][0]
                if eddlim1!=0:
                    eddratio1=bhmdot1/eddlim1
                else:
                    eddratio1=0
            else:
                bhmdot1=0
                bh1mass=0
                bh1density=0
                eddratio1=0
                bh1hsml=0
                # bh1totmass=0

            if len(bh['BH_Mdot'][mask2])==1:
                bhmdot2=bh['BH_Mdot'][mask2][0]*10.22
                eddlim2=bh['BH_MdotEddington'][mask2][0]*10.22
                bh2mass=bh['BH_Mass'][mask2][0]
                bh2density=bh['BH_Density'][mask2][0]
                bh2hsml=bh['BH_Hsml'][mask2][0]
                # bh2totmass=bh['Masses'][mask2][0]
                if eddlim2!=0:
                    eddratio2=bhmdot2/eddlim2
                else:
                    eddratio2=0
            else:
                bhmdot2=0
                bh2mass=0
                bh2density=0
                eddratio2=0
                bh2hsml=0
                # bh2totmass=0

            if len(bh['ParticleIDs'])==2:
                bhsep=bh['Coordinates'][0]-bh['Coordinates'][1]
                bhsep=np.sqrt(bhsep[0]**2 + bhsep[1]**2 + bhsep[2]**2)
            else:
                bhsep=0
        except:
            bhmdot1=bhmdot2=eddratio1=eddratio2=bh1mass=bh2mass=bh1density=bh2density=bhsep=bh1hsml=bh2hsml=0

        time=header.get('Time')
        gasfraction=totgasmass/(totgasmass+totstarmass)
        return [time,totgasmass,totstarmass,initstarmass,totsfr,gasfraction,bhmdot1,bhmdot2,eddratio1,eddratio2,bh1mass,bh2mass,bh1density,bh2density,bhsep,bh1hsml,bh2hsml]

def merging_gal(path):
    '''Calculate basic global properties of galaxy merger simulations for every snapshot'''
    N=len(glob.glob1(path,"snapshot*.hdf5"))    
    cpus = multiprocessing.cpu_count()
    p=Pool(cpus)
    args = [(path,snap) for snap in range(N)]
    result=np.array(p.starmap(calc_stuff_merger,args))
    data={}
    data['Time']=result[:,0]
    data['GasMass']=result[:,1]
    data['StellarMass']=result[:,2]
    data['InitialStellarMass']=result[:,3]
    data['StarFormationRate']=result[:,4]
    data['GasFraction']=result[:,5]
    data['BH_Mdot1']=result[:,6]
    data['BH_Mdot2']=result[:,7]
    data['EddingtonRatio1']=result[:,8]
    data['EddingtonRatio2']=result[:,9]
    data['BH_Mass1']=result[:,10]
    data['BH_Mass2']=result[:,11]
    data['BH_Density1']=result[:,12]
    data['BH_Density2']=result[:,13]
    data['BH_Separation']=result[:,14]
    data['BH_Hsml1']=result[:,15]
    data['BH_Hsml2']=result[:,16]
    return data

def get_particle_data(path,snap,ptypes,fields):
    result = {name:[] for name in fields}
    result['count']=0
    for pt in ptypes:
        if 'Masses' in fields:
            fields1=fields.copy()
            fields1.remove('Masses')
        else: fields1=fields
        if 'Temperature' in fields1:
            if ptypes != '0':
                raise Exception('Only gas has temperature!')
            fields1.remove('Temperature')
        particles = il.snapshot.loadSubset(path,snap,eval(pt),fields=fields1,sq=False)
        if particles['count'] > 0:
            result['count'] += particles['count']
            for name in fields1:
                result[name].extend(particles[name])
            if 'Masses' in fields:
                header = il.snapshot.loadHeader(path,snap)
                m=header.get('MassTable')[eval(pt)]
                if m == 0:
                    mm = il.snapshot.loadSubset(path,snap,eval(pt),fields=['Masses'])
                else:
                    mm=[m]*particles['count']
                result['Masses'].extend(mm)
            if 'Temperature' in fields:
                gas = il.snapshot.loadSubset(path,snap,eval(pt),fields=['InternalEnergy','ElectronAbundance'],sq=False)
                u=gas['InternalEnergy']
                xe=gas['ElectronAbundance']
                mp=1.672*10**(-24)
                mu=4*mp/(1+3*0.76+4*0.76*xe)
                kb=1.380*10**(-16)
                UnitEnergybyUnitMass=9.576*10**9
                Temp=u*(5.0/3.0-1)/kb*UnitEnergybyUnitMass*mu
                result['Temperature'].extend(Temp)
    for name in fields:
        result[name] = np.array(result[name])
    return result

 