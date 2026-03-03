import numpy as np
from cosmo_sim_tools import illustris as il
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from cosmo_sim_tools.arepo_tools.global_props import get_particle_data
from cosmo_sim_tools.extern import ptorrey_calc_hsml as calc_hsml

def galaxy2Dplots(path,snapnum,p_type,particle_property,view='xy',box_height=5,box_length=20,box_width=20,Nbins=1000,method='binning',
                  ngb=12,align=False,centre=None,fill=0,gauss=0,save_name=None,dpi=300,figure=None,axis=None,showBH=True,
                  colorbar=True,font_size=10,figuresize=None,scalelen='auto',smooth=True, plotlightpeak=False, vmin = None, vmax = None):
    '''
    path: path to the simulation output folder
    p_type: particle type.
    particle_property: can be Density, Temperature, Velocity, Pressure etc..
    view: can be xy, yx, zx etc..
    Nbins: resolution of image.
    method: binning will just put particles into bins and calculate avg/density. kernel will calculate weighted avg/density from nearest neighbors.
    fill: (only for method=binning) can be 0 or 1 -> used to fill empty grid points due to low density gas cells. 0= don't fill, 1= fill with with avg of nearest low density cells
    ngb: number of weighted neighbors if using kernel method.
    align: True will align the xy plane perpendicular to the angular momentum of baryonic matter within 5kpc/h of the center.
    gauss: use gaussian filter.
    scalelen: length of scale bar.
    axis: will add the plot to the provided axis.
    '''
    
    header = il.snapshot.loadHeader(path,snapnum)
    boxsize=header.get('BoxSize')
    if particle_property == 'Density':
        load_fields = ['Masses','Coordinates']
    elif particle_property == 'Temperature':
        if p_type != '0':
            print('Temperature only for gas. Set p_type to \'0\'')
            return
        load_fields=['Masses','Coordinates','InternalEnergy','ElectronAbundance']
    else: load_fields = ['Masses','Coordinates', particle_property]
    if p_type == '0':
        load_fields += ['Density']
        
    particle_data=get_particle_data(path,snapnum,p_type,load_fields)

    bhs=il.snapshot.loadSubset(path,snapnum,5,fields=['Coordinates','ParticleIDs'])
    
    if centre is None:
        cent=np.array([boxsize*0.5,boxsize*0.5,boxsize*0.5])
    else:
        cent=centre
        
    if bhs == {'count': 0}:
        bh_pos=np.array([[0,0,0]])
    else:
        bh_pos = bhs['Coordinates']- cent
    
    if align==False:
        if len(particle_data['Coordinates']) < 2:
            xpos=np.array([])
            ypos=np.array([])
            zpos=np.array([])
        else:
            xpos=particle_data['Coordinates'][:,0]-cent[0]
            ypos=particle_data['Coordinates'][:,1]-cent[1]
            zpos=particle_data['Coordinates'][:,2]-cent[2]
        bhx1 = bh_pos[:,0]
        bhy1 = bh_pos[:,1]
        bhz1 = bh_pos[:,2]
    else:
        Ldata = get_particle_data(path,snapnum,'02345',['Coordinates','Masses','Velocities'])
        pos = Ldata['Coordinates'] - cent
        r=np.linalg.norm(pos,axis=1)
        pos=pos[r<5]
        masses=Ldata['Masses'][r<5]
        vel=Ldata['Velocities'][r<5]
        totL=np.array([np.sum(masses*(pos[:,1]*vel[:,2]-pos[:,2]*vel[:,1])),
                       np.sum(masses*(pos[:,2]*vel[:,0]-pos[:,0]*vel[:,2])),
                       np.sum(masses*(pos[:,0]*vel[:,1]-pos[:,1]*vel[:,0]))])
        zaxis = totL/np.linalg.norm(totL)



        ct=zaxis[2]/np.sqrt(zaxis[0]**2+zaxis[1]**2+zaxis[2]**2)
        st=np.sqrt(1-ct**2)
        cp=zaxis[0]/np.sqrt(zaxis[0]**2+zaxis[1]**2)
        sp=zaxis[1]/np.sqrt(zaxis[0]**2+zaxis[1]**2)
        if len(particle_data['Coordinates']) < 2:
            x1=np.array([])
            y1=np.array([])
            z1=np.array([])
        else:
            x1=particle_data['Coordinates'][:,0]-cent[0]
            y1=particle_data['Coordinates'][:,1]-cent[1]
            z1=particle_data['Coordinates'][:,2]-cent[2]

        bhx=bh_pos[:,0]
        bhy=bh_pos[:,1]
        bhz=bh_pos[:,2]

        xpos=x1*ct*cp+y1*sp*ct-st*z1
        ypos=-x1*sp+y1*cp
        zpos=x1*st*cp+y1*st*sp+z1*ct

        bhx1=bhx*ct*cp+bhy*sp*ct-st*bhz
        bhy1=-bhx*sp+bhy*cp
        bhz1=bhx*st*cp+bhy*st*sp+bhz*ct
        
        # if showCOM == True:
        #     com=tf.getCOM(path,snapnum)-cent
        #     comx=com[0]comy=com[1]comz=com[2]
        #     comx1=comx*ct*cp+comy*sp*ct-st*comz
        #     comy1=-comx*sp+comy*cp
        #     comz1=comx*st*cp+comy*st*sp+comz*ct

    if (view=='xy'):
        axis1=xpos; axis2=ypos; axis3=zpos; bhaxis1=bhx1; bhaxis2=bhy1
    if (view=='yz'):
        axis1=ypos; axis2=zpos; axis3=xpos; bhaxis1=bhy1; bhaxis2=bhz1
    if (view=='xz'):
        axis1=xpos; axis2=zpos; axis3=ypos; bhaxis1=bhx1; bhaxis2=bhz1
    if (view=='yx'):
        axis1=ypos; axis2=xpos; axis3=zpos; bhaxis1=bhy1; bhaxis2=bhx1
    if (view=='zy'):
        axis1=zpos; axis2=ypos; axis3=xpos; bhaxis1=bhz1; bhaxis2=bhy1
    if (view=='zx'):
        axis1=zpos; axis2=xpos; axis3=ypos; bhaxis1=bhz1; bhaxis2=bhx1
        
    
    mask1=(axis3 > -box_height/2.0) & (axis3 < box_height/2.0) & (axis1>0.5*(-box_length))&(axis1<0.5*(box_length))&(axis2>0.5*(-box_width))&(axis2<0.5*(box_width))
    
    axis1=axis1[mask1]
    axis2=axis2[mask1]
    if method == 'sph':
        axis3=axis3[mask1]
    Num=len(axis1)

    ax1=np.linspace(0.5*(-box_length),0.5*(box_length),Nbins)
    h=np.diff(ax1)[0]
    ax2=np.arange(0.5*(-box_width),0.5*(box_width)+0.9*h,h)
    if method == 'sph':
        ax3=np.arange(0.5*(-box_height),0.5*(box_height)+0.9*h,h)
    lx1=len(ax1)
    lx2=len(ax2)
       

    First,Second=np.meshgrid(ax1,ax2)
    
    if particle_property == 'Density':
        prop = particle_data['Masses'][mask1]
    elif particle_property == 'Temperature':
        u=particle_data['InternalEnergy'][mask1]
        xe=particle_data['ElectronAbundance'][mask1]
        mp=1.672*10**(-24)
        mu=4*mp/(1+3*0.76+4*0.76*xe)
        kb=1.380*10**(-16)
        UnitEnergybyUnitMass=9.576*10**9
        Temp=u*(5.0/3.0-1)/kb*UnitEnergybyUnitMass*mu
        prop=Temp
    elif particle_property == 'Velocities':
        prop = np.linalg.norm(particle_data['Velocities'][mask1],axis=1)
    else:
        prop = particle_data[particle_property][mask1]

    
    if method == 'sph':
        x, y, z = np.meshgrid(ax1, ax2, ax3, indexing='ij')
        # coord = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        if particle_property == 'Density':
            quant = calc_hsml.get_gas_density_around_stars( axis1, axis2, axis3, prop, x.flatten(), y.flatten(), z.flatten(), DesNgb=ngb)
            quant *= box_height
        else:
            particle_masses = particle_data['Masses'][mask1]
            quant=calc_hsml.get_gas_temperature_around_stars( axis1, axis2, axis3, particle_masses, [0]*len(prop), prop, x.flatten(), y.flatten(), z.flatten(), DesNgb=ngb)
            rho = calc_hsml.get_gas_density_around_stars( axis1, axis2, axis3, particle_masses, x.flatten(), y.flatten(), z.flatten(), DesNgb=ngb)
        numz = len(ax3)
        i=0
        proj_property=[]
        while i+numz <= len(quant):
            if particle_property == 'Density':
                proj_property.append(np.nanmean(quant[i:i+numz]))
            else:
                proj_property.append(np.nansum(quant[i:i+numz] * rho[i:i+numz])/np.nansum(rho[i:i+numz]))
            i+=numz
        proj_property=np.array(proj_property)
        proj_property = proj_property.reshape((lx1,lx2))
        
    elif method == 'binning':
        proj_property=np.zeros((lx1,lx2))
        proj_freq=np.zeros((lx1,lx2))
        if p_type == '0':
            densities = particle_data['Density'][mask1]
        if p_type == '0' and particle_property == 'Density':           
            for idx in np.arange(Num):
                cellsize = min((prop[idx]/densities[idx])**(1/3), box_height)
                input_mass = min(prop[idx],densities[idx]*h*h*cellsize)
                proj_property[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)] += input_mass
                proj_freq[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)]+=1
        else:     
            for idx in np.arange(Num):
                proj_property[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)]+=prop[idx]
                proj_freq[int((axis1[idx] - 0.5*(-box_length))/h),int((axis2[idx] - 0.5*(-box_width))/h)]+=1
            
        if fill==1:
            if '0' not in p_type:
                print('fill is only for p_type=0')
                return
            size1=(particle_data['Masses'][mask1]/densities)/(h**3)
            lowdensmask=(size1>7)
            size=size1[lowdensmask]
            axis1_low=axis1[lowdensmask]
            axis2_low=axis2[lowdensmask]
            prop1=prop[lowdensmask]
            density_low = densities[lowdensmask]
            # arr=[x for _,x in sorted(zip( size, np.arange(len(prop1)) ))] 
            arr = np.argsort(size)
            proj_freq_temp=proj_freq*100
            for idx in arr:
                rad=int(2*0.5*size[idx]**(1/3))
                xi=int((axis1_low[idx] - 0.5*(-box_length))/h)
                yi=int((axis2_low[idx] - 0.5*(-box_width))/h)
                for xx in np.arange( max(0,xi-rad) , min(lx1,xi+rad)):
                    for yy in np.arange( max(0,yi-int(np.sqrt(rad**2-(xx-xi)**2))) , min(lx2,yi+int(np.sqrt(rad**2-(xx-xi)**2)))):
                        if proj_freq_temp[xx,yy]<5:
                            if particle_property == 'Density':
                                proj_property[xx,yy] += density_low[idx] * h**3 * np.sqrt(rad**2-(xx-xi)**2-(yy-yi)**2)
                            else:
                                proj_property[xx,yy] += prop1[idx]
                            proj_freq[xx,yy]+=1
                            proj_freq_temp[xx,yy]+=1
                            
            if particle_property=='Density':
                proj_property[proj_freq==0]=1e-100
                proj_freq[proj_freq==0]=1
            # if particle_property=='Pressure':
            #     proj_property[proj_freq==0]=1e-100
            #     proj_freq[proj_freq==0]=1
            # if particle_property== 'Velocity' or 'Temperature':
            else:
                for i in range(lx1):
                    for j in range(lx2):
                        if proj_freq[i,j]==0:
                            arr=proj_property[max(0,i-int(0.05/h)) : min(lx1,i+int(0.05/h)), max(0,j-int(0.05/h)) : min(lx2,j+int(0.05/h))]
                            proj_property[i,j] = arr[np.nonzero(arr)].mean()
                            proj_freq[i,j]=1
        
        if (particle_property == 'Density'):
            proj_property /= (h*h)
            proj_property[proj_freq==0]=1e-100
        elif particle_property == ('StarFormationRate'):
            proj_property /= (h*h*1e6/0.7/0.7)
        else:
            proj_property /= proj_freq
        if smooth == True:
            proj_property = scipy.ndimage.gaussian_filter(proj_property, sigma=0.5)
        
    proj_property=np.transpose(proj_property)
    if particle_property=='Density':
        proj_property=proj_property*0.7*1e4
    if method == 'binning' and gauss !=0:
        proj_property=gaussian_filter(proj_property,sigma=gauss)
    partname=['Gas','DM',' ','Tracers','Stars','BH']
    if axis == None:
        fig, ax0 = plt.subplots(dpi=dpi)
        if figuresize is not None:
            fig.set_size_inches(figuresize[0],figuresize[1])
        if colorbar == True:
            axins=inset_axes(ax0,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax0.transAxes,
                       borderpad=0,
                       )
    else:
        ax0=axis
    
    if (particle_property=='Density'):
        # im=ax0.pcolor(First,Second,proj_property,norm=mcolors.LogNorm(vmin=1e-1,vmax=1e3),cmap='inferno',rasterized=True,shading='auto')
        # im=ax0.imshow(proj_property,norm=mcolors.LogNorm(vmin=1e-1,vmax=1e3),cmap='inferno',rasterized=True)
        # im=ax0.pcolormesh(First,Second,np.log10(proj_property),vmin=-4,vmax=2.5,cmap='inferno',rasterized=True,shading='gouraud')
        # im=ax0.pcolormesh(First,Second,np.log10(proj_property),vmin=-4,vmax=2.5+1,cmap='inferno',rasterized=True)
        if (vmin == None) & (vmax == None):
            # vmin = -3; vmax = 3.5 
            vmin = -1; vmax = 3
        im=ax0.pcolormesh(First,Second,np.log10(proj_property),vmin=vmin,vmax=vmax,cmap='coolwarm',rasterized=True)

        if (axis==None and colorbar==True):
            cbar=fig.colorbar(im,cax=axins)
            cbar.ax.tick_params(axis='y', direction='out')
            # ax0.set_aspect('equal')
            cbar.set_label(r'$\mathrm{Log(Density [M_{\odot}pc^{-2}])}$')
            # ax0.set_title('{} density, t = {}'.format(partname[p_type],snapnum/200))
        ax0.text(0.01, 0.03, '{:.3f}Gyr'.format(header.get('Time')/0.7), transform=ax0.transAxes, fontsize=font_size,color='white')
       
    elif (particle_property=='Temperature'):
        if (vmin == None) & (vmax == None):
            vmin = 3; vmax = 7
        # im=ax0.pcolor(First,Second,proj_property,norm=mcolors.LogNorm(vmin=1e3,vmax=1e7),rasterized=True,shading='auto')
        im=ax0.pcolormesh(First,Second,np.log10(proj_property),vmin=vmin,vmax=vmax,rasterized=True,shading='gouraud')
        ax0.set_aspect('equal')
        if (axis==None and colorbar==True):
            cbar=fig.colorbar(im,cax=axins)
            cbar.ax.tick_params(axis='y', direction='out')
            cbar.set_label(r'Log(Temperature [$K$])')
        ax0.text(0.01, 0.03, '{:.3f}Gyr'.format(header.get('Time')/0.7), transform=ax0.transAxes, fontsize=font_size,color='white')
        
    elif (particle_property=='Velocities'):
        if (vmin == None) & (vmax == None):
            vmin = 2; vmax = 3.6
        # im=ax0.pcolor(First,Second,proj_property,vmin=0,vmax=1000,cmap='hot',rasterized=True,shading='auto')
        # im=ax0.pcolormesh(First,Second,proj_property,vmin=0,vmax=1000,cmap='hot',rasterized=True,shading='gouraud')
        im=ax0.pcolormesh(First,Second,np.log10(proj_property),vmin=2,vmax=3.6,cmap='hot',rasterized=True,shading='gouraud')
        ax0.set_aspect('equal')
        if (axis==None and colorbar==True):
            cbar=fig.colorbar(im,cax=axins)
            cbar.ax.tick_params(axis='y', direction='out')
            cbar.set_label(r'Log(Speed [$km/s$])')
        ax0.text(0.01, 0.03, '{:.3f}Gyr'.format(header.get('Time')/0.7), transform=ax0.transAxes, fontsize=font_size,color='white')
        
    elif (particle_property=='Pressure'):
        if (vmin == None) & (vmax == None):
            vmin = -15; vmax = -9
        # im=ax0.pcolor(First,Second,proj_property,norm=mcolors.LogNorm(vmin=1e-4,vmax=1e2),cmap='coolwarm',rasterized=True,shading='auto')
        im=ax0.pcolormesh(First,Second,np.log10(proj_property*7e-12),vmin=-15,vmax=-9,cmap='coolwarm',rasterized=True,shading='gouraud')
        ax0.set_aspect('equal')
        if (axis==None and colorbar==True):
            cbar=fig.colorbar(im,cax=axins)
            cbar.ax.tick_params(axis='y', direction='out')
            cbar.set_label(r'Log(Pressure [$M_\odot/(pc\,yr^2)$])')
        ax0.text(0.01, 0.03, '{:.3f}Gyr'.format(header.get('Time')/0.7), transform=ax0.transAxes, fontsize=font_size,color='black')

    elif (particle_property=='StarFormationRate'):     
        if (vmin == None) & (vmax == None):
            im=ax0.pcolormesh(First,Second,np.log10(proj_property),rasterized=True,cmap='ocean')
        else:
            im=ax0.pcolormesh(First,Second,np.log10(proj_property),vmin=vmin,vmax=vmax,rasterized=True,cmap='ocean')
        if (axis==None and colorbar==True):
            cbar=fig.colorbar(im,cax=axins)
            cbar.ax.tick_params(axis='y', direction='out')
            cbar.set_label(rf'$\mathrm{{Log(SFR surf density \;[M_\odot/yr/pc^2])}}$')
        ax0.text(0.01, 0.03, '{:.3f}Gyr'.format(header.get('Time')/0.7), transform=ax0.transAxes, fontsize=font_size,color='white')    
    
        
    fontprops = fm.FontProperties(size=font_size)
    if scalelen == 'auto':
        scalelen=int(box_length/4/0.7)*0.7
        if scalelen==0:
            scalelen=int(10*box_length/4/0.7)*0.07
        if (scalelen>=0.7) & (scalelen < 7):
            scalebar = AnchoredSizeBar(ax0.transData,
                            scalelen, '{} kpc'.format(int(scalelen/0.7)), 'upper left', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=0.01,
                            fontproperties=fontprops)
        elif scalelen >= 7:
            scalelen = int(scalelen/7)*7
            scalebar = AnchoredSizeBar(ax0.transData,
                            scalelen, '{} kpc'.format(int(scalelen/0.7)), 'upper left', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=0.01,
                            fontproperties=fontprops)
        else:
            scalebar = AnchoredSizeBar(ax0.transData,
                            scalelen, '{} pc'.format(int(scalelen/0.7*1000)), 'upper left', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=0.002,
                            fontproperties=fontprops)
        ax0.add_artist(scalebar)
    elif scalelen != -1:
        scalebar = AnchoredSizeBar(ax0.transData,
                            scalelen, '{} kpc'.format(int(scalelen/0.7)), 'upper left', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=0.01,
                            fontproperties=fontprops)
        ax0.add_artist(scalebar)
    
    bhmask=(bhaxis1>0.5*(-box_length))&(bhaxis1<0.5*(box_length))&(bhaxis2>0.5*(-box_width))&(bhaxis2<0.5*(box_width))
    bhaxis1=bhaxis1[bhmask]
    bhaxis2=bhaxis2[bhmask]
    if showBH==True:
        if particle_property=='Temperature':
            ax0.scatter(bhaxis1, bhaxis2,s=5,color='red')
        else:   
            ax0.scatter(bhaxis1, bhaxis2,s=5,color='springgreen')
        #to plot surface density maximum
    if plotlightpeak == True:
        sqlen = 0.7*2
        Lmaxid = [0,0]; Lmax=0; Ncells = int(0.5*0.7/h/2) 
        Ncells = 2
        for i in range(int(sqlen/h)):
            for j in range(int(sqlen/h)):
                Lavg = np.average(proj_property[int((bhaxis2[0] + 0.5*box_width -sqlen/2)/h) + i - Ncells:int((bhaxis2[0] + 0.5*box_width -sqlen/2)/h) + i + Ncells +1 , int((bhaxis1[0] + 0.5*box_length -sqlen/2)/h) + j - Ncells:int((bhaxis1[0] + 0.5*box_length -sqlen/2)/h) + j + Ncells +1 ])
                if  Lavg > Lmax:
                    Lmaxid = [int((bhaxis2[0] + 0.5*box_width -sqlen/2)/h) + i, int((bhaxis1[0] + 0.5*box_length -sqlen/2)/h) + j]
                    Lmax = Lavg
        # ax0.scatter( [bhaxis1[0] + Lmaxid[1]*h - sqlen/2], [bhaxis2[0] + Lmaxid[0]*h - sqlen/2], color='black',marker='^',s=4)
        ax0.scatter( [Lmaxid[1]*h + 0*h/2 - box_length/2], [Lmaxid[0]*h + 0*h/2- box_width/2], color='black',marker='^',s=4)
        # print(f'Lmax=({bhaxis1[0] + Lmaxid[0]*h - sqlen/2}|{bhaxis2[0] + Lmaxid[1]*h - sqlen/2})')
           
            
 
    #To plot bh_hsml radius     
    #bh_hsml=il.snapshot.loadSubset(path,snapnum,5,fields='BH_Hsml')
    #circle=plt.Circle((bh_pos[0][0], bh_pos[0][1]), bh_hsml, color='white',fill=False)
    #ax0.add_patch(circle)
    
    ax0.set_xticks([])
    ax0.set_yticks([])
    if figuresize is None:
        ax0.set_aspect('equal')
    
    if (save_name==None):
#         plt.show()
        return im
        # return proj_property, proj_freq
    else:
        plt.savefig(save_name,bbox_inches='tight',dpi=dpi)
        plt.close()
        return
