# This script also reads in test_field coefficient data
# that we did NOT report in the paper because it was exponentially growing
# and thus unstable.

# July 29th, 2018
# Since some time series data can be a pain to read
# I am switching the ts read off and using the rms V,B
# from the xyaver.in file
# This is also more self-consistent

import numpy as np
import matplotlib.pyplot as plt
import pencilnew as pcn

#ts = pcn.read.ts()
xyave = pcn.read.aver(plane_list=['xy'])
param = pcn.read.param(param2=True)

# Read in the fields
bxm    = xyave.xy.bxmz
bym    = xyave.xy.bymz
emfx   = xyave.xy.Exmz
emfy   = xyave.xy.Eymz
jxm    = xyave.xy.jxmz
jym    = xyave.xy.jymz
bmean2 = bxm**2 + bym**2 # not saved in the npz file
b2tot  = xyave.xy.b2mz
u2tot  = xyave.xy.u2mz

# Read dimensions
tpts  = bmean2.shape[0]
nz    = bmean2.shape[1] 
nzby2 = int(nz/2)
tt    = xyave.t

# Define forcing wavemode
kf = 10.03

# Dynamo parameters
uave = np.sqrt(np.mean(u2tot[:-100]))#np.mean(ts.urms[:-100])
print('urms,urms^2 = B_eq^2:',uave,uave**2)

tres = 1/(kf**2 * param.eta)
print('t resistive:',tres)

Rm = uave/(kf * param.eta)
print('Rm based on urms and kf:',Rm)

# Save fields
np.savez('mfields.npz',tres=tres,Rm=Rm,uave=uave,kf=kf,tt=tt,bxm=bxm,bym=bym,b2tot=b2tot,u2tot=u2tot,emfx=emfx,emfy=emfy,jxm=jxm,jym=jym)
np.savez('tfields.npz',alp11z=xyave.xy.alp11z,alp12z=xyave.xy.alp12z,alp21z=xyave.xy.alp21z,alp22z=xyave.xy.alp22z,eta11z=xyave.xy.eta11z,eta12z=xyave.xy.eta12z,eta21z=xyave.xy.eta21z,eta22z=xyave.xy.eta22z)

# PLOT 1: Time history of urms,brms vs mean fields
plt.plot(tt/tres,np.sqrt(np.mean(bmean2[:,nzby2:],axis=1))/uave,label=r'$\bar{B}^N_{rms}/u_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(bmean2[:,:nzby2],axis=1))/uave,label=r'$\bar{B}^S_{rms}/u_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(b2tot,           axis=1))/uave,label=r'$b_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(u2tot,           axis=1))/uave,label=r'$u_{rms}$')
plt.xlabel('Time (resistive)')
plt.yscale('log')
plt.legend()
plt.savefig('tser.pdf',bbox_inches='tight',rasterized=True)
plt.close()

# PLOT 2: Time history of alpha_rms
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.alp11z**2,axis=1)),label=r'$\alpha^{11}_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.alp12z**2,axis=1)),label=r'$\alpha^{12}_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.alp21z**2,axis=1)),label=r'$\alpha^{21}_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.alp22z**2,axis=1)),label=r'$\alpha^{22}_{rms}$')
plt.xlabel('Time (resistive)')
plt.yscale('log')
plt.legend()
plt.savefig('alpha.pdf',bbox_inches='tight',rasterized=True)
plt.close()

# PLOT 3: Time history of eta_rms<
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.eta11z**2,axis=1)),label=r'$\eta^{11}_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.eta12z**2,axis=1)),label=r'$\eta^{12}_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.eta21z**2,axis=1)),label=r'$\eta^{21}_{rms}$')
plt.plot(tt/tres,np.sqrt(np.mean(xyave.xy.eta22z**2,axis=1)),label=r'$\eta^{22}_{rms}$')
plt.xlabel('Time (resistive)')
plt.yscale('log')
plt.legend()
plt.savefig('<eta.pdf',bbox_inches='tight',rasterized=True)
plt.close()

