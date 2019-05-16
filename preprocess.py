# May 9th, 2019
# Normalize EMF,J and B with B_eq
# Introduced read_mf_norm that replaces read_mf everywhere

# Feb. 24th, 2019
# EMF -> -EMF
# EMF is really E in pc: vxb

# Several changes/additions 
# Feb. 19th, 2019
# Has support for train-test sequential splits instead of random sklearn splits
# PCA, Scaling back and forth, Log(B^2) for temporal data (B(t) that is)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # for directory exist checks
# https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python

from matplotlib.colors import ListedColormap

def read_mf_norm(fname='mfields.npz'):
  # PENCIL data
  # Normalize everything by urms = B_eq
  mf = np.load(fname)
  bxm = mf['bxm']/mf['uave']
  bym = mf['bym']/mf['uave']
  jxm = mf['jxm']/mf['uave']
  jym = mf['jym']/mf['uave']
  Exm = mf['emfx']/mf['uave']
  Eym = mf['emfy']/mf['uave']
  return bxm, bym, jxm, jym, Exm, Eym

def read_mf(fname='mfields.npz'):
  # PENCIL data
  mf = np.load(fname)
  return mf['bxm'], mf['bym'], mf['jxm'], mf['jym'], mf['emfx'], mf['emfy']

def ave_t(arr,tone=1000,ttwo=2000,verbose=None):
  if verbose:
    print(f't1: {tone}, t2: {ttwo}')
  return np.mean(arr[tone:ttwo,:],axis=0)

def ave_z(arr,zone=128,ztwo=-1,verbose=None):
  if verbose:
    print(f'z1: {zone}, z2: {ztwo}')
  return np.mean(arr[:,zone:ztwo],axis=1)

def gen_df_tave(fname='mfields.npz',t1=1000,t2=2000,verbose=None):
  '''
  name_conv: True refers to pencil code data: bxm,bym,jxm,jym,emfx,emfy
  '''

  if verbose:
    print(f"Generating time averaged dataframe with t1: {t1} and t2: {t2}")
  
  bxm,bym,jxm,jym,Exm,Eym = read_mf_norm(fname=fname)

  return pd.DataFrame.from_dict({
        'Bx': ave_t(bxm,tone=t1,ttwo=t2),
        'By': ave_t(bym,tone=t1,ttwo=t2),
        'Jx': ave_t(jxm,tone=t1,ttwo=t2),
        'Jy': ave_t(jym,tone=t1,ttwo=t2),
        'Ex': -1. * ave_t(Exm,tone=t1,ttwo=t2), # changed signs
        'Ey': -1. * ave_t(Eym,tone=t1,ttwo=t2)        
        })

def gen_df_zave(fname='mfields.npz',z1=128,z2=-1,verbose=True):
  '''
  name_conv: True refers to pencil code data: bxm,bym,jxm,jym,emfx,emfy
  '''

  if verbose:
    print(f"Generating z averaged dataframe with z1: {z1} and z2: {z2}")
  
  bxm,bym,jxm,jym,Exm,Eym = read_mf_norm(fname=fname)

  return pd.DataFrame.from_dict({
        'Bx': ave_z(bxm,zone=z1,ztwo=z2),
        'By': ave_z(bym,zone=z1,ztwo=z2),
        'Jx': ave_z(jxm,zone=z1,ztwo=z2),
        'Jy': ave_z(jym,zone=z1,ztwo=z2),
        'Ex': -1. * ave_z(Exm,zone=z1,ztwo=z2),
        'Ey': -1. * ave_z(Eym,zone=z1,ztwo=z2)        
        })

# PANDAS data frame of poly features
def gen_df_poly(df,deg=3,ignorej=True,verbose=True,dropone=True,feateng=True,keepE=True): 
  # Should throw in SCALED data:
  # df_poly = gen_df_poly(df)
  from sklearn.preprocessing import PolynomialFeatures
  dum_data = df.copy()

  # Feature Engineering in linear phase: E_x,E_y,B_x,B_y,J_x,J_y
  dum_data.drop(['Ex','Ey'],axis=1,inplace=True)
  if ignorej: 
    dum_data.drop(['Jx','Jy'],axis=1,inplace=True)

  p = PolynomialFeatures(degree=deg,include_bias=True).fit(dum_data)
  xpoly = p.fit_transform(dum_data)
  newdf = pd.DataFrame(xpoly, columns = p.get_feature_names(dum_data.columns))

  if keepE:
    newdf['Ex'] = df[['Ex']]
    newdf['Ey'] = df[['Ey']]

  # Feature Engineering in polynomial phase: 
  # Get rid of 1
  # Get rid of Bx^2, By^2 as independent features
  if dropone:
    newdf.drop(['1'],axis=1,inplace=True) # Drop constant
  if feateng:
    newdf['B^2']    = newdf['Bx^2'] + newdf['By^2']
    newdf['B^2 Bx'] = newdf['Bx By^2'] + newdf['Bx^3']
    newdf['B^2 By'] = newdf['Bx^2 By'] + newdf['By^3']
    newdf.drop(['Bx^2','By^2','Bx By^2','Bx^3','Bx^2 By','By^3'],axis=1,inplace=True)

  if verbose:
    print("Feature names:",list(newdf))#newdf.columns.values.tolist())
    print("Feature array shape:",newdf.shape)
    
  # Shape of the columns was a bit messed up; rearranging order
  cols = list(newdf)
  cols = cols[3:] + cols[:3]
  newdf = newdf[cols]
    
  return newdf

def scale_df(df):
  '''
  Call:   df_ss, scl = scale_df(df)
  Inv. Transform: dfn = scl.inverse_transform(df_ss)
  Check equality: np.allclose(df.to_numpy(),dfn.to_numpy())
  '''

  from sklearn.preprocessing import StandardScaler
  df_ss = df.copy()
  scl   = StandardScaler()
  df_ss = scl.fit_transform(df_ss)
  return pd.DataFrame(df_ss,columns=df.columns),scl

def gen_df_zave_log(fname='mfields.npz',z1=128,z2=-1,verbose=True):
  '''
  name_conv: True refers to pencil code data: bxm,bym,jxm,jym,emfx,emfy
  '''

  if verbose:
    print("Generating z averaged dataframe with z1: {} and z2: {}".format(z1,z2))
  
  bxm,bym,jxm,jym,Exm,Eym = read_mf_norm(fname=fname)
  
  return pd.DataFrame.from_dict({
        'Bx2l': np.log10(ave_z(bxm**2,zone=z1,ztwo=z2)),
        'By2l': np.log10(ave_z(bym**2,zone=z1,ztwo=z2)),
        'Jx2l': np.log10(ave_z(jxm**2,zone=z1,ztwo=z2)),
        'Jy2l': np.log10(ave_z(jym**2,zone=z1,ztwo=z2)),
        'Ex2l': np.log10(ave_z(Exm**2,zone=z1,ztwo=z2)),
        'Ey2l': np.log10(ave_z(Eym**2,zone=z1,ztwo=z2))        
        })

def gen_df_poly_zave_log(fname='mfields.npz',z1=128,z2=-1,verbose=True,name_conv=True):
  '''
  name_conv: True refers to pencil code data: bxm,bym,jxm,jym,emfx,emfy
  '''

  if verbose:
    print("Generating z averaged dataframe with z1: {} and z2: {}".format(z1,z2))
  
  if name_conv:
    bxm,bym,jxm,jym,Exm,Eym = read_mf_norm(fname=fname)
  else:
    bxm,bym,jxm,jym,Exm,Eym = read_mf2(fname=fname)

  ff = np.load(fname)
  beq2 = ff['uave']**2
  
  me = bxm**2 + bym**2
  me /= beq2

  bxm_quench = bxm/(1 + me)
  bym_quench = bym/(1 + me)
  jxm_quench = jxm/(1 + me)
  jym_quench = jym/(1 + me)
  
  return pd.DataFrame.from_dict({
        'Bx2l': np.log10(ave_z(bxm_quench**2,zone=z1,ztwo=z2)),
        'By2l': np.log10(ave_z(bym_quench**2,zone=z1,ztwo=z2)),
        'Jx2l': np.log10(ave_z(jxm_quench**2,zone=z1,ztwo=z2)),
        'Jy2l': np.log10(ave_z(jym_quench**2,zone=z1,ztwo=z2)),
        'Ex2l': np.log10(ave_z(Exm**2,zone=z1,ztwo=z2)),
        'Ey2l': np.log10(ave_z(Eym**2,zone=z1,ztwo=z2))        
        })

def pca_df(df,n_components=2,verbose=True):
  '''
  Call:   df_ss, scl = pca_df(df)
  Inv. Transform: dfn = scl.inverse_transform(df_ss)
  Check equality: np.allclose(df.to_numpy(),dfn.to_numpy())
  '''
  
  from sklearn.decomposition import PCA
  df_ss = df.copy()
  pca   = PCA(n_components=n_components)
  df_ss = pca.fit_transform(df_ss)
  if verbose:
    print(f"Variance:     {pca.explained_variance_ratio_}")
    print(f"Sing. Values: {pca.singular_values_}")
  if n_components==2:
    return pd.DataFrame(df_ss,columns=["Vector 1", "Vector 2"]),pca
  else:
    return pd.DataFrame(df_ss),pca
  
def scale_df_Xy(X_train,y_train):
  # X_train must be a dataframe
  # X_train,y_train,_ = scale_df_Xy(X_train,y_train)

  from sklearn.preprocessing import StandardScaler
  Xprime = X_trian.copy()
  scl    = StandardScaler()
  Xprime = scl.fit_transform(df_ss)
  return pd.DataFrame(Xprime,columns=X_prime.columns),y_train,scl

def train_test_seq(X,y,test_size=0.2):
    '''
    Assume X,y are dataframes
    '''
    
    print(f"Test size: {test_size}")
    train_size = 1.0 - test_size
    totsz = X.shape[0]
    X_train, X_test = X.iloc[:int(train_size*totsz)], X.iloc[int(train_size*totsz):]
    y_train, y_test = y.iloc[:int(train_size*totsz)], y.iloc[int(train_size*totsz):]
    
    return X_train, X_test, y_train, y_test

# Convenience routines for correlation plots (OLD but still work)
def df_corr_plot(df,key_name=None,fname=None,tave=True,t1=1000,t2=2000,z1=128,z2=-1,scal=None,verbose=True,savfig=True,labs=None,nocolor=False):
  if fname:
    print('Using file: ',fname)
    if tave:
      print('Time averaging')
      df = gen_df_tave(fname=fname,t1=t1,t2=t2,verbose=verbose)
    else:
      print('Vertical (z) averaging')
      df = gen_df_zave(fname=fname,z1=z1,z2=z2,verbose=verbose)
  else:
    if scal:
      print('Scaling data before plotting correlations')
      df_ss,_ = scale_df(df)
      df = df_ss.copy()

  corr_matrix = df.corr()

  sns.set(style="white")

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr_matrix, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  fig, ax = plt.subplots(figsize=(11, 9))

  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(220, 10, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  if nocolor:
    #with sns.axes_style('white'):
    sns.heatmap(corr_matrix, annot=True, cbar=False, mask=mask, cmap=ListedColormap(['white']), vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5)
  else:
    sns.heatmap(corr_matrix, annot=True, cbar=False, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

  if labs is None:
    labs = [r'$\overline{B}_x$',r'$\overline{B}_y$',r'$\overline{J}_x$',r'$\overline{J}_y$',r'$\mathcal{E}_x$',r'$\mathcal{E}_y$']

  #ax.set_xticklabels(labels=df.columns,fontsize=15)
  #ax.set_yticklabels(labels=df.columns,fontsize=15)
  ax.set_xticklabels(labels=labs,fontsize=16,rotation=45)
  ax.set_yticklabels(labels=labs,fontsize=16,rotation=45)
  #ax.set_yticklabels(labels=labs[::-1],fontsize=16,rotation=45)

  if savfig:
    try:
      os.makedirs("correlation")
    except FileExistsError:
      # directory already exists
      pass
    if fname: 
      savfile = 'correlation/corr_' + fname[:-4] + '.pdf'
    else:
      savfile = 'correlation/corr_' + key_name + '.pdf'
    fig.savefig(savfile,bbox_inches='tight')

def df_poly_corr_plot(df,key_name=None,fname=None,tave=True,t1=1000,t2=2000,z1=128,z2=255,scal=None,deg=3,ignorej=True,verbose=True,savfig=True):
  # Input: Unscaled df with just B's,J's,EMF's
  # Output: Correlation plot with EMF's retained (J dropped), polynomial in B
  # Data is SCALED
  if fname:
    print('Using file: ',fname)
    if tave:
      print('Time averaging')
      df = gen_df_tave(fname=fname,t1=t1,t2=t2,verbose=verbose)
    else:
      print('Vertical (z) averaging')
      df = gen_df_zave(fname=fname,z1=z1,z2=z2,verbose=verbose)
    if scal:
      print('Scaling data before plotting correlations')
      df_ss,_ = scale_df(df)
      df = df_ss.copy()

  df_poly = gen_df_poly(df,deg=deg,ignorej=ignorej,verbose=verbose)
  df_poly['Ex'] = df[['Ex']]
  df_poly['Ey'] = df[['Ey']]

  #df_poly.drop(['1'],axis=1,inplace=True) # Drop constant

  df_poly_ss,_ = scale_df(df_poly)

  corr_matrix = df_poly_ss.corr()

  sns.set(style="white")
  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr_matrix, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  fig, ax = plt.subplots(figsize=(11, 9))
  
  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(220, 10, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr_matrix, annot=True, cbar=False, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

  ax.set_xticklabels(labels=df_poly_ss.columns,fontsize=15,rotation=45)
  ax.set_yticklabels(labels=df_poly_ss.columns,fontsize=15,rotation=45)
  #ax.set_xticklabels(labels=[r'$\overline{B}_x$',r'$\overline{B}_y$',r'$\overline{J}_x$',r'$\overline{J}_y$',r'$\mathcal{E}_x$',r'$\mathcal{E}_y$'],fontsize=16)
  #ax.set_yticklabels(labels=[r'$\overline{B}_x$',r'$\overline{B}_y$',r'$\overline{J}_x$',r'$\overline{J}_y$',r'$\mathcal{E}_x$',r'$\mathcal{E}_y$'],fontsize=16)

  if savfig:
    try:
      os.makedirs("correlation")
    except FileExistsError:
      # directory already exists
      pass
    if fname: 
      savfile = 'correlation/corr_' + fname[:-4] + '.pdf'
    else:
      savfile = 'correlation/corr_' + key_name + '.pdf'
    fig.savefig(savfile, bbox_inches='tight')
