import matlab.engine
import os 
import configparser
import shutil
import glob as glob




def dhi_config(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname  
    os.chdir(current_dir)
    Config = configparser.ConfigParser()
    cfgfile = open("DHIconfig.ini",'w')
    Config.add_section('INPUT')
    Config.set('INPUT','data_dir',current_dir)
    path,dwi_file = os.path.split(nii_file)
    path,bval = os.path.split(os.path.join(current_dir + os.sep + 'bval'))
    path,bvec = os.path.split(os.path.join(current_dir + os.sep + 'bvec'))  
    Config.set('INPUT','dwi_file',"%s"%dwi_file)
    Config.set('INPUT','mask_file',"NA")
    Config.set('INPUT','rotation_matrix','NA')
    Config.set('INPUT','bval_file',"%s"%bval)
    Config.set('INPUT','bvec_file',"%s"%bvec)
    Config.set('INPUT','preprocess','NA')
    Config.set('INPUT','slices_to_compute','0')
    Config.set('INPUT','dhi_mode','map')
    Config.set('INPUT','norm_by_bvec','no')
    Config.set('INPUT','bmax_dhi',' ')
    Config.set('INPUT','dhi_input_file','dhi_input.mat')
    Config.add_section('DHI')
    Config.set('DHI','dhi_input_file','dhi_input.mat')
    Config.set('DHI','dhi_config_file',r'/bmr207/nmrgrp/nmr202/dhi_release/dhiConfigFiles/Configuration_InVivo_Human_GBM.mat')
    Config.set('DHI','dhi_class_file','dhi_class.mat')
    Config.add_section('OUTPUT')
    Config.set('OUTPUT','output_option','0')
    Config.set('OUTPUT','output_format','nii')
    Config.set('OUTPUT','iso_threshold','0.2,0.8,0.8,2.5,2.5')
    Config.set('OUTPUT','output_fib','0')
    Config.set('OUTPUT','output_fib_res','1,1,1')
    Config.write(cfgfile)
    cfgfile.close()    
def dhi_load(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nojvm -nodisplay")
    eng.addpath(r'/bmr207/nmrgrp/nmr202/dhi_release',r'/bmr207/nmrgrp/nmr202/dhi_release/Misc', r'/bmr207/nmrgrp/nmr202/dhi_release/Misc/NIfTI')      
    eng.dhi_load(current_dir + '/' + 'DHIconfig.ini',nargout=0)
    eng.quit()
def dhi_calc(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nodisplay")
    eng.addpath(r'/bmr207/nmrgrp/nmr202/dhi_release',r'/bmr207/nmrgrp/nmr202/dhi_release/Misc', r'/bmr207/nmrgrp/nmr202/dhi_release/Misc/NIfTI')      
    eng.dhi_calc('DHIconfig.ini',nargout=0)
    eng.quit()
def dhi_save(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nojvm -nodisplay")
    eng.addpath(r'/bmr207/nmrgrp/nmr202/dhi_release',r'/bmr207/nmrgrp/nmr202/dhi_release/Misc', r'/bmr207/nmrgrp/nmr202/dhi_release/Misc/NIfTI')      
    eng.dhi_save_4iso(current_dir + '/' + 'DHIconfig.ini',nargout=0)
    eng.quit()


new_dwi_files = glob.glob(r"/bmr207/nmrgrp/nmr202/MCSIM/mosaic_6x6_out/**/totalSignal.nii")

for patient in new_dwi_files:
    path, file = os.path.split(patient)
    shutil.copyfile(src = r"/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI/bval-99.bval", dst = path + os.sep + 'bval')
    shutil.copyfile(src = r"/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI/bvec-99.bvec", dst = path + os.sep + 'bvec')
    dhi_config(patient)
    dhi_load(patient)
    dhi_calc(patient)
    dhi_save(patient)
