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
    cfgfile = open("config.ini",'w')
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
    Config.set('DHI','dhi_config_file',r"C:\dhi_release_sc\Configuration_DHI_IA_Human.mat")
    Config.set('DHI','dhi_class_file','dhi_class.mat')
    Config.add_section('OUTPUT')
    Config.set('OUTPUT','output_option','1')
    Config.set('OUTPUT','output_format','nii')
    Config.set('OUTPUT','iso_threshold','0.3,0.3,3.0,3.0')
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
    eng.addpath(r'C:\dhi_release_sc',r'C:\dhi_release_sc\Misc', r'C:\dhi_release_sc\Misc\NIfTI_20140122')   
    eng.dhi_load(current_dir + '/' + 'config.ini',nargout=0)
    eng.quit()
def dhi_calc(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nojvm -nodisplay")
    eng.addpath(r'C:\dhi_release_sc',r'C:\dhi_release_sc\Misc', r'C:\dhi_release_sc\Misc\NIfTI_20140122')      
    eng.dhi_calc('config.ini',nargout=0)
    eng.quit()
def dhi_save(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nojvm -nodisplay")
    eng.addpath(r'C:\dhi_release_sc',r'C:\dhi_release_sc\Misc', r'C:\dhi_release_sc\Misc\NIfTI_20140122')      
    eng.dhi_save(current_dir + '/' + 'config.ini',nargout=0)
    eng.quit()


new_dwi_files = glob.glob(r"C:\MCSIM\dMRI-MCSIM-main\6x6-Mosaic_DBSI-vs-NEO\Row=?_Col=?\_Total_Signal*.nii")

for patient in new_dwi_files:
    path, file = os.path.split(patient)
    shutil.copyfile(src = r"C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval\bval-99.bval", dst = path + os.sep + 'bval')
    shutil.copyfile(src = r"C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bvec\bvec-99.bvec", dst = path + os.sep + 'bvec')
    dhi_config(patient)
    dhi_load(patient)
    dhi_calc(patient)
    dhi_save(patient)
