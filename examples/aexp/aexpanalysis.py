#!/usr/bin/env python 
import argparse
import ROOT
import parametersweep as ps
import glob
import os
import sys
import numpy as np
import subprocess as sp
import utilities as ut
import visualize

ROOT.gROOT.SetBatch(ROOT.kTRUE)



'''Performs the analysis of AdExp model. Unlike the lifanalysis.py and qifanalysis.py script, it assumes that the matrix files for these experiments already
have been generated. It is assumed that all model and matrix files have been provided with this script and that they reside in this director.
Run the check_setup_routine to find out if this is the case.'''

MODELS={}
MATRICES={}
MODELS['oslo']='aexp7ee3879d-3dd7-41d2-972e-47f4d16845cd.model'
MATRICES[MODELS['oslo']]=['aexp7ee3879d-3dd7-41d2-972e-47f4d16845cd_-1_0_0_0_.mat','aexp7ee3879d-3dd7-41d2-972e-47f4d16845cd_1_0_0_0_.mat']

J= [-1.,1.] # jump size.
DIR_SPECTRUM = 'spectrum' # directory where the resonance results should be produced


# This analysis script draws on the following meshes and mat files: 
model='aexp_oslo.model'

def check_setup_routine():
    for model in MODELS:
        if not os.path.exists( MODELS[model]):
            raise NameError('Model file missing: ' + MODELS[model])
        for matrix in MATRICES[MODELS[model]]:
            if not os.path.exists(matrix):
                raise NameError('Matrix file missing: ' + matrix)
    print 'All files present'

def generate_spectrum_xml_files(model_file, J, omega, matnames, dir_spectrum):

    for el in omega:
        f=ps.xml_file (model_file)
        abs_path = os.path.join(dir_spectrum, dir_spectrum + '_' + str(el) +  '.xml')
        tag_om = ps.xml_tag('<Variable Name="omega">1.0</Variable>')
        f.replace_xml_tag(tag_om,el)
        tag_st = ps.xml_tag('<t_end>0.2</t_end>')
        f.replace_xml_tag(tag_st,0.5)
        tag_exp =ps.xml_tag('<expression>2000</expression>')
        f.replace_xml_tag(tag_exp,'2000*sin(omega*t) >= 0. ?  2000*sin(omega*t) : 0.',order=0,split=False)
        f.replace_xml_tag(tag_exp,'2000*sin(omega*t) <  0. ? -2000*sin(omega*t) : 0.',order=1,split=False)
        f.write(abs_path)

def generate_omega_values():
    expos = np.linspace(1.,2.,100)
    omega=np.power(10,expos)
    return omega

def generate_spectrum(rerun=True,batch=False):

    if not os.path.exists(DIR_SPECTRUM):
        sp.call(['mkdir', DIR_SPECTRUM] )

    # the matrix and model files must be present in the spectrum directories. 
    for name in MATRICES[MODELS['oslo']]:
        sp.call(['cp', name, DIR_SPECTRUM])

    modelname = glob.glob('*.model')
    for mn in modelname:
        sp.call(['cp', mn, DIR_SPECTRUM])

    omega = generate_omega_values()
    generate_spectrum_xml_files('aexp_oslo.xml',J,omega,MATRICES[MODELS['oslo']],DIR_SPECTRUM)

    if rerun == True:
        ut.instantiate_jobs(DIR_SPECTRUM,batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', action='store_true')
    parser.add_argument('--d', action='store_true')
    args = parser.parse_args()


    if args.d == False:
        print 'Generating simulation files'
        generate_spectrum(rerun=True,batch=args.b)

    if args.d == True:
	if args.b == True:
            print 'Batch option ignored in DST production.'

        J1, nu1, ms1 = generate_response_curve_values_low_noise(sigma=0.158)
        J2, nu2, ms2 = generate_response_curve_values_low_noise(sigma=0.316)
        J5, nu5, ms5 = generate_response_curve_values_low_noise(sigma=0.791)
        J  = np.concatenate((J1,  J2,  J5))
	nu = np.concatenate((nu1, nu2, nu5))
        ms = ms1 + ms2 + ms5
        fns = [ 'response_' + str(el[0]) + '_' + str(el[1]) for el in zip(J,nu) ]
        d = { el[0]: el[1] for el in zip(fns,ms)}

	ut.produce_data_summary('response',[0],model,d)
        ut.produce_data_summary('spectrum',[0],model)
