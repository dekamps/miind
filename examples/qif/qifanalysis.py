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

# This analysis script draws on the following meshes and mat files:
model='qif470e055c-54b4-4e9d-8687-addeb7bbab80.model'

# the neural constants used in the generation of this mesh were:
I     = 1
tau   = 20e-3

# how many MC points to use the transition matrices
nr_mc = 100

# directory of the files generated
DIR_RESPONSE = 'response'
DIR_SPECTRUM = 'spectrum'

def generate_response_curve_values_low_noise(mu = np.arange(2.0, 7.0, 0.2 ), sigma = 2e-1):
    '''Takes mu and sigma, return rate and efficacy to emulate a Gaussian white noise as a Possion input with rate rate and efficacy J.'''
    J  = []
    nu = []
    ms = []

    # Use this inelegant loop in order to store the associations
    for m in mu:        
        J.append(sigma*sigma/m)
        nu.append(np.power(m,2)/(sigma*sigma*tau))
        ms.append([m,sigma])

    return np.array(J), np.array(nu), ms



def generate_transition_matrices(J, dir):
    '''Takes an array of synatic efficacies and generates the transition matrices, adds the names of the generated matrices to global array transition_matrices. It                  
    is essential that the fiducial file cover the synaptic efficacies, so the file life23f9d42-38a0-450c-bd59-5047dd0f5326.fid should be present in the directory                    
    where this script is run. It is the user's responsibility to check whether the file is adequate. The fiducial file is generated for trainsitions up 0.01,                        
    both excitatory and inhibitory'''
    for j in J:
        print 'Generating transition: ', j
        sp.check_output(['MatrixGenerator', model, model.split('.')[0] + '.fid', str(nr_mc),str(j),'0','0'])
    name=sp.check_output(['ls','-t']).split('\n')
    matnames = glob.glob('*.mat')
    for name in matnames:
        sp.call(['mv', name, dir])

    modelname = glob.glob('*.model')
    for mn in modelname:
        sp.call(['cp', mn, dir])

    losts=glob.glob('*lost')
    for lost in losts:
        sp.call(['rm', lost ])
    return matnames



def generate_response_xml_files(J, nu, ms, matnames, directory):
    '''Take the basic lif xmp template and adapt to the J, nu under consideration.'''

    for el in zip(J, nu, ms):
        f=ps.xml_file ('qif.xml')
        tag_mat = ps.xml_tag('<MatrixFile>life23f9d42-38a0-450c-bd59-5047dd0f5326_0.0001860465116_0_0_0_.mat</MatrixFile>')
        matname = ut.associate_efficacy(el[0], matnames)
        f.replace_xml_tag(tag_mat,matname)
        tag_rate = ps.xml_tag('<expression>1000.</expression>')
        f.replace_xml_tag(tag_rate,el[1])
        tag_con = ps.xml_tag('<Connection In="Inp" Out="AdExp E">1 0.0001860465116 0</Connection>')
        f.replace_xml_tag(tag_con,el[0],1)
        tag_st = ps.xml_tag('<t_end>0.3</t_end>')
        f.replace_xml_tag(tag_st,30.0)
        tag_fn = ps.xml_tag('<SimulationName>lif.dat</SimulationName>')
        f.replace_xml_tag(tag_fn,'response')
        abs_path = os.path.join(DIR_RESPONSE, DIR_RESPONSE + '_' + str(el[0]) + '_' + str(el[1]) + '.xml')
        f.write(abs_path)


def generate_response(rerun=True,batch=False):
    if not os.path.exists(DIR_RESPONSE):
        sp.call(['mkdir', DIR_RESPONSE] )

    J, nu, ms = generate_response_curve_values_low_noise(sigma=0.158)
    matnames = generate_transition_matrices(J, DIR_RESPONSE)
    generate_response_xml_files(J, nu, ms, matnames, DIR_RESPONSE)

    J, nu, ms = generate_response_curve_values_low_noise(sigma=0.316)
    matnames = generate_transition_matrices(J, DIR_RESPONSE)
    generate_response_xml_files(J, nu, ms, matnames, DIR_RESPONSE)

    J, nu, ms = generate_response_curve_values_low_noise(sigma = 0.791)
    matnames = generate_transition_matrices(J, DIR_RESPONSE)
    generate_response_xml_files(J, nu, ms, matnames, DIR_RESPONSE)

    if rerun == True:
        ut.instantiate_jobs(DIR_RESPONSE,batch)    


def generate_spectrum_xml_files(J, omega, matnames, dir_spectrum):
    for el in omega:
        f=ps.xml_file ('qif.xml')
        abs_path = os.path.join(dir_spectrum, dir_spectrum + '_' + str(el) +  '.xml')
        tag_mat = ps.xml_tag('<MatrixFile>life23f9d42-38a0-450c-bd59-5047dd0f5326_0.0001860465116_0_0_0_.mat</MatrixFile>')
        matname = ut.associate_efficacy(J[0], matnames)
        f.replace_xml_tag(tag_mat,matname)
        tag_fn = ps.xml_tag('<SimulationName>lif.dat</SimulationName>')
        f.replace_xml_tag(tag_fn,'spectrum')
        tag_con = ps.xml_tag('<Connection In="Inp" Out="AdExp E">1 0.0001860465116 0</Connection>')
        f.replace_xml_tag(tag_con,J[0],1)
        tag_st = ps.xml_tag('<t_end>0.3</t_end>')
        f.replace_xml_tag(tag_st,20.0)
        tag_rate = ps.xml_tag('<expression>1000.</expression>')
        f.replace_xml_tag(tag_rate,'30*sin(' + str(el) + '*t)+2500')

        f.write(abs_path)

def generate_omega_values():
    J = 0.1*np.sqrt(I)
    expos = np.linspace(0,3.3,100)
    omega=np.power(10,expos)
    return [J], omega

def generate_spectrum(rerun=True,batch=False):
    if not os.path.exists(DIR_SPECTRUM):
        sp.call(['mkdir', DIR_SPECTRUM] )

    J, omega = generate_omega_values()
    matnames = generate_transition_matrices(J, DIR_SPECTRUM)
    generate_spectrum_xml_files(J,omega,matnames,DIR_SPECTRUM)

    if rerun == True:
        ut.instantiate_jobs(DIR_SPECTRUM,batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', action='store_true')
    parser.add_argument('--d', action='store_true')
    args = parser.parse_args()


    if args.d == False:
        print 'Generating simulation files'
        generate_response(rerun=True,batch=args.b)
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
