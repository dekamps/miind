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


'''Performs the analysis of conductance-based model. Unlike the lifanalysis.py and qifanalysis.py script, it assumes that the matrix files for these experiments already
have been generated. It is assumed that all model and matrix files have been provided with this script and that they reside in this director.
Run the check_setup_routine to find out if this is the case.'''

MODELS={}
MATRICES={}
MODELS['standard']='condee2a5ff4-0087-4d69-bae3-c0a223d03693.model'
MATRICES[MODELS['standard']]=['condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.05_0_0_.mat', 'condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.1_0_0_.mat']
MODELS['clipped']='cond7bc12c35-9d87-43fc-8ccc-9223c2714440.model'
MATRICES[MODELS['clipped']]=['cond7bc12c35-9d87-43fc-8ccc-9223c2714440_0_0.05_0_0_.mat','cond7bc12c35-9d87-43fc-8ccc-9223c2714440_0_0.1_0_0_.mat']

J= [0.05,0.1] # jump size.
DIR_GAIN_STANDARD = 'gain_standard' # directory where the gain results should be produced
DIR_GAIN_CLIPPED  = 'gain_clipped'


def check_setup_routine():
    for model in MODELS:
        if not os.path.exists( MODELS[model]):
            raise NameError('Model file missing: ' + MODELS[model])
        for matrix in MATRICES[MODELS[model]]:
            if not os.path.exists(matrix):
                raise NameError('Matrix file missing: ' + matrix)
    print 'All files present'


def generate_gain_xml_files(xml_file, rates, J, model, matrices, dir):

    for j in J:
        for rate in rates:
            f=ps.xml_file (xml_file)
            abs_path = os.path.join(dir, dir + '_' + str(j) + '_' + str(rate) +  '.xml')
            tag_st = ps.xml_tag('<t_end>0.3</t_end>')
            f.replace_xml_tag(tag_st,2.0)
            tag_exp =ps.xml_tag('<expression>1000</expression>')
            f.replace_xml_tag(tag_exp,rate)
            tag_mf = ps.xml_tag('<MatrixFile>condee2a5ff4-0087-4d69-bae3-c0a223d03693_0_0.05_0_0_.mat</MatrixFile>')
            if j == 0.05:
                f.replace_xml_tag(tag_mf,matrices[0])
            if j == 0.1:
                f.replace_xml_tag(tag_mf,matrices[1])

            tag_con= ps.xml_tag('<Connection In="Inp" Out="AdExp E">1   0.05 0</Connection>')
            f.replace_xml_tag(tag_con,j,1)
 
            algs = f.tree.getroot().find('Algorithms')
            for a in algs:
                if a.attrib['type'] == 'MeshAlgorithm':
                    a.attrib['modelfile'] = model
        
            f.write(abs_path)

def generate_gain(rerun=True,batch=False):

    if not os.path.exists(DIR_GAIN_STANDARD):
        sp.call(['mkdir', DIR_GAIN_STANDARD] )
    if not os.path.exists(DIR_GAIN_CLIPPED):
        sp.call(['mkdir', DIR_GAIN_CLIPPED] )

    # the matrix and model files must be present in the gain directories. 
    sp.call(['cp',MODELS['standard'], DIR_GAIN_STANDARD])
    for name in MATRICES[MODELS['standard']]:
        sp.call(['cp', name, DIR_GAIN_STANDARD])
    # the matrix and model files must be present in the gain directories. 
    sp.call(['cp',MODELS['clipped'], DIR_GAIN_CLIPPED])
    for name in MATRICES[MODELS['clipped']]:
        sp.call(['cp', name, DIR_GAIN_CLIPPED])
    
    input_rates = np.arange(0.,3500, 100.)

    generate_gain_xml_files('cond.xml',input_rates, [0.05, 0.1], MODELS['standard'], MATRICES[MODELS['standard']],DIR_GAIN_STANDARD)
    generate_gain_xml_files('cond.xml',input_rates, [0.05, 0.1], MODELS['clipped'], MATRICES[MODELS['clipped']], DIR_GAIN_CLIPPED)

    if rerun == True:
        ut.instantiate_jobs(DIR_GAIN_STANDARD,batch)
        ut.instantiate_jobs(DIR_GAIN_CLIPPED, batch)

def demofy():
    '''This functions massages batch submission scripts so that they have appropriate settings for parameters. If
    none are present it is assumed that this is not needed.'''

    if not os.path.exists('sub.sh'):
        return
    with open('sub.sh') as f:
	lines = f.readlines()
	lines[-1] =  '#$ -l h_vmem=16000M\n' 
	lines.append('demo2D.py ' + MODELS['oslo'] + ' y  b')
        replines = [ w.replace('4:','12:') for w in lines ]

    with open('demo.sh','w') as g:
	for line in replines:
	    g.write(line)
    sp.call(['chmod', '+x', 'demo.sh'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', action='store_true')
    parser.add_argument('--d', action='store_true')
    args = parser.parse_args()

    demofy()
    if args.d == False:
        print 'Generating simulation files'
        generate_gain(rerun=True,batch=args.b)

    if args.d == True:
	if args.b == True:
            print 'Batch option ignored in DST production.'
    	else:
	    dst_name = 'DST_' + DIR_GAIN
	    sp.call(['mkdir',dst_name])
	    dst_path = os.path.join(sys.path[0], dst_name)
	    dir_list = sp.check_output(['ls','spectrum/spectrum']).split()
	    for d in dir_list:
		om = d.split('_')[1]
		ratefile = 'single'
#		sp.call(['cp',os.path.join(DIR_SPECTRUM,DIR_SPECTRUM,d,ratefile),os.path.join(dst_name,'rate_file' + '_' + om)])
            # that out of the way, submit visualization as batch job
	    for d in dir_list:
		dir_path = os.path.join(DIR_SPECTRUM, DIR_SPECTRUM, d)
		sp.call(['cp','demo.sh',dir_path])
		with ut.cd(dir_path):
		    sp.call(['qsub','demo.sh'])		
