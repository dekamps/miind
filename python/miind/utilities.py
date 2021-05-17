from contextlib import contextmanager
import glob
import os
import numpy as np
import subprocess as sp
import sys
import miind.visualize as visualize
import ROOT

# the tolerance by which efficacies can be distinguished from their file names
tolerance = 1e-8

# use: with cd('...'), returns to the original directory upon encountering exception
# from: http://stackoverflow.com/questions/431684/how-do-i-cd-in-python

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def associate_efficacy(J, matnames):
    '''Select the  matname corresponding to efficacy J out of a list of matrix names, returns the name.  Raises an exception when the name can't be found.'''
    for name in matnames:
        eff = float(name.split('_')[1])
        if np.fabs(J - eff) < tolerance:
            return name
    raise ValueError


def instantiate_jobs(dirname, batch):
    with cd(dirname):
        xmlfiles   = glob.glob('*.xml')
        matfiles   = glob.glob('*.mat')
        modelfile  = glob.glob('*model')
        cmd = ['miind.py', '--d',dirname]
        cmd.extend(['--mpi','--openmp'])
        cmd.extend(xmlfiles)
        cmd.extend(['--m'])
        cmd.extend(modelfile)
        cmd.extend(matfiles)
        sp.call(cmd)
        if batch == False:
            sp.call(['submit.py',dirname])
        else:
            sp.call(['submit.py',dirname,sys.path[0]])

def find_results(dir_name):
    '''This function helps to navigate the standard workflow. In general, an analysis script will create a directory dir_name and write the XML files there.
    For the purpose of the workflow, it is therefore assumed that this directory already exists. The miind.py script will be run from directory dir_name, which
    results in the creation of directory dir_name within dir_name, and the latter directory contains all results directory. Although this appears complex,
    the rationale is that typically, one wants to organise the results of the script in a directory, which is the top level dir_response. miind.py will be run from
    there, which then creates the hiearchy dir_response/[list of result directories]. This routine will return
    the absolute path and a list of directories, each containing the simulation results.'''
    if not os.path.exists(dir_name):
        raise ValueError

    respath = os.path.join(dir_name,dir_name)
    with cd(respath):
        fns=str(sp.check_output(['ls'],encoding='UTF-8')).split('\n')
        abspath = os.getcwd()
        retlist = [ d for d in fns if dir_name in d]
        return abspath, retlist



def extract_rate_graph(root_file, population_list, dstname):
    f = ROOT.TFile(root_file)
    ts = []
    fs = []
    for id in population_list:
        graphname = 'rate_' + str(id)
        g = f.Get(graphname)
        x_buff = g.GetX()
        y_buff = g.GetY()
        N = g.GetN()
        x_buff.SetSize(N)
        y_buff.SetSize(N)
        # Create numpy arrays from buffers, copy to prevent data loss
        x_arr = np.ndarray(N,buffer=x_buff, dtype= np.float32)
        y_arr = np.ndarray(N,buffer=y_buff, dtype= np.float32)
        ts.append(x_arr)
        fs.append(y_arr)
    return ts, fs



def find_last_density_file(dense_list):
    '''Out of a list with density file names, produce the one that relates to the latest simulation time.'''
    clean_list = [ f for f in dense_list if len(f.split('_')) > 1 ]
    return sorted(clean_list, key = lambda f: float(f.split('_')[2]))[-1]


def produce_data_summary(dir_name, population_list, model, simulationname='', mapping_dictionary = None):
    ''' dir_name is the directory name of the simulation results. The standard workflow is assumed, which produces a hierarchy of three directories deep:
    dir_name/dir_name/[list of result directories]. Only dir_name must be provided, the result directories will be found and listed by the routines.
    Population list is a list of NodeId's. T directory DST_<dir_name> will be created. The firing rates of the populations in
    the poplation_list will be extracted, and the steady state density file, i.e. the density with the latest time stamp will be saved there.'''
    dstname = 'DST_' + dir_name
    if not os.path.exists(dstname):
        os.makedirs(dstname)
    path, dirs = find_results(dir_name)
    curpath = os.getcwd()

    for di in dirs:
        with cd(os.path.join(path,di)):
            if simulationname == '':
                rt = dir_name + '_' + '0' + '.root'
            else:
                rt = simulationname + '_' + '0' + '.root'
            if os.path.exists(rt):
                ts, fs = extract_rate_graph(rt, population_list, dstname)
                # create the DST directory in the script directory
                dst_dir_name = os.path.join(curpath,os.path.join(sys.path[0],dstname))
                if not os.path.exists(dst_dir_name):
                    os.makedirs(dst_dir_name)
                if mapping_dictionary == None:
                    fname = di
                else:
                    fname = str(mapping_dictionary[di][0])
                    for i in range(1, len(mapping_dictionary[di])):
                        fname += '_' + str(mapping_dictionary[di][i])

                for i, id in enumerate(population_list):
                    print(os.path.join(dst_dir_name, fname + '_' + str(id) +  '.rate'))
                    with open(os.path.join(dst_dir_name, fname + '_' + str(id) +  '.rate'),'w') as f:
                        for t in ts[i]:
                            f.write(str(t) + '\t')
                        f.write('\n')
                        for freq in fs[i]:
                            f.write(str(freq) + '\t')
                        f.write('\n')
            else:
                print('Cannot find: ', rt)

                    # the density files reside here:
                    #dir_density = model + '_mesh'
                    #with cd(dir_density):
                    #    dfs = sp.check_output(['ls']).split('\n')
                    #    dfn=find_last_density_file(dfs)
                    #    v=visualize.Model1DVisualizer('../' + model)
                    #    v.showfile(dfn)
                    #with open(os.path.join(dst_dir_name, fname + '_' + str(id) + '.steady'),'w') as f:
                    #    for volt in v.interpretation:
                    #        f.write(str(volt) + '\t')
                    #    f.write('\n')
                    #    for dens in v.density:
                    #        f.write(str(dens) + '\t')
                    #    f.write('\n')
