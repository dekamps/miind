#!/usr/bin/env python

import argparse
import codegen
import sys
import os
import os.path as op
import directories
import jobs
import miind_api as api
import matplotlib.pyplot as plt
import directories

def getMiindPythonPath():
    return os.path.join(directories.miind_root(), 'python')

def _help(command):
    command_name = command[0]
    name = 'help'
    alts = ['h', '?', 'commands', 'list']

    if command_name in [name] + alts:
        print ''
        print 'MIIND UI'
        print 'For more information on any command type the command name and a \'?\' [eg. sim?]'
        print ''
        print 'help                     : Get this help menu.'
        print 'quit                     : Close the UI.'
        print ''
        print '***** Commands for Creating and Running Simulations *****'
        print ''
        print 'sim                      : Set the current simulation from an xml file or generate a new xml file.'
        print 'models                   : List all model files used by the current simulation.'
        print 'settings                 : Set certain persistent parameters to match your MIIND installation (ENABLE MPI, OPENMP, ROOT).'
        print 'submit                   : Generate and build (make) the code from the current simulation.'
        print 'run                      : Run the current submitted simulation.'
        print 'build-shared-lib         : Generate and build (make) a shared library for use with python from the current simulation.'
        print ''
        print '***** Commands for Analysing and Presenting Completed Simulations *****'
        print ''
        print 'rate                     : Plot the mean firing rate of a given node in the current simulation.'
        print 'plot-density             : Plot the 2D density of a given node at a given time in the simulation.'
        print 'plot-marginals           : Plot the marginal densities of a given node at a given time in the simulation.'
        print 'generate-density-movie   : Generate a movie of the 2D density for a given node in the simulation.'
        print 'generate-marginal-movie  : Generate a movie of the Marginal densities for a given node in the simulation.'
        print ''
        print '***** Commands for Building New Models and Matrix Files *****'
        print ''
        print 'generate-model           : Generate a model file from existing mesh, rev and stat files.'
        print 'generate-empty-fid       : Generate a stub .fid file.'
        print 'generate-matrix          : Generate a matrix file from existing model and fid files.'
        print 'fix-lost                 : Open the fiducial tool for capturing lost points.'
        print 'generate-lif-mesh        : Example to illustrate a mesh generation script to build a LIF neuron mesh.'
        print 'draw-mesh                : Draw the mesh described in an existing .mesh file.'
        print ''

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : Display the available commands.'

def sim(command, current_sim):
    command_name = command[0]
    name = 'sim'

    if command_name in [name]:
        if len(command) == 1:
            if not current_sim:
                print 'No simulation currently defined.'
                print ''
                sim([name+'?'])
            print 'Original XML File : {}'.format(current_sim.original_xml_path)
            print 'Project Name : {}'.format(current_sim.submit_name)
            print 'Parameters : {}'.format(current_sim.parameters)
            print 'Generated XML File : {}'.format(current_sim.xml_fname)
            print 'Output Directory : {}'.format(current_sim.output_directory)
            print 'Variables :'
            for name in current_sim.variablenames:
                print '   ' + name
            print 'Nodes :'
            for name in current_sim.nodenames:
                if name in [n for (n,m) in current_sim.meshnodenames]:
                    print '   ' + name + ' (Mesh Node)'
                else:
                    print '   ' + name
        if len(command) == 2:
            current_sim = api.MiindSimulation(command[1])
        if len(command) == 3:
            current_sim = api.MiindSimulation(command[1], command[2])
        if len(command) > 3:
            comm_dict = {}
            for comm in command[3:]:
                kv = comm.split('=')
                comm_dict[kv[0]] = kv[1]
            current_sim = api.MiindSimulation(command[1], command[2], **comm_dict)

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : Provide information on the current simulation.'
        print name + ' [XML filename] : Use this XML file for the current simulation. The default project name is the same as the XML filename.'
        print name + ' [XML filename] [Project name] : Use this XML file for the current simulation and set a different project name.'
        print name + ' [XML filename] [Project name] [Parameter 1] [Parameter 2] ... : Use this XML file as a template with this project name to generate a new xml file (if not already generated) in which the Variable objects are set using the given Parameters.'

    return current_sim

def models(command, current_sim):
    command_name = command[0]
    name = 'models'

    if command_name in [name]:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        print 'Model files used in ' + current_sim.submit_name + ' :\n'
        for mf in current_sim.modelfiles:
            print mf

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : List all model files used by the current simulation.'

def _settings(command):
    command_name = command[0]
    name = 'settings'

    if command_name in [name]:
        if len(command) == 1:
            print 'Current Settings :\n'
            for k,v in settings.iteritems():
                print k + ' = ' + str(v)

        if len(command) > 1 and len(command) != 4:
            print "settings command expects [ENABLE_MPI] [ENABLE_OPENMP] [ENABLE_ROOT]."

        settings['mpi_enabled'] = (command[1] in ['True', 'true', 'TRUE'])
        settings['openmp_enabled'] = (command[2] in ['True', 'true', 'TRUE'])
        settings['root_enabled'] = (command[3] in ['True', 'true', 'TRUE'])

        with open(settingsfilename, 'w') as settingsfile:
            for k,v in settings.iteritems():
                settingsfile.write(k + '=' + str(v) + '\n')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : List the current settings. Settings are stored in \'miind_ui_settings\' in the MIIND python directory.'
        print name + ' [ENABLE_MPI] [ENABLE_OPENMP] [ENABLE_ROOT] : Expects \'True\' or \'False\' for each of the three settings. Settings are persistent.'


def submit(command, current_sim):
    command_name = command[0]
    name = 'submit'

    if command_name in [name]:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 1:
            current_sim.submit(True,
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'])
        if len(command) >= 2:
            current_sim.submit(True,
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'], *command[1:])

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : Generate and \'make\' the code from the current simulation xml file. Ensure you have the correct settings (call \'settings\').'
        print name + ' [make argument 1] [make argument 2] ... : Generate and \'make (with the provided additional arguments)\' the code from the current simulation xml file.'

def run(command, current_sim):
    command_name = command[0]
    name = 'run'

    if command_name in [name]:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 1:
            current_sim.run()
        if len(command) == 2:
            current_sim.run_mpi(int(command[1]))

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : Run the current simulation. The simulation must have previously been \'submitted\'.'
        print name + ' [Number of cores] : Run the current simulation using mpiexec with the given number of cores. MPI must be enabled in MIIND.'

def buildSharedLib(command, current_sim):
    command_name = command[0]
    name = 'build-shared-lib'
    alts = ['bsl']

    if command_name in [name] + alts:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 1:
            current_sim.submit_shared_lib(True,
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'])
        if len(command) >= 2:
            current_sim.submit_shared_lib(True,
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'], *command[1:])

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : Generate and \'make\' a shared library for use with python from the current simulation xml file. Ensure you have the correct settings (call \'settings\').'
        print name + ' [make argument 1] [make argument 2] ... : Generate and \'make (with the provided additional arguments)\' a shared library for use with python from the current simulation xml file.'
        print 'Alternative command names : ' + ' '.join(alts)

def rate(command, current_sim):
    command_name = command[0]
    name = 'rate'

    if command_name in [name]:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 1:
            print 'The following nodes can be queried for a mean firing rate :'
            for name in current_sim.nodenames:
                print name
        if len(command) == 2:
            current_sim.plotRate(command[1])

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : List the nodes for which a rate plot is available in the current simulation. The current simulation must have been submitted and run.'
        print name + ' [Node name] : Plot the mean firing rate against time for the given node.'

def densityMovie(command, current_sim):
    command_name = command[0]
    name = 'generate-density-movie'
    alts = ['gdm']

    if command_name in [name] + alts:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 5:
            print 'Warning : This take a *long* time to complete and use *large* amounts of disk space.'
            current_density = api.Density(current_sim, command[1])
            current_density.generateDensityAnimation(command[4], int(command[2]),
                                      True,
                                      float(command[3]))
        else:
            print name + ' expects four parameters.'
            densityMovie(name+'?', current_sim)

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print 'Warning : This command can take a *long* time to complete and use *large* amounts of disk space.'
        print name + ' [Node name] [Frame size] [Time scale] [Movie filename] : Generate all 2D density plot images for the given node with the given frame size. From the images, generate a movie with the given time scale (eg 0.5 = twice as fast, 10 = ten times slower).'
        print 'Alternative command names : ' + ' '.join(alts)

def plotDensity(command, current_sim):
    command_name = command[0]
    name = 'plot-density'
    alts = ['pd']

    if command_name in [name] + alts:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 3:
            current_density = api.Density(current_sim, command[1])
            filename = current_density.findDensityFileFromTime(command[2])
            print 'Plotting ' + filename + '.'

            fig, axis = plt.subplots()
            current_density.plotDensity(filename, ax=axis)
            plt.show(block=False)
        else:
            print name + ' expects two parameters.'
            plotDensity(name+'?', current_sim)

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Node name] [Time] : Plot the 2D density of the given node at the given time. The time must be between the begin and end time of the current simulation and be a multiple of the report time (t_report).'
        print 'Alternative command names : ' + ' '.join(alts)

def marginalMovie(command, current_sim):
    command_name = command[0]
    name = 'generate-marginal-movie'
    alts = ['gmm']

    if command_name in [name] + alts:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 5:
            current_marginal = api.Marginal(current_sim, command[1])
            current_marginal.generateMarginalAnimation(command[4], int(command[2]),
                                      True,
                                      float(command[3]))
        elif len(command) == 7:
            current_marginal = api.Marginal(current_sim, command[1],
                                     int(command[2]), int(command[3]))
            current_marginal.generateMarginalAnimation(command[6], int(command[4]),
                                      True,
                                      float(command[5]))
        else:
            print name + ' expects four or six parameters.'
            marginalMovie(name+'?', current_sim)

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print 'Warning : This command can take a *long* time to complete and use *large* amounts of disk space.'
        print name + ' [Node name] [Frame size] [Time scale] [Movie filename] : Generate all marginal density plot images for the given node with the given frame size. From the images, generate a movie with the given time scale (eg 0.5 = twice as fast, 10 = ten times slower).'
        print name + ' [Node name] [Number V bins] [Number W bins] [Frame size] [Time scale] [Movie filename] : Generate all marginal density plot images (using the provided number of bins for each dimension) for the given node with the given frame size. '
        print 'Alternative command names : ' + ' '.join(alts)

def plotMarginals(command, current_sim):
    command_name = command[0]
    name = 'plot-marginals'
    alts = ['pm']

    if command_name in [name]:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 3:
            current_marginal = api.Marginal(current_sim, command[1])
            fig, axis = plt.subplots(1,2)
            current_marginal.plotV(float(command[2]), axis[0])
            current_marginal.plotW(float(command[2]), axis[1])
            plt.show(block=False)
        elif len(command) == 5:
            current_marginal = api.Marginal(current_sim, command[1],
                                     int(command[2]), int(command[3]))
            fig, axis = plt.subplots(1,2)
            current_marginal.plotV(float(command[4]), axis[0])
            current_marginal.plotW(float(command[4]), axis[1])
            plt.show(block=False)
        else:
            print name + ' expects two or four parameters.'
            plotMarginals(name+'?', current_sim)

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Node name] [Time] : Plot the marginal densities of the given node at the given time. The time must be between the begin and end time of the current simulation and be a multiple of the report time (t_report).'
        print name + ' [Node name] [Number V bins] [Number W bins] [Time] : Plot the marginal densities (using the provided number of bins for each dimension) of the given node at the given time.'
        print 'Alternative command names : ' + ' '.join(alts)

def generateLifMesh(command):
    command_name = command[0]
    name = 'generate-lif-mesh'

    if command_name in [name]:
        if len(command) == 2:
            gen = api.LifMeshGenerator(command[1])
            gen.generateLifMesh()
            gen.generateLifStationary()
            gen.generateLifReversal()
        else:
            print name + ' expects one parameter.'
            generateLifMesh(name+'?')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Basename] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Leaky Integrate and Fire Neuron.'

def generateModel(command):
    command_name = command[0]
    name = 'generate-model'

    if command_name in [name]:
        if len(command) == 4:
            api.MeshTools.buildModelFileFromMesh(command[1],
                                  float(command[2]), float(command[3]))
        else:
            print name + ' expects three parameters.'
            generateModel(name+'?')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Basename] [Reset Value] [Threshold] : Generate a new Basename.model with the given Reset and Threshold values. Requires valid Basename.mesh, Basename.stat and Basename.rev files in the current working directory.'

def generateEmptyFid(command):
    command_name = command[0]
    name = 'generate-empty-fid'
    alts = ['fid']

    if command_name in [name] + alts:
        if len(command) == 2:
            api.MeshTools.generateStubFidFile(command[1])
        else:
            print name + ' expects one parameter.'
            generateEmptyFid(name+'?')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Basename] : Generate an empty stub Basename.fid.'
        print 'Alternative command names : ' + ' '.join(alts)

def generateMatrix(command):
    command_name = command[0]
    name = 'generate-matrix'

    if command_name in [name]:
        if len(command) == 4:
            api.MeshTools.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[1] + '.fid', num_mc_points=int(command[3]))
        elif len(command) == 5:
            api.MeshTools.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[1] + '.fid', num_mc_points=int(command[3]), spike_shift_w=float(command[4]))
        elif len(command) == 6:
            api.MeshTools.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[1] + '.fid', num_mc_points=int(command[3]), spike_shift_w=float(command[4]), reset_shift_w=float(command[5]))
        else:
            print name + ' expects three, four or five parameters.'
            generateMatrix(name+'?')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Basename] [Spike V-efficacy] [Number of points] : Generate a Basename.mat file (and Basename.lost file) with the given spike shift in the V direction using a Monte Carlo method with the given number of points per cell. Expects a valid Basename.model and Basename.fid file in the current working directory.'
        print name + ' [Basename] [Spike V-efficacy] [Number of points] [Spike W-efficacy] : Generate a Basename.mat file (and Basename.lost file) with the given spike shift in the V and W directions.'
        print name + ' [Basename] [Spike V-efficacy] [Number of points] [Spike W-efficacy] [Reset W shift]: Generate a Basename.mat file (and Basename.lost file) with the given spike shift in the V and W directions and given W shift during reset.'

def lost(command):
    command_name = command[0]
    name = 'fix-lost'

    if command_name in [name]:
        if len(command) == 2:
            api.MeshTools.plotLost(command[1])
        else:
            print name + ' expects one parameter.'
            lost(name+'?')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Filename] : Open the .lost file to allow drawing of additional Fiducial areas. Requires an associated .fid file in the current working directory.'

def drawMesh(command):
    command_name = command[0]
    name = 'draw-mesh'

    if command_name in [name]:
        if len(command) == 2:
            api.MeshTools.plotMesh(command[1])
        else:
            print name + ' expects one parameter.'
            lost(name+'?')

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' [Filename] : Display the mesh in the given mesh file.'


def isQuitCommand(command):
    command_name = command[0]
    name = 'quit'
    alts = ['exit', 'close']

    if command_name in [name]+alts:
        return True

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print name + ' : Quit the UI.'
        print 'Alternative command names : ' + ' '.join(alts)

    return False

if __name__ == "__main__":
  current_sim = None

  settings = {}
  settings['mpi_enabled'] = False
  settings['openmp_enabled'] = False
  settings['root_enabled'] = True

  settingsfilename = op.join(getMiindPythonPath(), 'miind_ui_settings')
  if not op.exists(settingsfilename):
      with open(settingsfilename, 'w') as settingsfile:
          for k,v in settings.iteritems():
              settingsfile.write(k + '=' + str(v) + '\n')
  else:
      with open(settingsfilename, 'r') as settingsfile:
          for line in settingsfile:
              tokens = line.split('=')
              settings[tokens[0].strip()] = (tokens[1].strip() == 'True')

  while True:
      command_string = raw_input('> ')
      command = command_string.split(' ')
      command_name = command[0]
    #   try:
      _help(command)

      current_sim = sim(command, current_sim)

      models(command, current_sim)

      _settings(command)

      submit(command, current_sim)

      run(command, current_sim)

      buildSharedLib(command, current_sim)

      rate(command, current_sim)

      densityMovie(command, current_sim)

      plotDensity(command, current_sim)

      marginalMovie(command, current_sim)

      plotMarginals(command, current_sim)

      generateLifMesh(command)

      generateModel(command)

      generateEmptyFid(command)

      generateMatrix(command)

      drawMesh(command)

      lost(command)

      if isQuitCommand(command):
        break

    #   except BaseException as e:
    #       print e
    #       continue
