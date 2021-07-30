#!/usr/bin/env python3

#import matplotlib
#matplotlib.use('Agg')

import argparse
import miind.codegen3 as codegen3
import sys
import glob
import os
import os.path as op
import miind.miind_api as api
import matplotlib.pyplot as plt
import miind.directories3 as directories3

class MiindIO:
    def __init__(self):
        self.cwdfilename = 'miind_cwd'
        self.settingsfilename = op.expanduser('~/.miind_settings')
        self.available_settingsfilename = os.path.join(directories3.miind_python_dir(),'miind_settings')
        self.debug = False
        self.cwd_settings = {}
        self.settings = {}
        self.available_settings = {}
        self.c_compiler = None
        self.cxx_compiler = None
        self.single_shot_command = False # If miindio is being called with just a single command (not using the CLI)

    def getMiindPythonPath(self):
        return os.path.join(directories3.miind_python_dir())

    def _help(self,command):
        command_name = command[0]
        name = 'help'
        alts = ['h', '?', 'commands', 'list']

        if command_name in [name] + alts:
            print('')
            print('MIIND UI')
            print('')
            print('(To debug errors, call miindio.py -debug for the full python stack trace.)')
            print('')
            print('For more information on any command type the command name and a \'?\' [eg. sim?]')
            print('')
            print('help                     : Get this help menu.')
            print('quit                     : Close the UI.')
            print('')
            print('***** Commands for Creating and Running Simulations *****')
            print('')
            print('sim                      : Set the current simulation from an xml file or generate a new xml file.')
            print('models                   : List all model files used by the current simulation.')
            print('settings                 : Set certain persistent parameters to match your MIIND installation (ENABLE ROOT, CUDA).')
            print('submit                   : Generate and build (make) the code from the current simulation.')
            print('run                      : Run the current submitted simulation.')
            print('submit-python            : Generate and build (make) a shared library for use with python from the current simulation.')
            print('')
            print('***** Commands for Analysing and Presenting Completed Simulations *****')
            print('')
            print('rate                     : Plot the mean firing rate of a given node in the current simulation.')
            print('avgv                     : Plot the mean membrane potential of a given node in the current simulation.')
            print('plot-density             : Plot the 2D density of a given node at a given time in the simulation.')
            print('plot-marginals           : Plot the marginal densities of a given node at a given time in the simulation.')
            print('generate-density-movie   : Generate a movie of the 2D density for a given node in the simulation.')
            print('generate-marginal-movie  : Generate a movie of the Marginal densities for a given node in the simulation.')
            print('')
            print('***** Commands for Building New Models and Matrix Files *****')
            print('')
            print('generate-model           : Generate a model file from existing mesh, rev and stat files.')
            print('generate-empty-fid       : Generate a stub .fid file.')
            print('generate-matrix          : Generate a matrix file from existing model and fid files.')
            print('regenerate-reset         : Regenerate the reset mapping for an existing model.')
            print('lost                     : Open the fiducial tool for capturing lost points.')
            print('generate-lif-mesh        : Helper command to build a LIF neuron mesh.')
            print('generate-qif-mesh        : Helper command to build a QIF neuron mesh.')
            print('generate-eif-mesh        : Helper command to build a EIF neuron mesh.')
            print('draw-mesh                : Draw the mesh described in an existing .mesh file.')
            print('')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : Display the available commands.')

    def sim(self,command, current_sim):
        command_name = command[0]
        name = 'sim'

        if command_name in [name]:
            if len(command) == 1:
                if not current_sim:
                    print('No simulation currently defined.')
                    print('')
                    self.sim([name+'?'], None)
                    return
                print('Original XML File : {}'.format(current_sim.original_xml_path))
                print('Project Name : {}'.format(current_sim.submit_name))
                print('Parameters : {}'.format(current_sim.parameters))
                print('Output Directory : {}'.format(current_sim.getOutputDirectory()))
                print('Variables :')
                for name in current_sim.variablenames:
                    print('   ' + name)
                print('Nodes :')
                for (name,_) in current_sim.nodenames:
                    if name in [n for (n,m) in current_sim.meshnodenames]:
                        print('   ' + name + ' (Mesh Node)')
                    else:
                        print('   ' + name)
                print('')
            if len(command) == 2:
                current_sim = api.MiindSimulation(command[1])

                self.cwd_settings['sim'] = current_sim.xml_fname
                self.cwd_settings['sim_params'] = {}

                with open(self.cwdfilename, 'w') as settingsfile:
                    settingsfile.write('sim=' + str(self.cwd_settings['sim']) + '\n')
                    settingsfile.write('sim_params=\n')
            if len(command) >= 3:
                comm_dict = {}
                for comm in command[2:]:
                    kv = comm.split('=')
                    comm_dict[kv[0]] = kv[1]
                current_sim = api.MiindSimulation(command[1], **comm_dict)

                self.cwd_settings['sim'] = current_sim.xml_fname
                self.cwd_settings['sim_params'] = current_sim.parameters

                with open(self.cwdfilename, 'w') as settingsfile:
                    settingsfile.write('sim=' + str(self.cwd_settings['sim']) + '\n')
                    param_string = ""
                    for k,v in self.cwd_settings['sim_params'].items():
                      param_string = param_string + k + ',' + v + ','
                    settingsfile.write('sim_params=' + param_string + '\n')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : Provide information on the current simulation.')
            print (name + ' [XML filename] : Use this XML file for the current simulation. The default project name is the same as the XML filename.')
            print (name + ' [XML filename] [Parameter 1] [Parameter 2] ... : Use this XML file with the Variable objects set using the given Parameters.')

        return current_sim

    def models(self,command, current_sim):
        command_name = command[0]
        name = 'models'

        if command_name in [name]:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            print('Model files used in ' + current_sim.submit_name + ' :\n')
            for mf in current_sim.modelfiles:
                print (mf)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print(name + ' : List all model files used by the current simulation.')

    def _settings(self,command):
        command_name = command[0]
        name = 'settings'

        if command_name in [name]:
            if len(command) == 1:
                print('Current Settings :\n')
                for k,v in self.settings.items():
                    print( k + ' = ' + str(v))

            if len(command) > 1:
                if len(command) != 3:
                    print ("settings command expects [ENABLE_ROOT] [ENABLE_CUDA].")
                else:
                    if (command[1] in ['True', 'true', 'TRUE', 'ON', 'on']):
                        if self.available_settings['root_enabled']:
                            self.settings['root_enabled'] = True
                        else:
                            print('ROOT was not enabled in the MIIND installation and cannot be set.')
                    else:
                        self.settings['root_enabled'] = False

                    if (command[2] in ['True', 'true', 'TRUE', 'ON', 'on']):
                        if self.available_settings['cuda_enabled']:
                            self.settings['cuda_enabled'] = True
                        else:
                            print('CUDA was not enabled in the MIIND installation and cannot be set.')
                    else:
                        self.settings['cuda_enabled'] = False

                    with open(self.settingsfilename, 'w') as settingsfile:
                        for k,v in self.settings.items():
                            if v:
                                settingsfile.write(k + '=ON\n')
                            else:
                                settingsfile.write(k + '=OFF\n')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : List the current settings. Settings are stored in \'.miind_settings\' in your home (~/) directory.')
            print (name + ' [ENABLE_ROOT] [ENABLE_CUDA]: Expects \'True\' or \'False\' for each of the settings. Settings are persistent.')


    def submit(self,command, current_sim):
        command_name = command[0]
        name = 'submit'

        if command_name in [name]:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            cmake_args = []
            if self.c_compiler and self.cxx_compiler:
                cmake_args = ['-DCMAKE_self.c_compiler={}'.format(self.c_compiler), '-DCMAKE_self.cxx_compiler={}'.format(self.cxx_compiler)]

            if len(command) == 1:
                current_sim.submit(True, [],
                      self.available_settings['mpi_enabled'], self.available_settings['openmp_enabled'], self.settings['root_enabled'], self.settings['cuda_enabled'],*cmake_args)
            if len(command) == 2:
                current_sim.submit(True, glob.glob(command[1]),
                      self.available_settings['mpi_enabled'], self.available_settings['openmp_enabled'], self.settings['root_enabled'], self.settings['cuda_enabled'],*cmake_args)
            if len(command) >= 3:
                current_sim.submit(True, command[1:],
                      self.available_settings['mpi_enabled'], self.available_settings['openmp_enabled'], self.settings['root_enabled'], self.settings['cuda_enabled'],*cmake_args)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : Generate and \'make\' the code from the current simulation xml file. Ensure you have the correct settings (call \'settings\').')
            print (name + ' [xml file] : Generate and \'make\' the code from the given xml file (or xml files matching the a given regular expression).')
            print (name + ' [xml file 1] [xml file 2] [xml file 3] ... : Generate and \'make\' the code from the given xml files.')

    def run(self,command, current_sim):
        command_name = command[0]
        name = 'run'

        if command_name in [name]:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 1:
                current_sim.run()
            if len(command) == 2:
                current_sim.run_mpi(int(command[1]))

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : Run the current simulation. The simulation must have previously been \'submitted\'.')
            print (name + ' [Number of cores] : Run the current simulation using mpiexec with the given number of cores. MPI must be enabled in MIIND.')

    def buildSharedLib(self,command, current_sim):
        command_name = command[0]
        name = 'submit-python'
        alts = []

        if command_name in [name] + alts:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            cmake_args = []
            if self.c_compiler and self.cxx_compiler:
                cmake_args = ['-DCMAKE_self.c_compiler={}'.format(self.c_compiler), '-DCMAKE_self.cxx_compiler={}'.format(self.cxx_compiler)]

            if len(command) == 1:
                current_sim.submit_shared_lib(True, [],
                      self.available_settings['mpi_enabled'], self.available_settings['openmp_enabled'], self.settings['root_enabled'], self.settings['cuda_enabled'],*cmake_args)
            if len(command) == 2:
                current_sim.submit_shared_lib(True, glob.glob(command[1]),
                      self.available_settings['mpi_enabled'], self.available_settings['openmp_enabled'], self.settings['root_enabled'], self.settings['cuda_enabled'],*cmake_args)
            if len(command) >= 3:
                current_sim.submit(True, command[1:],
                      self.available_settings['mpi_enabled'], self.available_settings['openmp_enabled'], self.settings['root_enabled'], self.settings['cuda_enabled'],*cmake_args)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : Generate and \'make\' a shared library for use with python from the current simulation xml file. Ensure you have the correct settings (call \'settings\').')
            print (name + ' [xml file] : Generate and \'make\' the code from the given xml file (or xml files matching the a given regular expression).')
            print (name + ' [xml file 1] [xml file 2] [xml file 3] ... : Generate and \'make\' the code from the given xml files.')

    def rate(self,command, current_sim):
        command_name = command[0]
        name = 'rate'

        if command_name in [name]:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 1:
                print('The following nodes can be queried for a mean firing rate :')
                for (name,_) in current_sim.nodenames:
                    print (str(current_sim.getIndexFromNode(name)) + ' : ' + name)
            if len(command) == 2:
                current_sim.plotRate(command[1], wait_on_show=self.single_shot_command)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : List the nodes for which a rate plot is available in the current simulation. The current simulation must have been submitted and run.')
            print (name + ' [Node name] : Plot the mean firing rate against time for the given node.')

    def avgv(self,command, current_sim):
        command_name = command[0]
        name = 'avgv'

        if command_name in [name]:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 1:
                print('The following nodes can be queried for a mean membrane potential :')
                for (name,_) in current_sim.nodenames:
                    print (str(current_sim.getIndexFromNode(name)) + ' : ' + name)
            if len(command) == 2:
                current_sim.plotAvgV(command[1], wait_on_show=self.single_shot_command)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : List the nodes for which an average membrane potentaial plot is available in the current simulation. The current simulation must have been submitted and run.')
            print (name + ' [Node name] : Plot the mean membrane potential against time for the given node.')

    def densityMovie(self,command, current_sim):
        command_name = command[0]
        name = 'generate-density-movie'
        alts = ['gdm']

        if command_name in [name] + alts:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 5:
                print('Warning : This take a *long* time to complete and use *large* amounts of disk space.')
                current_density = current_sim.getDensityByNodeName(command[1])
                current_density.generateDensityAnimation(command[4], int(command[2]),
                                          True,
                                          float(command[3]))
            else:
                print (name + ' expects four parameters.')
                self.densityMovie(name+'?', current_sim)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print('Warning : This command can take a *long* time to complete and use *large* amounts of disk space.')
            print (name + ' [Node name] [Frame size] [Time step] [Movie filename] : Generate all 2D density plot images for the given node with the given frame size. From the images, generate a movie with the given time step.')
            print('Alternative command names : ' + ' '.join(alts))

    def plotDensity(self,command, current_sim):
        command_name = command[0]
        name = 'plot-density'
        alts = ['pd']

        if command_name in [name] + alts:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 3:
                current_density = current_sim.getDensityByNodeName(command[1])
                filename = current_density.findDensityFileFromTime(command[2])
                print('Plotting ' + filename + '.')

                fig, axis = plt.subplots()
                current_density.plotDensity(filename, ax=axis)
                plt.show(block=self.single_shot_command)
            else:
                print (name + ' expects two parameters.')
                self.plotDensity(name+'?', current_sim)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print( name + ' [Node name] [Time] : Plot the 2D density of the given node at the given time. The time must be between the begin and end time of the current simulation and be a multiple of the report time (t_report).')
            print('Alternative command names : ' + ' '.join(alts))

    def marginalMovie(self,command, current_sim):
        command_name = command[0]
        name = 'generate-marginal-movie'
        alts = ['gmm']

        if command_name in [name] + alts:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 5:
                current_marginal = current_sim.getMarginalByNodeName(command[1])
                current_marginal.generateMarginalAnimation(command[4], int(command[2]),
                                          True,
                                          float(command[3]))
            elif len(command) == 7:
                current_marginal = current_sim.getMarginalByNodeName(command[1])
                current_marginal.vn = int(command[2])
                current_marginal.wn = int(command[3])
                current_marginal.generateMarginalAnimation(command[6], int(command[4]),
                                          True,
                                          float(command[5]))
            else:
                print( name + ' expects four or six parameters.')
                self.marginalMovie(name+'?', current_sim)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print('Warning : This command can take a *long* time to complete and use *large* amounts of disk space.')
            print (name + ' [Node name] [Frame size] [Time scale] [Movie filename] : Generate all marginal density plot images for the given node with the given frame size. From the images, generate a movie with the given time scale (eg 0.5 = twice as fast, 10 = ten times slower).')
            print (name + ' [Node name] [Number V bins] [Number W bins] [Frame size] [Time scale] [Movie filename] : Generate all marginal density plot images (using the provided number of bins for each dimension) for the given node with the given frame size. ')
            print('Alternative command names : ' + ' '.join(alts))

    def plotMarginals(self,command, current_sim):
        command_name = command[0]
        name = 'plot-marginals'
        alts = ['pm']

        if command_name in [name]:
            if not current_sim:
                print('No simulation currently defined. Please call command \'sim\'.')

            if len(command) == 3:
                current_marginal = current_sim.getMarginalByNodeName(command[1])
                fig, axis = plt.subplots(1,2)
                current_marginal.plotV(command[2], axis[0])
                current_marginal.plotW(command[2], axis[1])
                plt.show(block=self.single_shot_command)
            elif len(command) == 5:
                current_marginal = current_sim.getMarginalByNodeName(command[1])
                current_marginal.vn = int(command[2])
                current_marginal.wn = int(command[3])
                fig, axis = plt.subplots(1,2)
                current_marginal.plotV(command[4], axis[0])
                current_marginal.plotW(command[4], axis[1])
                plt.show(block=self.single_shot_command)
            else:
                print (name + ' expects two or four parameters.')
                self.plotMarginals(name+'?', current_sim)

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Node name] [Time] : Plot the marginal densities of the given node at the given time. The time must be between the begin and end time of the current simulation and be a multiple of the report time (t_report).')
            print (name + ' [Node name] [Number V bins] [Number W bins] [Time] : Plot the marginal densities (using the provided number of bins for each dimension) of the given node at the given time.')
            print('Alternative command names : ' + ' '.join(alts))

    def generateLifMesh(self,command):
        command_name = command[0]
        name = 'generate-lif-mesh'

        if command_name in [name]:
            if len(command) == 2:
                gen = api.LifMeshGenerator(command[1])
                gen.generateLifMesh()
                gen.generateLifStationary()
                gen.generateLifReversal()
            elif len(command) == 8:
                gen = api.LifMeshGenerator(command[1], float(command[2]), float(command[3]), float(command[4]), float(command[5]), float(command[6]), int(command[7]))
                gen.generateLifMesh()
                gen.generateLifStationary()
                gen.generateLifReversal()
            else:
                print (name + ' expects one or seven parameters.')
                self.generateLifMesh(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Leaky Integrate and Fire Neuron.')
            print ('Defaults : Time Scale = 10e-3, Threshold Potential = -50.0, Resting Potential = -65.0, Min Potential = -80.0, Time step = 0.0001, Bin Count = 300')
            print (name + ' [Basename] [Time Scale] [Threshold Potential] [Resting Potential] [Min Potential] [Time step] [Bin Count] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Leaky Integrate and Fire Neuron.')

    def generateQifMesh(self,command):
        command_name = command[0]
        name = 'generate-qif-mesh'

        if command_name in [name]:
            if len(command) == 2:
                gen = api.QifMeshGenerator(command[1])
                gen.generateQifMesh()
                gen.generateQifStationary()
                gen.generateQifReversal()
            elif len(command) == 3:
                if (float(command[2]) == 0.0):
                    print ('An I value of 0.0 is not allowed for this mesh. Use a very small epsilon if no current is required.')
                else:
                    gen = api.QifMeshGenerator(command[1], I=float(command[2]))
                    gen.generateQifMesh()
                    gen.generateQifStationary()
                    gen.generateQifReversal()
            elif len(command) == 7:
                if (float(command[6]) == 0.0):
                    print ('An I value of 0.0 is not allowed for this mesh. Use a very small epsilon if no current is required.')
                else:
                    gen = api.QifMeshGenerator(command[1], float(command[2]), float(command[3]), float(command[4]), float(command[5]), float(command[6]))
                    gen.generateQifMesh()
                    gen.generateQifStationary()
                    gen.generateQifReversal()
            else:
                print (name + ' expects one, two or six parameters.')
                self.generateQifMesh(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Quadratic Integrate and Fire Neuron (with constant input current I=1.0).')
            print (name + ' [Basename] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Quadratic Integrate and Fire Neuron (with constant input current I=-1.0).')
            print ('Defaults : Time Scale = 10e-3, Min Potential = -10.0, Max Potential = 10.0, Time step = 0.0001')
            print (name + ' [Basename] [Time Scale] [Min Potential] [Max Potential] [Time step] [I (Non-zero Current)] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Quadratic Integrate and Fire Neuron.')
            
    def generateEifMesh(self,command):
        command_name = command[0]
        name = 'generate-eif-mesh'

        if command_name in [name]:
            if len(command) == 2:
                gen = api.EifMeshGenerator(command[1])
                gen.generateEifMesh()
                gen.generateEifStationary()
                gen.generateEifReversal()
            elif len(command) == 10:
                gen = api.EifMeshGenerator(command[1], float(command[2]), float(command[3]), float(command[4]), float(command[5]), float(command[6]), int(command[7]))
                gen.generateEifMesh()
                gen.generateEifStationary()
                gen.generateEifReversal()
            else:
                print (name + ' expects one or nine parameters.')
                self.generateEifMesh(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for an Exponential Integrate and Fire Neuron.')
            print ('Defaults : Leak Conductance (g_l) = 0.3, Resting Potential (v_l) = -70.0, Threshold Potential (v_th) = -56, Delta t (delta_t) = 1.48, Min Potential (V_min) = -90.0, Max Potential (V_max) = -51.5, Time step = 0.1, Epsilon = 0.01')
            print (name + ' [Basename] [Leak Conductance] [Resting Potential] [Threshold Potential] [Delta t] [Min Potential] [Max Potential] [Time step] [Epsilon] : Generate a new Basename.mesh, Basename.stat and Basename.rev file for a Leaky Integrate and Fire Neuron.')


    def generateModel(self,command):
        command_name = command[0]
        name = 'generate-model'

        if command_name in [name]:
            if len(command) == 4:
                api.MeshTools.buildModelFileFromMesh(command[1],
                                      float(command[2]), float(command[3]))
            else:
                print (name + ' expects three parameters.')
                self.generateModel(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] [Reset Value] [Threshold] : Generate a new Basename.model with the given Reset and Threshold values. Requires valid Basename.mesh, Basename.stat and Basename.rev files in the current working directory.')

    def generateEmptyFid(self,command):
        command_name = command[0]
        name = 'generate-empty-fid'
        alts = ['fid']

        if command_name in [name] + alts:
            if len(command) == 2:
                api.MeshTools.generateStubFidFile(command[1])
            else:
                print (name + ' expects one parameter.')
                self.generateEmptyFid(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] : Generate an empty stub Basename.fid.')
            print('Alternative command names : ' + ' '.join(alts))

    def generateTransform(self,command):
        command_name = command[0]
        name = 'generate-transform'

        if command_name in [name]:
            if len(command) == 3:
                    api.MeshTools.buildTransformFileFromModel(command[1], num_mc_points=int(command[2]))
            else:
                print (name + ' expects two parameters.')
                self.generateTransform(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] [Number of points] : Generate a Basename.tmat file. Expects two model files - Basename.model and Basename_transform.model.')

    def generateMatrix(self,command):
        command_name = command[0]
        name = 'generate-matrix'

        if command_name in [name]:
            if len(command) == 4:
                if (len(command[3].split('.')) > 1):
                    api.MeshTools.buildMatrixFileFromModel(command[1], 0.1, fidfile=command[1] + '.fid', num_mc_points=int(command[2]), jump_file=command[3])
                else:
                    api.MeshTools.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[1] + '.fid', num_mc_points=int(command[3]))
            elif len(command) == 5:
                api.MeshTools.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[1] + '.fid', num_mc_points=int(command[3]), spike_shift_w=float(command[4]))
            elif len(command) == 7:
                api.MeshTools.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[1] + '.fid', num_mc_points=int(command[3]), spike_shift_w=float(command[4]), reset_shift_w=float(command[5]), use_area_calculation=(command[6] in ['True', 'true', 'TRUE']))
            else:
                print (name + ' expects three, four or six parameters.')
                self.generateMatrix(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] [Spike V-efficacy] [Number of points] : Generate a Basename.mat file (and Basename.lost file) with the given spike shift in the V direction using a Monte Carlo method with the given number of points per cell. Expects a valid Basename.model and Basename.fid file in the current working directory.')
            print (name + ' [Basename] [Number of points] [Jump File Name] : Generate a Basename.mat file (and Basename.lost file) using the given jump file.')
            print (name + ' [Basename] [Spike V-efficacy] [Number of points] [Spike W-efficacy] : Generate a Basename.mat file (and Basename.lost file) with the given spike shift in the V and W directions.')
            print (name + ' [Basename] [Spike V-efficacy] [Number of points] [Spike W-efficacy] [Reset W shift] [Use Area Calculation (True/False)]: Generate a Basename.mat file (and Basename.lost file) with the given spike shift in the V and W directions and given W shift during reset.')

    def regenerateModelReset(self,command):
        command_name = command[0]
        name = 'regenerate-reset'

        if command_name in [name]:

            if len(command) == 3:
                api.MeshTools.buildMatrixFileFromModel(command[1], 0.0, reset_shift_w=float(command[2]), mode='reset')
            else:
                print (name + ' expects two parameters.')
                self.regenerateModelReset(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] [Reset W shift] : Regenerate the reset mapping for Basename.model.')

    def regenerateModelResetTransform(self,command):
        command_name = command[0]
        name = 'regenerate-transform-reset'

        if command_name in [name]:

            if len(command) == 3:
                api.MeshTools.buildTransformFileFromModel(command[1], reset_shift_w=float(command[2]), mode='resettransform')
            else:
                print (name + ' expects two parameters.')
                self.regenerateModelResetTransform(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Basename] [Reset W shift] : Regenerate the reset mapping for Basename.model.')

    def lost(self,command):
        command_name = command[0]
        name = 'lost'

        if command_name in [name]:
            if len(command) == 2:
                api.MeshTools.plotLost(command[1])
            else:
                print (name + ' expects one parameter.')
                self.lost(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Filename] : Open the .lost file to allow drawing of additional Fiducial areas. Requires an associated .fid file in the current working directory.')

    def drawMesh(self,command):
        command_name = command[0]
        name = 'draw-mesh'

        if command_name in [name]:
            if len(command) == 2:
                api.MeshTools.plotMesh(command[1])
            else:
                print (name + ' expects one parameter.')
                self.drawMesh(name+'?')

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' [Filename] : Display the mesh in the given mesh file.')


    def isQuitCommand(self,command):
        command_name = command[0]
        name = 'quit'
        alts = ['exit', 'close']

        if command_name in [name]+alts:
            return True

        if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
            print (name + ' : Quit the UI.')
            print('Alternative command names : ' + ' '.join(alts))

        return False

    def checkCommands(self,command, current_sim):
        self._help(command)

        current_sim = self.sim(command, current_sim)

        self.models(command, current_sim)

        self._settings(command)

        self.submit(command, current_sim)

        self.run(command, current_sim)

        self.buildSharedLib(command, current_sim)

        self.rate(command, current_sim)

        self.avgv(command, current_sim)

        self.densityMovie(command, current_sim)

        self.plotDensity(command, current_sim)

        self.marginalMovie(command, current_sim)

        self.plotMarginals(command, current_sim)

        self.generateLifMesh(command)

        self.generateQifMesh(command)
        
        self.generateEifMesh(command)

        self.generateModel(command)

        self.generateEmptyFid(command)

        self.generateMatrix(command)

        self.generateTransform(command)

        self.regenerateModelReset(command)

        self.regenerateModelResetTransform(command)

        self.drawMesh(command)

        self.lost(command)

        return current_sim

    def loadMiindSettings(self):
        if op.exists(self.available_settingsfilename):
          # Read available settings from MIIND installation.
          with open(self.available_settingsfilename, 'r') as settingsfile:
              for line in settingsfile:
                  tokens = line.split('=')
                  if tokens[0].strip() == 'self.c_compiler':
                      self.c_compiler = tokens[1].strip()
                  elif tokens[0].strip() == 'self.cxx_compiler':
                      self.cxx_compiler = tokens[1].strip()
                  else:
                      self.available_settings[tokens[0].strip()] = (tokens[1].strip() in ['YES','Y','ON','1','TRUE'])

          # Read or create settings as long as they're available in the installation.
          if not op.exists(self.settingsfilename):
              print('Settings file ('+ self.settingsfilename +') created. Using defaults from MIIND installation:')
              print('ROOT ENABLED = ' + str(self.available_settings['root_enabled']) + '')
              print('CUDA ENABLED = ' + str(self.available_settings['cuda_enabled']) + '\n')

              self.settings['root_enabled'] = self.available_settings['root_enabled']
              self.settings['cuda_enabled'] = self.available_settings['cuda_enabled']

              with open(self.settingsfilename, 'w') as settingsfile:
                  for k,v in self.settings.items():
                      if v:
                          settingsfile.write(k + '=ON\n')
                      else:
                          settingsfile.write(k + '=OFF\n')
          else:
              with open(self.settingsfilename, 'r') as settingsfile:
                  for line in settingsfile:
                      tokens = line.split('=')
                      self.settings[tokens[0].strip()] = (tokens[1].strip() == 'ON')

              # Verify settings.
              self.settings['root_enabled'] = self.available_settings['root_enabled'] and self.settings['root_enabled']
              self.settings['cuda_enabled'] = self.available_settings['cuda_enabled'] and self.settings['cuda_enabled']
        else:
          print('WARNING : MIIND installation is missing the available settings file ' +  + '. All settings switched OFF.')

          # Read or create settings as long as they're available in the installation.
          if not op.exists(self.settingsfilename):
              print('Settings file ('+ self.settingsfilename +') created. All settings disabled.\n')
              self.settings['root_enabled'] = False
              self.settings['cuda_enabled'] = False

              with open(self.settingsfilename, 'w') as settingsfile:
                  for k,v in self.settings.items():
                      if v:
                          settingsfile.write(k + '=ON\n')
                      else:
                          settingsfile.write(k + '=OFF\n')

          else:
              with open(self.settingsfilename, 'r') as settingsfile:
                  for line in settingsfile:
                      tokens = line.split('=')
                      settings[tokens[0].strip()] = (tokens[1].strip() == 'ON')
        
    def main(self):
        self.available_settings['mpi_enabled'] = False
        self.available_settings['openmp_enabled'] = False
        self.available_settings['root_enabled'] = False
        self.available_settings['cuda_enabled'] = False

        self.settings['root_enabled'] = False
        self.settings['cuda_enabled'] = False

        self.cwd_settings['sim'] = 'NOT_SET'
        self.cwd_settings['sim_params'] = {}

        self.loadMiindSettings()

        if not op.exists(self.cwdfilename):
          with open(self.cwdfilename, 'w') as cwdsettingsfile:
              cwdsettingsfile.write('sim=NOT_SET\n')
              cwdsettingsfile.write('sim_params=\n')
        else:
          with open(self.cwdfilename, 'r') as cwdsettingsfile:
              for line in cwdsettingsfile:
                  tokens = line.split('=')
                  # Expect sim to be a filename, otherwise expect a boolean
                  if tokens[0].strip() == 'sim':
                       if tokens[1].strip() == 'NOT_SET':
                           self.cwd_settings['sim'] = None
                       else:
                           self.cwd_settings['sim'] = tokens[1].strip()
                  elif tokens[0].strip() == 'sim_params':
                       if tokens[1].strip() == '':
                           self.cwd_settings['sim_params'] = {}
                       else:
                           param_string = tokens[1].strip()
                           params = tokens[1].strip().split(',')
                           param_keys = params[::2]
                           param_vals = params[1::2]
                           param_dict = {}
                           if len(param_keys) > 0:
                               for p in range(len(param_keys)-1):
                                  param_dict[param_keys[p]] = param_vals[p]
                           self.cwd_settings['sim_params'] = param_dict

        current_sim = None
        if self.cwd_settings['sim'] != None:
          if self.cwd_settings['sim'] != 'NOT_SET':
              
              current_sim = api.MiindSimulation(self.cwd_settings['sim'], **self.cwd_settings['sim_params'])
              print('**** Current Simulation Details ****\n')
              self.sim(['sim'], current_sim)
          else:
              print('\nWARNING : No Simulation currently set. Please call \'sim\' to set it. \n')
              current_sim = None
        else:
          print('\nWARNING : No Simulation currently set. Please call \'sim\' to set it. \n')
          current_sim = None

        if len(sys.argv) > 1 and sys.argv[1] != '-debug':
          command = sys.argv[1:]
          self.single_shot_command = True
          self.checkCommands(command, current_sim)
        else:
          self.debug = len(sys.argv) > 1 and sys.argv[1] == '-debug'
          self.single_shot_command = False
          try:
              import gnureadline as readline
          except ImportError:
              import readline

          readline.parse_and_bind('tab: complete')
          readline.parse_and_bind('set editing-mode vi')

          while True:
              command_string = input('> ')
              command = command_string.split(' ')
              if self.debug:
                  current_sim = self.checkCommands(command, current_sim)

                  if self.isQuitCommand(command):
                    break
              else:
                  try:
                      current_sim = self.checkCommands(command, current_sim)

                      if self.isQuitCommand(command):
                        break

                  except BaseException as e:
                      print (e)
                      print('For a more meaningful error (!), re-run miindio.py with argument -debug and call this command to get the full python stack trace.')
                      continue

if __name__ == "__main__":
    m = MiindIO()
    m.main()
  
