#!/usr/bin/env python

import argparse
import codegen
import sys
import os
import os.path as op
import directories
import jobs
import api
import matplotlib.pyplot as plt
import directories

def getMiindPythonPath():
    return os.path.join(directories.miind_root(), 'python')

def sim(command):
    command_name = command[0]
    name = 'sim'

    if command_name in [name]:
        if len(command) == 1:
            if not current_sim:
                print 'No simulation currently defined.'
                print ''
                sim(['sim?'])
            print 'Original XML File : {}'.format(current_sim.original_xml_path)
            print 'Project Name : {}'.format(current_sim.submit_name)
            print 'Parameters : {}'.format(current_sim.parameters)
            print 'Generated XML File : {}'.format(current_sim.xml_fname)
            print 'Output Directory : {}'.format(current_sim.output_directory)
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
        print 'sim : Provide information on the current simulation.'
        print 'sim [XML filename] : Use this XML file for the current simulation. The default project name is the same as the XML filename.'
        print 'sim [XML filename] [Project name] : Use this XML file for the current simulation and set a different project name.'
        print 'sim [XML filename] [Project name] [Parameter 1] [Parameter 2] ... : Use this XML file as a template with this project name to generate a new xml file (if not already generated) in which the Variable objects are set using the given Parameters.'

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
        print 'models : List all model files used by the current simulation.'

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
        print 'settings : List the current settings. Settings are stored in \'miind_ui_settings\' in the MIIND python directory.'
        print 'settings [ENABLE_MPI] [ENABLE_OPENMP] [ENABLE_ROOT] : Expects \'True\' or \'False\' for each of the three settings. Settings are persistent.'


def submit(command, current_sim):
    command_name = command[0]
    name = 'submit'

    if command_name in [name]:
        if not current_sim:
            print 'No simulation currently defined. Please call command \'sim\'.'

        if len(command) == 1:
            current_sim.submit(False,
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'])
        if len(command) == 2:
            current_sim.submit(command[1] in ['True', 'true', 'TRUE'],
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'])
        if len(command) >= 3:
            current_sim.submit(command[1] in ['True', 'true', 'TRUE'],
                  settings['mpi_enabled'], settings['openmp_enabled'], settings['root_enabled'], *command[2:])

    if command_name in [name+'?', name+' ?', name+' -h', name+' -?', name+' help', 'man '+name]:
        print 'submit : Generate and compile the code from the current simulation xml file. Ensure you have the correct settings (call \'settings\').'
        print 'submit '

if __name__ == "__main__":
  current_sim = None
  current_density = None
  current_marginal = None

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
      #try:

      sim(command)

      models(command, current_sim)

      _settings(command)

      submit(command, current_sim)




      if command_name in ['run']:
          if not current_sim:
              print 'No simulation currently defined.'

          if len(command) == 1:
              current_sim.run()
          if len(command) == 2:
              current_sim.run_mpi(int(command[1]))

      if command_name in ['rate']:
          if not current_sim:
              print 'No simulation currently defined.'

          current_sim.plotRate(int(command[1]))

      if command_name in ['density']:
          if not current_sim:
              print 'No simulation currently defined.'

          if len(command) == 1:
              if not current_density:
                  print 'No density model currently defined.'
              print 'Current density model : {}'.format(current_density.modelfname)
          if len(command) == 2:
              current_density = api.Density(current_sim, command[1])

      if command_name in ['generate-density-images']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_density:
              print 'No density model currently defined.'

          current_density.generateAllDensityPlotImages()

      if command_name in ['list-density-files']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_density:
              print 'No density model currently defined.'

          for f in current_density.fnames:
              print f

      if command_name in ['plot-density']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_density:
              print 'No density model currently defined.'

          fig, axis = plt.subplots()
          current_density.plotDensity(command[1], ax=axis)
          plt.show()

      if command_name in ['generate-density-movie']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_density:
              print 'No density model currently defined.'

          current_density.generateDensityAnimation(command[1], int(command[2]),
                                        command[3] in ['True', 'true', 'TRUE'],
                                        float(command[4]))

      if command_name in ['marginal']:
           if not current_sim:
               print 'No simulation currently defined.'

           if len(command) == 1:
               if not current_marginal:
                   print 'No marginal density model currently defined.'
               print 'Current marginal density model : {}'.format(current_marginal.modelfname)
           if len(command) == 2:
               current_marginal = api.Marginal(current_sim, command[1])
           if len(command) == 4:
               current_marginal = api.Marginal(current_sim, command[1],
                                        int(command[2]), int(command[3]))

      if command_name in ['list-marginal-times']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_marginal:
              print 'No marginal density model currently defined.'

          for t in current_marginal.times:
              print t

      if command_name in ['plot-marginal-densities']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_marginal:
              print 'No marginal density model currently defined.'

          fig, axis = plt.subplots(1,2)
          current_marginal.plotV(float(command[1]), axis[0])
          current_marginal.plotW(float(command[1]), axis[1])
          fig.show()

      if command_name in ['generate-marginal-images']:
          if not current_sim:
              print 'No simulation currently defined.'
          if not current_marginal:
              print 'No marginal density model currently defined.'

          current_marginal.generatePlotImages(int(command[1]))

      if command_name in ['generate-lif-mesh']:
          gen = api.LifMeshGenerator(command[1])
          gen.generateLifMesh()
          gen.generateLifStationary()
          gen.generateLifReversal()

      if command_name in ['generate-model']:
          api.ModelGenerator.buildModelFileFromMesh(command[1],
                                    float(command[2]), float(command[3]))

      if command_name in ['generate-matrix']:
          if len(command) == 3:
              api.ModelGenerator.buildMatrixFileFromModel(command[1], float(command[2]))
          if len(command) == 4:
              api.ModelGenerator.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[3])
          if len(command) == 5:
              api.ModelGenerator.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[3], num_mc_points=int(command[4]))
          if len(command) == 6:
              api.ModelGenerator.buildMatrixFileFromModel(command[1], float(command[2]), fidfile=command[3], num_mc_points=int(command[4]), spike_shift_w=float(command[5]))

      if command_name in ['exit', 'quit', 'close']:
          break
      # except BaseException as e:
      #     print e
      #     continue
