#!/usr/bin/env python

import argparse
import codegen
import sys
import os
import directories
import jobs
import api
import matplotlib.pyplot as plt

if __name__ == "__main__":
  current_sim = None
  current_density = None
  current_marginal = None
  mpi_enabled = False
  openmp_enabled = False
  root_enabled = True
  while True:
      command_string = raw_input('> ')
      command = command_string.split(' ')
      command_name = command[0]
      #try:
      if command_name in ['sim']:
          if len(command) == 1:
              if not current_sim:
                  print 'No simulation currently defined.'
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

      if command_name in ['list-model-files']:
          if not current_sim:
              print 'No simulation currently defined.'

          for mf in current_sim.modelfiles:
              print mf

      if command_name in ['set-submit-parameters']:
          if len(command) != 4:
              print "set-submit-parameters expects [ENABLE_MPI] [ENABLE_OPENMP] [ENABLE_ROOT]."

          mpi_enabled = (command[1] in ['True', 'true', 'TRUE'])
          openmp_enabled = (command[2] in ['True', 'true', 'TRUE'])
          root_enabled = (command[3] in ['True', 'true', 'TRUE'])

      if command_name in ['submit']:
          if not current_sim:
              print 'No simulation currently defined.'

          if len(command) == 1:
              current_sim.submit(False, mpi_enabled, openmp_enabled, root_enabled)
          if len(command) == 2:
              current_sim.submit(command[1] in ['True', 'true', 'TRUE'], mpi_enabled, openmp_enabled, root_enabled)
          if len(command) >= 3:
              current_sim.submit(command[1] in ['True', 'true', 'TRUE'],
                    mpi_enabled, openmp_enabled, root_enabled, *command[2:])

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
          api.ModelGenerator.buildMatrixFileFromModel(command[1], float(command[2]))

      if command_name in ['exit', 'quit', 'close']:
          break
      # except BaseException as e:
      #     print e
      #     continue
