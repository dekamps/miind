#!/usr/bin/env python

import argparse
import codegen
import sys
import os
import directories
import jobs
import api

if __name__ == "__main__":
  current_sim = None
  while True:
      command_string = raw_input('> ')
      command = command_string.split(' ')
      command_name = command[0]
      try:
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

          if command_name in ['submit']:
              if not current_sim:
                  print 'No simulation currently defined.'

              if len(command) == 1:
                  current_sim.submit()
              if len(command) == 2:
                  current_sim.submit()

          if command_name in ['exit', 'quit', 'close']:
              break
      except BaseException as e:
          print e
          continue
