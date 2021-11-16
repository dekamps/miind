#!/usr/bin/env python3

import sys
import pylab
import numpy
import matplotlib.pyplot as plt
import subprocess
import os
import shutil

test_gpu = (len(sys.argv) > 1 and sys.argv[1] in ['gpu','GPU','g','G','-g','-G','-gpu','-GPU'])
hide_all_output = False
lif_steady_lower_bound = 0.34
lif_steady_upper_bound = 0.36
lif_steady_lower_bound_refract = 0.3
lif_steady_upper_bound_refract = 0.32

miindio_command = 'miindio.py'
if len(sys.argv) > 1:
	# allow the user to set the run command - useful for windows where calling
	# miindio.py as a command doesn't run the script.
	miindio_command = sys.argv[1]

# Check miindio.py can be found and runs
def TestMiindio():
	print('Checking miindio.py can be found and runs...')
	results = subprocess.run(miindio_command + " quit", shell=True, check=True)
	print('Success. Clean up.\n')
	os.remove('miind_cwd')

# 1. Basic MeshAlgorithm LIF simulation
def TestMeshAlgorithmLIF():
	print('Checking LIF MeshAlgorithm test 1...')
	os.chdir('1')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		try:
			assert len(lines) == 998, "The rate file rate_0 did not contain 998 lines as expected."
			assert lines[-1].split('\t')[0] == "0.998", "The final line of rate file rate_0 did not correspond to time = 0.998."
			steady_rate = float(lines[-1].split('\t')[1])
			assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound, "Steady state rate " + str(steady_rate) + " is not between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "."
		except AssertionError as msg:
			print(msg)
			print("What was in rate_0?")
			print(lines)
			raise
			
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')
		
# 2. MeshAlgorithmCustom CustomConnectionParameters
def TestMeshAlgorithmLIFCustomConnectionParameters():
	print('Checking LIF MeshAlgorithm with CustomConnectionParameters test 2...')
	os.chdir('2')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 3. MeshAlgorithmGroup CustomConnectionParameters Vectorised
def TestMeshAlgorithmLIFCustomConnectionParametersGPU():
	if not test_gpu:
		print('LIF MeshAlgorithm with CustomConnectionParameters GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithm with CustomConnectionParameters GPU test 3...')
	os.chdir('3')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 4. MeshAlgorithmGroup DelayedConnection Vectorised
def TestMeshAlgorithmLIFGPU():
	if not test_gpu:
		print('LIF MeshAlgorithm with DelayedConnection GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithm with DelayedConnection GPU test 4...')
	os.chdir('4')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 5. Variables
def TestVariables():
	print('Checking Variables test 5...')
	os.chdir('5')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 6. Delay MeshAlgorithm DelayedConnection
def TestDelayMeshAlgorithmDelayedConnection():
	print('Checking MeshAlgorithm DelayedConnection Delay test 6...')
	os.chdir('6')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 7. Delay MeshAlgorithmCustom CustomConnectionParameters
def TestDelayMeshAlgorithmCustomConnectionParameters():
	print('Checking MeshAlgorithm CustomConnectionParameters Delay test 7...')
	os.chdir('7')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 8. Delay MeshAlgorithmGroup CustomConnectionParameters
def TestDelayMeshAlgorithmCustomConnectionParametersGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup CustomConnectionParameters Delay GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking MeshAlgorithmGroup CustomConnectionParameters Delay test 8...')
	os.chdir('8')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 9. Delay MeshAlgorithmGroup DelayedConnection
def TestDelayMeshAlgorithmDelayedConnectionGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup DelayedConnection Delay GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking MeshAlgorithmGroup DelayedConnection Delay GPU test 9...')
	os.chdir('9')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 10. MeshAlgorithm DelayeConnection tau_refractive
def TestMeshAlgorithmRefractive():
	print('Checking LIF MeshAlgorithm with Refractive test 10...')
	os.chdir('10')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 1998
		assert lines[-1].split('\t')[0] == "1.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound_refract) + " and " + str(lif_steady_upper_bound_refract) + "?")
		assert steady_rate > lif_steady_lower_bound_refract and steady_rate < lif_steady_upper_bound_refract # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 11. MeshAlgorithm DelayeConnection tau_refractive
def TestMeshAlgorithmRefractiveGPU():
	if not test_gpu:
		print('LIF MeshAlgorithm with Refractive GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with Refractive test 11...')
	os.chdir('11')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 1998
		assert lines[-1].split('\t')[0] == "1.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate) + " between " + str(lif_steady_lower_bound_refract) + " and " + str(lif_steady_upper_bound_refract) + "?")
		assert steady_rate > lif_steady_lower_bound_refract and steady_rate < lif_steady_upper_bound_refract # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 12. MeshAlgorithm DelayeConnection with network timestep multiple
def TestMeshAlgorithmTimeStepMultiple():
	print('Checking LIF MeshAlgorithm with network timestep multiple test 12...')
	os.chdir('12')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 999
		assert lines[-1].split('\t')[0] == "0.999"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between 0.39 and 0.41?")
		assert steady_rate > 0.39 and steady_rate < 0.41 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 13. MeshAlgorithmGroup DelayeConnection with network timestep multiple 
def TestMeshAlgorithmTimeStepMultipleGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup with network timestep multiple GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with network timestep multiple test 13...')
	os.chdir('13')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 999
		assert lines[-1].split('\t')[0] == "0.999"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between 0.39 and 0.41?")
		assert steady_rate > 0.37 and steady_rate < 0.41 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 14. MeshAlgorithm DelayeConnection multiple connections 
def TestMeshAlgorithmMultipleConnections():
	print('Checking LIF MeshAlgorithm with multiple connections test 14...')
	os.chdir('14')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between 2.11 and 2.13?")
		assert steady_rate > 2.11 and steady_rate < 2.13 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 15. MeshAlgorithm DelayeConnection multiple connections 
def TestMeshAlgorithmMultipleConnectionsGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup with multiple connections GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with multiple connections test 15...')
	os.chdir('15')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between 2.11 and 2.13?")
		assert steady_rate > 2.11 and steady_rate < 2.13 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 16. MeshAlgorithm DelayeConnection density recording
def TestMeshAlgorithmDensityRecording():
	print('Checking LIF MeshAlgorithm with density recording test 16...')
	os.chdir('16')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif/lif.model_mesh')
	assert len([name for name in os.listdir('.') if os.path.isfile(name)]) == 499
	assert all([float(name.split('_')[4]) > 0.99998 and float(name.split('_')[4]) < 1.0001 for name in os.listdir('.') if os.path.isfile(name)])
	print('Success. Clean up.\n')
	os.chdir('../../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 17. MeshAlgorithm DelayeConnection density recording
def TestMeshAlgorithmGroupDensityRecordingGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup with density recording GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with density recording test 17...')
	os.chdir('17')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif/densities')
	assert len([name for name in os.listdir('.') if os.path.isfile(name)]) == 499
	assert all([float(name.split('_')[4]) > 0.99998 and float(name.split('_')[4]) < 1.0001 for name in os.listdir('.') if os.path.isfile(name)])
	print('Success. Clean up.\n')
	os.chdir('../../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 18. MeshAlgorithm Two nodes One Algorithm
def TestMeshAlgorithmTwoNodes():
	print('Checking LIF MeshAlgorithm two nodes one algorithm test 18...')
	os.chdir('18')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 19. MeshAlgorithmGroup Two nodes One Algorithm
def TestMeshAlgorithmGroupTwoNodesGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup two nodes one algorithm GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup two nodes one algorithm test 19...')
	os.chdir('19')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 20. MeshAlgorithm Two nodes One Algorithm
def TestMeshAlgorithmTwoNodesSelfConnections():
	print('Checking LIF MeshAlgorithm two nodes one algorithm self connections test 20...')
	os.chdir('20')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(0.375) + " and " + str(0.385) + "?")
		assert steady_rate > 0.375 and steady_rate < 0.385 # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(0.375) + " and " + str(0.385) + "?")
		assert steady_rate > 0.375 and steady_rate < 0.385 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 21. MeshAlgorithmGroup Two nodes One Algorithm Self Connections
def TestMeshAlgorithmGroupTwoNodesSelfConnectionsGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup two nodes one algorithm self connections GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup two nodes one algorithm self connections test 21...')
	os.chdir('21')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(0.365) + " and " + str(0.375) + "?")
		assert steady_rate > 0.365 and steady_rate < 0.375 # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(0.365) + " and " + str(0.375) + "?")
		assert steady_rate > 0.365 and steady_rate < 0.375 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.chdir('..')

# 22. Generate Fid File and Matrix
def TestGenerateFidAndMatrix():
	print('Checking Generate Fid file and Matrix 22...')
	os.chdir('22')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('generate-empty-fid lif.')
	results = subprocess.run(miindio_command + " generate-empty-fid lif", shell=True, check=True)
	with open('lif.fid', 'r') as f:
		assert f
	print('generate-matrix lif 0.01 10000 0.0 0.0 True.')
	results = subprocess.run(miindio_command + " generate-matrix lif 0.01 10000 0.0 0.0 True", shell=True, check=True)
	with open('lif_0.01_0_0_0_.mat', 'r') as f:
		assert f
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(lif_steady_lower_bound) + " and " + str(lif_steady_upper_bound) + "?")
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.remove('lif.fid')
	os.remove('lif.res')
	os.remove('lif_0.01_0_0_0_.mat')
	os.chdir('..')

# 23. Generate Matrix Monte Carlo
def TestGenerateMatrixMC():
	print('Checking Generate Matrix Monte Carlo 23...')
	os.chdir('23')
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('generate-matrix lif 0.01 10 0.0 0.0 False.')
	results = subprocess.run(miindio_command + " generate-matrix lif 0.01 10 0.0 0.0 False", shell=True, check=True)
	with open('lif_0.01_0_0_0_.mat', 'r') as f:
		assert f
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(0.245) + " and " + str(0.255) + "?")
		assert steady_rate > 0.245 and steady_rate < 0.255 # Obviously this range is wrong but that's because we used MC=10
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.remove('lif.res')
	os.remove('lif_0.01_0_0_0_.mat')
	os.remove('lif_0.01_0_0_0_.lost')
	os.chdir('..')

# 24. Generate LIF Mesh and Model
def TestGenerateMeshModel():
	print('Checking Generate LIF Mesh and Model 24...')
	os.chdir('24')
	print('generate-lif-mesh lif.')
	results = subprocess.run(miindio_command + " generate-lif-mesh lif", shell=True, check=True)
	print('generate-model lif 0.0 1.0.')
	results = subprocess.run(miindio_command + " generate-model lif -65.0 -50.2", shell=True, check=True)
	print('generate-model generate-empty-fid lif.')
	results = subprocess.run(miindio_command + " generate-empty-fid lif", shell=True, check=True)
	print('generate-matrix lif 1 10000 0.0 0.0 True.')
	results = subprocess.run(miindio_command + " generate-matrix lif 1 10000 0.0 0.0 True", shell=True, check=True)
	print('sim lif.xml.')
	results = subprocess.run(miindio_command + " sim lif.xml", shell=True, check=True)
	print('submit.')
	results = subprocess.run(miindio_command + " submit", shell=True, check=True)
	print('run.')
	results = subprocess.run(miindio_command + " run", shell=True, check=True)
	print('Checking rate output...')
	os.chdir('lif/lif')
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		print("Steady state rate is " + str(steady_rate), " between " + str(8.75) + " and " + str(8.78) + "?")
		assert steady_rate > 8.75 and steady_rate < 8.78 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('../..')
	shutil.rmtree('lif')
	os.remove('miind_cwd')
	os.remove('lif.res')
	os.remove('lif.stat')
	os.remove('lif.rev')
	os.remove('lif.model')
	os.remove('lif.mesh')
	os.remove('lif.fid')
	os.remove('lif_1_0_0_0_.mat')
	os.chdir('..')

TestMiindio()
TestMeshAlgorithmLIF()
TestMeshAlgorithmLIFCustomConnectionParameters()
TestMeshAlgorithmLIFCustomConnectionParametersGPU()
TestMeshAlgorithmLIFGPU()
TestVariables()
TestDelayMeshAlgorithmDelayedConnection()
TestDelayMeshAlgorithmCustomConnectionParameters()
TestDelayMeshAlgorithmCustomConnectionParametersGPU()
TestDelayMeshAlgorithmDelayedConnectionGPU()
TestMeshAlgorithmRefractive()
TestMeshAlgorithmRefractiveGPU()
TestMeshAlgorithmTimeStepMultiple()
TestMeshAlgorithmTimeStepMultipleGPU()
TestMeshAlgorithmMultipleConnections()
TestMeshAlgorithmMultipleConnectionsGPU()
TestMeshAlgorithmDensityRecording()
TestMeshAlgorithmGroupDensityRecordingGPU()
TestMeshAlgorithmTwoNodes()
TestMeshAlgorithmGroupTwoNodesGPU()
TestMeshAlgorithmTwoNodesSelfConnections()
TestMeshAlgorithmGroupTwoNodesSelfConnectionsGPU()
TestGenerateFidAndMatrix()
TestGenerateMatrixMC()
TestGenerateMeshModel()
print('All tests successful.')

