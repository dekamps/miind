#!/usr/bin/env python3

import sys
import pylab
import numpy
import matplotlib.pyplot as plt
import subprocess
import os
import shutil
import miind.directories3

test_gpu = (len(sys.argv) > 1 and sys.argv[1] in ['gpu','GPU','g','G','-g','-G','-gpu','-GPU'])
hide_all_output = False
lif_steady_lower_bound = 0.34
lif_steady_upper_bound = 0.36
lif_steady_lower_bound_refract = 0.3
lif_steady_upper_bound_refract = 0.32

# Copy test files to current directory
testdir_path = os.path.join(os.getcwd(), 'testfiles')
shutil.rmtree(testdir_path, ignore_errors=True) 
print('Copying testfiles to: ', testdir_path) 
testfile_dir = os.path.join(miind.directories3.miind_python_dir(), 'testfiles')
shutil.copytree(testfile_dir,testdir_path)
# Now set testfile_dir to just testfiles
testfile_dir = 'testfiles'

# Check miindio.py can be found and runs
def TestMiindio():
	print('Checking miindio.py can be found and runs...')
	results = subprocess.run("python -m miind.miindio quit", shell=True, check=True)
	print('Success. Clean up.\n')

# 1. Basic MeshAlgorithm LIF simulation
def TestMeshAlgorithmLIF():
	print('Checking LIF MeshAlgorithm test 1...')
	testdir = os.path.join(testfile_dir, '1')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
    
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

		
# 2. MeshAlgorithmCustom CustomConnectionParameters
def TestMeshAlgorithmLIFCustomConnectionParameters():
	print('Checking LIF MeshAlgorithm with CustomConnectionParameters test 2...')
	print("now in " + os.getcwd())
	testdir = os.path.join(testfile_dir, '2')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')

	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 3. MeshAlgorithmGroup CustomConnectionParameters Vectorised
def TestMeshAlgorithmLIFCustomConnectionParametersGPU():
	if not test_gpu:
		print('LIF MeshAlgorithm with CustomConnectionParameters GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithm with CustomConnectionParameters GPU test 3...')
	testdir = os.path.join(testfile_dir, '3')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
    
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 4. MeshAlgorithmGroup DelayedConnection Vectorised
def TestMeshAlgorithmLIFGPU():
	if not test_gpu:
		print('LIF MeshAlgorithm with DelayedConnection GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithm with DelayedConnection GPU test 4...')
	testdir = os.path.join(testfile_dir, '4')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 5. Variables
def TestVariables():
	print('Checking Variables test 5...')
	testdir = os.path.join(testfile_dir, '5')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 6. Delay MeshAlgorithm DelayedConnection
def TestDelayMeshAlgorithmDelayedConnection():
	print('Checking MeshAlgorithm DelayedConnection Delay test 6...')
	testdir = os.path.join(testfile_dir, '6')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 7. Delay MeshAlgorithmCustom CustomConnectionParameters
def TestDelayMeshAlgorithmCustomConnectionParameters():
	print('Checking MeshAlgorithm CustomConnectionParameters Delay test 7...')
	testdir = os.path.join(testfile_dir, '7')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 8. Delay MeshAlgorithmGroup CustomConnectionParameters
def TestDelayMeshAlgorithmCustomConnectionParametersGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup CustomConnectionParameters Delay GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking MeshAlgorithmGroup CustomConnectionParameters Delay test 8...')
	testdir = os.path.join(testfile_dir, '8')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 9. Delay MeshAlgorithmGroup DelayedConnection
def TestDelayMeshAlgorithmDelayedConnectionGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup DelayedConnection Delay GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking MeshAlgorithmGroup DelayedConnection Delay GPU test 9...')
	testdir = os.path.join(testfile_dir, '9')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 10. MeshAlgorithm DelayeConnection tau_refractive
def TestMeshAlgorithmRefractive():
	print('Checking LIF MeshAlgorithm with Refractive test 10...')
	testdir = os.path.join(testfile_dir, '10')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 1998
		assert lines[-1].split('\t')[0] == "1.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound_refract and steady_rate < lif_steady_upper_bound_refract # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 11. MeshAlgorithm DelayeConnection tau_refractive
def TestMeshAlgorithmRefractiveGPU():
	if not test_gpu:
		print('LIF MeshAlgorithm with Refractive GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with Refractive test 11...')
	testdir = os.path.join(testfile_dir, '11')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound_refract and steady_rate < lif_steady_upper_bound_refract # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 12. MeshAlgorithm DelayeConnection with network timestep multiple
def TestMeshAlgorithmTimeStepMultiple():
	print('Checking LIF MeshAlgorithm with network timestep multiple test 12...')
	testdir = os.path.join(testfile_dir, '12')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 999
		assert lines[-1].split('\t')[0] == "0.999"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.39 and steady_rate < 0.41 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 13. MeshAlgorithmGroup DelayeConnection with network timestep multiple 
def TestMeshAlgorithmTimeStepMultipleGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup with network timestep multiple GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with network timestep multiple test 13...')
	testdir = os.path.join(testfile_dir, '13')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.37 and steady_rate < 0.41 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 14. MeshAlgorithm DelayeConnection multiple connections 
def TestMeshAlgorithmMultipleConnections():
	print('Checking LIF MeshAlgorithm with multiple connections test 14...')
	testdir = os.path.join(testfile_dir, '14')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 2.11 and steady_rate < 2.13 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 15. MeshAlgorithm DelayeConnection multiple connections 
def TestMeshAlgorithmMultipleConnectionsGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup with multiple connections GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with multiple connections test 15...')
	testdir = os.path.join(testfile_dir, '15')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 2.11 and steady_rate < 2.13 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 16. MeshAlgorithm DelayeConnection density recording
def TestMeshAlgorithmDensityRecording():
	print('Checking LIF MeshAlgorithm with density recording test 16...')
	testdir = os.path.join(testfile_dir, '16')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	os.chdir('lif.model_mesh')
	assert len([name for name in os.listdir('.') if os.path.isfile(name)]) == 499
	assert all([float(name.split('_')[4]) > 0.99998 and float(name.split('_')[4]) < 1.0001 for name in os.listdir('.') if os.path.isfile(name)])
	print('Success. Clean up.\n')
	os.chdir('..')
	shutil.rmtree('lif.model_mesh')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 17. MeshAlgorithm DelayeConnection density recording
def TestMeshAlgorithmGroupDensityRecordingGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup with density recording GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup with density recording test 17...')
	testdir = os.path.join(testfile_dir, '17')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	os.chdir("lif_")
	os.chdir('densities')
	print('Success. Clean up.\n')
	os.chdir('..')
	shutil.rmtree('densities')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 18. MeshAlgorithm Two nodes One Algorithm
def TestMeshAlgorithmTwoNodes():
	print('Checking LIF MeshAlgorithm two nodes one algorithm test 18...')
	testdir = os.path.join(testfile_dir, '18')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 19. MeshAlgorithmGroup Two nodes One Algorithm
def TestMeshAlgorithmGroupTwoNodesGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup two nodes one algorithm GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup two nodes one algorithm test 19...')
	testdir = os.path.join(testfile_dir, '19')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 20. MeshAlgorithm Two nodes One Algorithm
def TestMeshAlgorithmTwoNodesSelfConnections():
	print('Checking LIF MeshAlgorithm two nodes one algorithm self connections test 20...')
	testdir = os.path.join(testfile_dir, '20')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.375 and steady_rate < 0.385 # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.375 and steady_rate < 0.385 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 21. MeshAlgorithmGroup Two nodes One Algorithm Self Connections
def TestMeshAlgorithmGroupTwoNodesSelfConnectionsGPU():
	if not test_gpu:
		print('LIF MeshAlgorithmGroup two nodes one algorithm self connections GPU SKIPPED (GPU tests disabled)')
		return

	print('Checking LIF MeshAlgorithmGroup two nodes one algorithm self connections test 21...')
	testdir = os.path.join(testfile_dir, '21')
	os.chdir(testdir)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.365 and steady_rate < 0.375 # This is a very generous margin!
	with open('rate_1', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.365 and steady_rate < 0.375 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 22. Generate Fid File and Matrix
def TestGenerateFidAndMatrix():
	print('Checking Generate Fid file and Matrix 22...')
	testdir = os.path.join(testfile_dir, '22')
	os.chdir(testdir)
	print('sim lif.xml.')
	results = subprocess.run("python -m miind.miindio sim lif.xml", shell=True, check=True)
	print('generate-empty-fid lif.')
	results = subprocess.run("python -m miind.miindio generate-empty-fid lif", shell=True, check=True)
	with open('lif.fid', 'r') as f:
		assert f
	print('generate-matrix lif 0.01 10000 0.0 0.0 True.')
	results = subprocess.run("python -m miind.miindio generate-matrix lif 0.01 10000 0.0 0.0 True", shell=True, check=True)
	with open('lif_0.01_0_0_0_.mat', 'r') as f:
		assert f
	print('run.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > lif_steady_lower_bound and steady_rate < lif_steady_upper_bound # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 23. Generate Matrix Monte Carlo
def TestGenerateMatrixMC():
	print('Checking Generate Matrix Monte Carlo 23...')
	testdir = os.path.join(testfile_dir, '23')
	os.chdir(testdir)
	print('sim lif.xml.')
	results = subprocess.run("python -m miind.miindio sim lif.xml", shell=True, check=True)
	print('generate-matrix lif 0.01 10 0.0 0.0 False.')
	results = subprocess.run("python -m miind.miindio generate-matrix lif 0.01 10 0.0 0.0 False", shell=True, check=True)
	with open('lif_0.01_0_0_0_.mat', 'r') as f:
		assert f
	print('run.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 0.245 and steady_rate < 0.255 # Obviously this range is wrong but that's because we used MC=10
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')

# 24. Generate LIF Mesh and Model
def TestGenerateMeshModel():
	print('Checking Generate LIF Mesh and Model 24...')
	testdir = os.path.join(testfile_dir, '24')
	os.chdir(testdir)
	print('generate-lif-mesh lif.')
	results = subprocess.run("python -m miind.miindio generate-lif-mesh lif", shell=True, check=True)
	print('generate-model lif 0.0 1.0.')
	results = subprocess.run("python -m miind.miindio generate-model lif -65.0 -50.2", shell=True, check=True)
	print('generate-model generate-empty-fid lif.')
	results = subprocess.run("python -m miind.miindio generate-empty-fid lif", shell=True, check=True)
	print('generate-matrix lif 1 10000 0.0 0.0 True.')
	results = subprocess.run("python -m miind.miindio generate-matrix lif 1 10000 0.0 0.0 True", shell=True, check=True)
	print('run lif.xml.')
	results = subprocess.run("python -m miind.run lif.xml", shell=True, check=True)
	print('Checking rate output...')
	os.chdir("lif_")
	with open('rate_0', 'r') as ratefile:
		lines = ratefile.read().splitlines()
		assert len(lines) == 998
		assert lines[-1].split('\t')[0] == "0.998"
		steady_rate = float(lines[-1].split('\t')[1])
		assert steady_rate > 8.75 and steady_rate < 8.78 # This is a very generous margin!
	print('Success. Clean up.\n')
	os.chdir('..')
	os.chdir('..')
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

print('Deleting local testfiles.')
shutil.rmtree(testdir_path)

