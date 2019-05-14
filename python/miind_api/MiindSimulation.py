import os
import os.path as op
import glob
import numpy as np
import subprocess
import shutil
import copy
import collections
import hashlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from .tools import *
from .Density import Density
from .Marginal import Marginal

# From MIIND
import directories3 as directories
import miind
import miind_lib

class MiindSimulation:
    def __init__(self, xml_path, submit_name=None, **kwargs):
        self.parameters = kwargs
        # original xml path used by the ui to reference this MiindSimulation
        self.original_xml_path = xml_path
        # If there are kwargs, we either want to create a new xml with these
        # parameters, or we want to load a previously generated xml
        # else just load what in xml_path and normal
        if self.parameters :
            xml_path_with_sha = xml_path.replace('.xml', self.sha + '.xml')
            if op.exists(xml_path_with_sha) :
                self.xml_path = op.abspath(xml_path_with_sha)
            else :
                self.xml_path = op.abspath(MiindSimulation.generateNewXmlWithDictionaryParameters(xml_path, kwargs))
        else :
            self.xml_path = op.abspath(xml_path)

        # check our (maybe new) xml exists
        assert op.exists(self.xml_path)

        # keep track of various names to make analysis easier later
        self.xml_location, self.xml_fname = op.split(self.xml_path)

        xml_base_fname, _ = op.splitext(self.xml_fname)
        self.submit_name = submit_name or xml_base_fname
        self.output_directory = op.join(self.xml_location, self.submit_name, xml_base_fname)
        self.miind_executable = xml_base_fname

        # grab the names of the model files used and whether we should expect
        # state files with this simulation run
        with open(self.xml_path) as xml_file:
            self.sim=ET.fromstring(xml_file.read())

        self.modelfiles = [m.attrib['modelfile']
                        for m in self.sim.findall('Algorithms/Algorithm')
                        if 'modelfile' in m.attrib]

        # Get the names of each Node instance. We know that MIIND indexes each node in order
        # of definition in the xml file and uses that number to index the firing rate results so
        # we can enumerate this list to get the index of each node.
        self.nodenames = [(m.attrib['name'], None)
                        for m in self.sim.findall('Nodes/Node')
                        if 'name' in m.attrib]

        # MIIND also indexes the density files in order of node definition in the xml file,
        # however, the index excludes non-Mesh nodes.
        mesh_algos = [(m.attrib['name'], m.attrib['modelfile'])
                        for m in self.sim.findall('Algorithms/Algorithm')
                        if 'modelfile' in m.attrib]

        node_algos = [(m.attrib['name'], m.attrib['algorithm'])
                        for m in self.sim.findall('Nodes/Node')]

        self.meshnodenames = [(nn,f) for (nn, a) in node_algos for (m, f) in mesh_algos if a == m ]

        for i in range(len(self.nodenames)):
            for j in range(len(self.meshnodenames)):
                if self.nodenames[i][0] == self.meshnodenames[j][0]:
                    self.nodenames[i] = self.meshnodenames[j]

        # Hold the Variable names, useful for the UI to know.
        self.variablenames = [m.attrib['Name'] for m in self.sim.findall('Variable') if 'Name' in m.attrib]

        # Loading a Density or Marginal object is *expensive* so allow each
        # one to register/cache itself here for reuse.

        self.density_cache = {}
        self.marginal_cache = {}

        simio = self.sim.find('SimulationRunParameter')
        sim_name = simio.find('SimulationName')
        if sim_name is not None:
            self.simulation_name = sim_name.text
        else:
            self.simulation_name = "unnamed_sim"

        # If run using, MPI, there are multiple root files.
        self.root_paths = []
        root_index = 0
        while(op.exists(op.join(self.output_directory,
                                      self.simulation_name + '_' + str(root_index) + '.root'))):
            self.root_paths.append(op.join(self.output_directory,
                                      self.simulation_name + '_' + str(root_index) + '.root'))
            root_index += 1

    # By generating a hash of the specific parameters for this simulation,
    # we can uniquely identify the xml, compiled code, executable and results
    # directory. Ideally, the user should never care what the actual hash is
    # and just interface with the files via this api, using the specified
    # parameters as identification.
    @property
    def sha(self):
        return MiindSimulation.generateShaFromDictionaryParameters(self.parameters)

    @staticmethod
    def generateShaFromParameters(**kwargs) :
        seed_string = ''.join('{}{}'.format(key, val) for key, val in kwargs.items())
        return hashlib.sha1(seed_string).hexdigest()

    # Unwinding a dictionary into keyword arguments might confuse users
    # so provide a definitive dictionary param version
    @staticmethod
    def generateShaFromDictionaryParameters(dict) :
        return MiindSimulation.generateShaFromParameters(**dict)

    @staticmethod
    def generateNewXmlWithParameters(template_xml, **kwargs) :
        template_xml = op.abspath(template_xml)
        assert op.exists(template_xml)
        template_xml_with_sha = template_xml.replace('.xml', MiindSimulation.generateShaFromParameters(**kwargs) + '.xml')

        with open(template_xml) as template :
            template_sim = ET.fromstring(template.read())
            for key, value in kwargs.items() :
                var = template_sim.findall('Variable[@Name=\'' + key + '\']')
                if not var:
                    raise(BaseException("Could not find Variable with Name=\'" + key + "\'."))
                if len(var) > 1 :
                    raise(BaseException("Found more than one Variable with Name=\'" + key + "\'."))
                var[0].text = str(value)

            with open(template_xml_with_sha, 'w') as new :
                new.write(ET.tostring(template_sim))
        return op.abspath(template_xml_with_sha)

    # Unwinding a dictionary into keyword arguments might confuse users
    # so provide a definitive dictionary param version
    @staticmethod
    def generateNewXmlWithDictionaryParameters(template_xml, dict) :
        return MiindSimulation.generateNewXmlWithParameters(template_xml, **dict)

    @property
    def rates(self):
        _rates = {}
        _rates['times'] = []

        for i in range(len(self.nodenames)):
            if not op.exists(self.output_directory + "/rate_" + str(i)):
                continue
            with open(self.output_directory + "/rate_" + str(i)) as rate_file:
                _rates[i] = []
                for line in rate_file:
                    tokens = line.split('\t')
                    _rates[i] = _rates[i] + [float(tokens[1])]
                    _rates['times'] = _rates['times'] + [float(tokens[0])]

        self._rates = _rates
        return _rates

    def getModelFilenameAndIndexFromNode(self, nodename):
        if nodename.isdigit():
            (i,m) = self.meshnodenames[int(nodename)]
            return i,m
        else:
            for index, (name, modelfile) in enumerate(self.nodenames):
                if name == nodename:
                    return index, modelfile
        return None, None

    def getDensityByNodeName(self, nodename):
        if nodename not in self.density_cache:
            self.density_cache[nodename] = Density(self, nodename)

        return self.density_cache[nodename]

    def getMarginalByNodeName(self, nodename):
        if nodename not in self.marginal_cache:
            self.marginal_cache[nodename] = Marginal(self, nodename)

        return self.marginal_cache[nodename]

    def getIndexFromNode(self, nodename):
        if nodename.isdigit():
            return int(nodename)
        else:
            for index, (name,_) in enumerate(self.nodenames):
                if name == nodename:
                    return index
        return None

    def plotRate(self, node, ax=None):
        node_index = self.getIndexFromNode(node)
        if not ax:
            fig, ax = plt.subplots()
            plt.title(node)

            rate_length = min(len(self.rates['times']), len(self.rates[node_index]))
            ax.plot(self.rates['times'][0:rate_length], self.rates[node_index][0:rate_length])
            fig.show()
        else:
            ax.plot(self.rates['times'][0:rate_length], self.rates[node_index][0:rate_length])

    # Check if this particular simulation has been run previously
    @property
    def runCompleted(self):
        xmlpath = op.join(self.output_directory, self.xml_fname)
        modelfiles = [op.join(self.output_directory, m)
                      for m in self.modelfiles]
        # Has the xml file been copied to this results directory?
        if not op.exists(xmlpath):
            return False

        # Has at least one root file been generated?
        if not op.exists(self.root_paths[0]):
            return False

        for p in modelfiles:
            if not op.exists(p + '_mesh'):
                return False
            if len(os.listdir(p + '_mesh')) == 0:
                return False

        return True

    @property
    def nodes(self):
        return self.nodenames

    def submit_shared_lib(self, overwrite=False, enable_mpi=False, enable_openmp=False, enable_root=True, enable_cuda=False, *args):
        if op.exists(self.output_directory) and overwrite:
            shutil.rmtree(self.output_directory)
        with cd(self.xml_location):
            miind_lib.generate_vectorized_network_lib(self.submit_name, [self.xml_path], '',
            enable_mpi, enable_openmp, enable_root, enable_cuda)
        fnames = os.listdir(self.output_directory)
        if 'CMakeLists.txt' in fnames:
            subprocess.call(['cmake .'] +
                             [a for a in args],
                             cwd=self.output_directory, shell=True)
            subprocess.call(['make'], cwd=self.output_directory)
            shutil.copyfile(self.xml_path, op.join(self.output_directory,
                                                        self.xml_fname))

    def submit(self, overwrite=False, enable_mpi=False, enable_openmp=False, enable_root=True, enable_cuda=False, *args):
        if op.exists(self.output_directory) and overwrite:
            shutil.rmtree(self.output_directory)
        with cd(self.xml_location):
            miind.generate_vectorized_network_executable(self.submit_name, [self.xml_path], '',
            enable_mpi, enable_openmp, enable_root, enable_cuda)
        fnames = os.listdir(self.output_directory)
        if 'CMakeLists.txt' in fnames:
            subprocess.call(['cmake .'] +
                             [a for a in args],
                             cwd=self.output_directory, shell=True)
            subprocess.call(['make'], cwd=self.output_directory)
            shutil.copyfile(self.xml_path, op.join(self.output_directory,
                                                        self.xml_fname))

    def run(self):
        subprocess.call('./' + self.miind_executable, cwd=self.output_directory)

    def run_mpi(self, cores):
        subprocess.call(['mpiexec', '--bind-to', 'core', '-n', str(cores),
                        self.miind_executable], cwd=self.output_directory)
