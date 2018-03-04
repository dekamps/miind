import os
import os.path as op
import glob
import ROOT
import numpy as np
import subprocess
import shutil
import copy
import collections
import hashlib
from tools import *
from density import Density, Marginal

import xml.etree.ElementTree as ET
# From MIIND
import directories

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

        simio = self.sim.find('SimulationIO')
        self.WITH_STATE = simio.find('WithState').text == 'TRUE'
        self.simulation_name = simio.find('SimulationName').text
        self.root_path = op.join(self.output_directory,
                                      self.simulation_name + '_0.root')

        # If there will be state files, we'll get a directory for each
        # mesh (model file)
        if self.WITH_STATE:
            modnames = [split_fname(mn, '.model')[0] for mn in self.modelfiles]
            self.density = {mn: Density(self, mn) for mn in modnames}
            self.marginal = {mn: Marginal(self, mn) for mn in modnames}

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
        if hasattr(self, '_rates'):
            return self._rates
        fnameout = op.join(self.output_directory,
                                 self.simulation_name + '_rates.npz')
        if op.exists(fnameout):
            return np.load(fnameout)['data'][()]
        f = ROOT.TFile(self.root_path)
        keys = [key.GetName() for key in list(f.GetListOfKeys())]
        graphs = {key: f.Get(key) for key in keys
                  if isinstance(f.Get(key), ROOT.TGraph)}
        _rates = {}
        for key, g in graphs.iteritems():
            x, y, N = g.GetX(), g.GetY(), g.GetN()
            x.SetSize(N)
            y.SetSize(N)
            xa = np.array(x, copy=True)
            ya = np.array(y, copy=True)
            times = xa.flatten()[2::2]
            if not 'times' in _rates:
                _rates['times'] = times
            else:
                assert not any(_rates['times'] - times > 1e-15)
            # TODO HACK TODO why every other here??? bug in miind??
            _rates[int(key.split('_')[-1])] = ya.flatten()[2::2]
        print 'Extracted %i rates from root file' % len(_rates.keys())
        self._rates = _rates
        np.savez(fnameout, data=_rates)
        return _rates

    @property
    def run_exists(self):
        '''
        checks if this particular
        '''
        xmlpath = op.join(self.output_directory, self.xml_fname)
        modelfiles = [op.join(self.output_directory, m)
                      for m in self.modelfiles]
        if not op.exists(xmlpath):
            return False
        old_params = convert_xml_dict(xmlpath)
        if not op.exists(self.root_path):
            return False
        if self.WITH_STATE:
            for p in modelfiles:
                if not op.exists(p + '_mesh'):
                    return False
                if len(os.listdir(p + '_mesh')) == 0:
                    return False
        return dict_changed(old_params, self.params) == set()

    @property
    def nodes(self):
        raise(BaseException("Not Implemented : MiindSimulation.nodes()"))

    def submit(self, overwrite=False, enable_mpi=False, enable_openmp=False, disable_root=False, *args):
        if op.exists(self.output_directory) and overwrite:
            shutil.rmtree(self.output_directory)
        with cd(self.xml_location):
            directories.add_executable(self.submit_name, [self.xml_path], '',
            enable_mpi, enable_openmp, disable_root)
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
