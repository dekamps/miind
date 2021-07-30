from .Result import Result

import os.path as op
import glob
import subprocess
import time
import matplotlib.pyplot as plt
import sys

from collections import OrderedDict as odict
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection

from .tools import *

import miind.mesh3 as meshmod

class Marginal(Result):
    def __init__(self, io, nodename, vn=1000, wn=1000):
        super(Marginal, self).__init__(io, nodename)
        self.path = op.join(self.io.getOutputDirectory(),
                            self.nodename + '_marginal_density')
        self.data_path = op.join(self.io.getOutputDirectory(), 'marginal_density.npz')
        self.projfname = self.modelpath.replace('.model', '.projection')
        self.vn, self.wn = vn, wn

    def __getitem__(self, name):
        return self.density[name]

    def uncached_density(self, time):
        self.read_projection()

        v = np.zeros(self.projection['N_V'])
        w = np.zeros(self.projection['N_W'])

        time_index = [i[0] for i in enumerate(self.times) if i[1] == time]
        assert len(time_index) == 1

        density, coords = read_density(self.fnames[time_index[0]])
        mass = calc_mass(self.mesh, density, coords)

        return self.uncached_calc_marginal_density(v, w, mass, coords, self.projection)

    def uncached_calc_marginal_density(self, v, w, masses, coords, proj):
        # temp function 'scale' to parse a transition row
        # in the projection file
        def scale(var, proj, mass):
            bins = [marg.split(',') for marg in proj.split(';')
                    if len(marg) > 0]
            for jj, dd in bins:
                var[int(jj)] += mass * float(dd)
            return var

        cells = proj['transitions'].findall('cell')
        cell_num = 0
        for trans in cells:
            cell_num += 1
            # Get the coordinates of this cell and its mass each time
            i, j = [int(a) for a in trans.find('coordinates').text.split(',')]
            cell_mass = masses[coords.index((i, j))]

            sys.stdout.write("%d%%   \r" % (100*(cell_num / len(cells))))
            sys.stdout.flush()

            # Calculate and add the density values for each bin
            v = scale(v, trans.find('vbins').text, cell_mass)
            w = scale(w, trans.find('wbins').text, cell_mass)

        # Generate the linspace values for plotting
        bins_v = np.linspace(proj['V_min'], proj['V_max'], proj['N_V'])
        bins_w = np.linspace(proj['W_min'], proj['W_max'], proj['N_W'])
        return v, w, bins_v, bins_w

    @property
    def density(self):
        # Get the projection file
        self.read_projection()
        # If there's no new projection and we've already calculated everything,
        # just return the existing calculated marginals
        if op.exists(self.data_path) and not self.new_projection:
            load_data = np.load(self.data_path,allow_pickle=True)['data'][()]
            if self.modelname in load_data:
                return load_data[self.modelname]

        # Initialise the marginal densities for each frame of the simulation
        v = np.zeros((len(self.times), self.projection['N_V']))
        w = np.zeros((len(self.times), self.projection['N_W']))

        print('Loading mass from densities...')

        # Load in the masses from the densities
        masses, coords_ = [], None
        for ii, fname in enumerate(self.fnames):
            sys.stdout.write("%d%%   \r" % (100*(ii / len(self.fnames))))
            sys.stdout.flush()
            density, coords = read_density(fname)
            if coords_ is None:
                coords_ = coords
            else:
                assert coords == coords_
            masses.append(calc_mass(self.mesh, density, coords))
        masses = np.vstack(masses)
        assert masses.shape[0] == len(self.fnames)

        print('Calculating marginal densities...')

        # Calculate the merginals for each frame and store in 'data'
        v, w, bins_v, bins_w = self.calc_marginal_density(
            v, w, masses, coords, self.projection)
        data = {'v': v, 'w': w, 'bins_v': bins_v,
                'bins_w': bins_w, 'times': self.times}

        # Save 'data' into a compressed file
        if op.exists(self.data_path):
            other = np.load(self.data_path,allow_pickle=True)['data'][()]
            other.update({self.modelname: data})
            save_data = other
        else:
            save_data = {self.modelname: data}
        np.savez(self.data_path, data=save_data)

        print('Done.')
        return data

    # Using the projection file, the mass from in each cell of the mesh
    # (for each frame of the simulation)
    # can be summed into the two marginal densities
    # note, v and w are 2D matrices of the marginal density bin values for each
    # frame of the simulation
    def calc_marginal_density(self, v, w, masses, coords, proj):
        # temp function 'scale' to parse a transition row
        # in the projection file
        def scale(var, proj, mass):
            bins = [marg.split(',') for marg in proj.split(';')
                    if len(marg) > 0]
            for jj, dd in bins:
                var[:, int(jj)] += mass * float(dd)
            return var

        # Each cell in the mesh has a transition row in the projection file
        cells = proj['transitions'].findall('cell')
        cell_num = 0
        for trans in cells:
            cell_num += 1
            # Get the coordinates of this cell and its mass each time
            i, j = [int(a) for a in trans.find('coordinates').text.split(',')]
            cell_mass = masses[:, coords.index((i, j))]

            sys.stdout.write("%d%%   \r" % (100*(cell_num / len(cells))))
            sys.stdout.flush()

            # Calculate and add the density values for each bin
            v = scale(v, trans.find('vbins').text, cell_mass)
            w = scale(w, trans.find('wbins').text, cell_mass)

        # Generate the linspace values for plotting
        bins_v = np.linspace(proj['V_min'], proj['V_max'], proj['N_V'])
        bins_w = np.linspace(proj['W_min'], proj['W_max'], proj['N_W'])
        return v, w, bins_v, bins_w

    def make_projection_file(self):
        # Run the projection app to analyse the model file and get
        # dimensions
        projection_exe = op.join(getMiindAppsPath(), 'Projection', 'Projection')
        out = subprocess.check_output(
          [projection_exe, self.modelfname], cwd=self.io.xml_location)

        # Parse the output
        vmax, wmax = np.array(out.decode('ascii').split('\n')[3].split(' ')[2:], dtype=float)
        vmin, wmin = np.array(out.decode('ascii').split('\n')[4].split(' ')[2:], dtype=float)
        # assert that we bound the range
        inc = lambda x: x * 1.01 if x > 0 else x * 0.99
        vmax, wmax = inc(vmax), inc(wmax)
        vmin, wmin = -inc(-vmin) , -inc(-wmin)

        # Run the projection app to generate the .projection file
        cmd = [projection_exe, self.modelfname, vmin, vmax,
               self.vn, wmin, wmax, self.wn]
        subprocess.call([str(c) for c in cmd], cwd=self.io.xml_location)

    def read_projection(self):
        proj_pathname = op.join(self.io.xml_location, self.projfname)
        self.new_projection = False
        # Does the pojection file exist? If not, generate it.
        if not op.exists(proj_pathname):
            print('No projection file found, generating...')
            print('This is a slow but one-shot process.')
            self.make_projection_file()
            self.new_projection = True

        # If we don't have projection data loaded, load it!
        if not hasattr(self, 'projection'):
            with open(proj_pathname) as proj_file:
                self._proj = ET.fromstring(proj_file.read())

        # Has the projection file changed since we last loaded? If so, reload.
        if (int(self._proj.find('W_limit/N_W').text) != self.wn or
            int(self._proj.find('V_limit/N_V').text) != self.vn):
            print('New N in bins, generating projection file...')
            self.make_projection_file()
            with open(proj_pathname) as proj_file:
                self._proj = ET.fromstring(proj_file.read())
            self.new_projection = True

        self.projection =  {
            'transitions': self._proj.find('transitions'),
            'V_min': float(self._proj.find('V_limit/V_min').text),
            'V_max': float(self._proj.find('V_limit/V_max').text),
            'N_V': int(self._proj.find('V_limit/N_V').text),
            'W_min': float(self._proj.find('W_limit/W_min').text),
            'W_max': float(self._proj.find('W_limit/W_max').text),
            'N_W': int(self._proj.find('W_limit/N_W').text),
        }

    def plotV(self, time, ax=None, showplot=True):
        v,w,bins_v,bins_w = self.uncached_density(time)
        if showplot:
            if not ax:
                fig, ax = plt.subplots()
                ax.plot(bins_v,v)
                fig.show()
            else:
                ax.plot(bins_v,v)
        return bins_v, v

    def plotW(self, time, ax=None, showplot=True):
        v,w,bins_v,bins_w = self.uncached_density(time)

        if showplot:
            if not ax:
                fig, ax = plt.subplots()
                ax.plot(bins_w, w)
                fig.show()
            else:
                ax.plot(bins_w, w)
        return bins_w, w

    def generatePlotImages(self, image_size=300):
        if not op.exists(self.path):
            os.mkdir(self.path)
        for ii in range(len(self['times'])):
            fig, axs = plt.subplots(1, 2)
            plt.suptitle('time = {}'.format(self['times'][ii]))
            self.plotV(self['times'][ii], axs[0])
            self.plotW(self['times'][ii], axs[1])
            required_padding = len(str(len(self.times)))
            padding_format_code = '{0:0' + str(required_padding) + 'd}'
            # We used to need the padding but don't any more (I hope)...
            #figname = op.join(self.path, (padding_format_code).format(ii) )
            figname = op.join(self.path, str(ii) )
            fig.savefig(figname, res=image_size, bbox_inches='tight')
            plt.close(fig)

    def generateMarginalAnimation(self, filename, image_size=300, generate_images=True, time_scale=1.0):
        # Generate the density image files
        if generate_images:
            self.generatePlotImages(image_size)

        try:
            # grab all the filenames
            files = glob.glob(op.join(self.path, '*.png'))
            files.sort()

            # note ffmpeg must be installed
            process = ['ffmpeg',
                '-r', str(1.0/time_scale),
                '-i', op.join(self.path, "%d.png")]

            process.append(filename + '.mp4')

            subprocess.call(process)

        except OSError as e:
            if e.errno == 2:
                print ("MIIND API Error : generateMarginalAnimation() requires ffmpeg to be installed.")
            else:
                print ("MIIND API Error : Unknown Error")
                print (e)
