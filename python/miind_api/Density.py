from Result import Result

import os.path as op
import glob
import subprocess
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import OrderedDict as odict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection

from tools import *

import mesh as meshmod

class Density(Result):
    def __init__(self, io, nodename):
        super(Density, self).__init__(io, nodename)
        self.path = op.join(self.io.output_directory,
                            self.nodename + '_density')

    @property
    def polygons(self):
        if not hasattr(self, '_polygons'):
            self._polygons = odict(
                ((i, j),
                Polygon([(float(x), float(y))
                         for x, y in zip(cell.vs, cell.ws)]))
                for i, cells in enumerate(self.mesh.cells)
                for j, cell in enumerate(cells)
            )
        return self._polygons

    @property
    def patches(self):
        if not hasattr(self, '_patches'):
            self._patches = [
                PolygonPatch(polygon)
                for polygon in self.polygons.values()
            ]
        return self._patches

    def plot_mesh(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        md = self.mesh.dimensions()
        p = PatchCollection(self.patches, alpha=1, edgecolors='k',
                            facecolors='w')
        ax.add_collection(p)
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])
        aspect = (md[0][1] - md[0][0]) / (md[1][1] - md[1][0])
        ax.set_aspect(aspect)
        return ax

    def colscale(self, density):
        cols = np.log10(np.array(density) + 1e-6)
        vmax = np.max(cols)
        vmin = np.min(cols)
        vals = (cols - vmin)/(vmax - vmin)
        return vmin, vmax, vals

    def generateDensityAnimation(self, filename, image_size=300, generate_images=True, time_scale=1.0,
                            colorbar=None, cmap='inferno'):
        # Generate the density image files
        if generate_images:
            self.generateAllDensityPlotImages(image_size, colorbar, cmap, '.png')

        try:
            # grab all the filenames
            files = glob.glob(op.join(self.path, '*.png'))
            files.sort()

            # note ffmpeg must be installed
            process = ['ffmpeg',
                '-r', str(int((1.0 / time_scale) * (len(files)/self.times[-1]))),
                '-pattern_type', 'glob',
                '-i', op.join(self.path, '*.png')]

            process.append(filename + '.mp4')

            subprocess.call(process)
        except OSError as e:
            if e.errno == 2:
                print "MIIND API Error : generateDensityAnimation() requires ffmpeg to be installed."
            else:
                print "MIIND API Error : Unknown Error"
                print e

    def generateAllDensityPlotImages(self, image_size=300, colorbar=None, cmap='inferno', ext='.png'):
        if not ext.startswith('.'):
            ext = '.' + ext

        fig, ax = plt.subplots()

        times = [get_density_time(fn) for fn in self.fnames]
        idxs = [self.times.index(t) for t in times]

        md = self.mesh.dimensions()
        poly_coords = list(self.polygons.keys())
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])

        ax.set_aspect('auto')
        p = PatchCollection(self.patches, cmap=cmap)
        ax.add_collection(p)

        if colorbar is not None:
            plt.colorbar(p)

        def animate(f):
            density, coords = read_density(self.fnames[f])
            sort_idx = sorted(range(len(coords)), key=coords.__getitem__)
            coords = [coords[i] for i in sort_idx]
            density = [density[i] for i in sort_idx]
            assert coords == poly_coords
            vmin, vmax, scaled_density = self.colscale(density)
            p.set_array(scaled_density)

            if not op.exists(self.path):
                os.mkdir(self.path)
            #calculate max padding required
            required_padding = len(str(len(self.times)))
            padding_format_code = '{0:0' + str(required_padding) + 'd}'
            figname = op.join(
                self.path, (padding_format_code).format(idx))
            plt.gcf().savefig(figname + ext, res=image_size, bbox_inches='tight')

            return p,

        ani = animation.FuncAnimation(fig, animate, frames=len(self.fnames), interval=1, blit=True, repeat=False)

        plt.show()


    def findDensityFileFromTime(self, time):
        for fname in self.fnames:
            path, filename = op.split(fname)
            tokens = filename.split('_')
            if tokens[2] == time:
                return fname
        return None

    # plot the density in file 'fname'. ax may be used to add to an existing
    # plot axis.
    def plotDensity(self, fname, image_size=300, colorbar=None, cmap='inferno', ax=None,
                     save=False, ext='.png'):
        if not ext.startswith('.'):
            ext = '.' + ext

        if ax is None:
            fig, ax = plt.subplots()

        time = get_density_time(fname)
        idx = self.times.index(time)
        md = self.mesh.dimensions()
        poly_coords = list(self.polygons.keys())
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])
        #aspect = (md[0][1] - md[0][0]) / (md[1][1] - md[1][0])
        #ax.set_aspect(aspect)
        ax.set_aspect('auto')
        p = PatchCollection(self.patches, cmap=cmap)
        density, coords = read_density(fname)
        sort_idx = sorted(range(len(coords)), key=coords.__getitem__)
        coords = [coords[i] for i in sort_idx]
        density = [density[i] for i in sort_idx]
        assert coords == poly_coords
        vmin, vmax, scaled_density = self.colscale(density)
        p.set_array(scaled_density)
        ax.add_collection(p)

        if colorbar is not None:
            plt.colorbar(p)

        if save:
            if not op.exists(self.path):
                os.mkdir(self.path)
            #calculate max padding required
            required_padding = len(str(len(self.times)))
            padding_format_code = '{0:0' + str(required_padding) + 'd}'
            figname = op.join(
                self.path, (padding_format_code).format(idx) + '_' +
                '{}'.format(time)).replace('.', '-')
            plt.gcf().savefig(figname + ext, res=image_size, bbox_inches='tight')
            plt.close(plt.gcf())
