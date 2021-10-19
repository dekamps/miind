#!/usr/bin/python

### 
# This script generates the C++ files Triangulator.hpp and Triangulator.cpp.
# The triangulation pattern for simplices of different dimensions
# is hard coded in C++. Currently, only 2D and 3D are supported but a new
# pair of files can be generated if higher dimensions are required.
### 

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import math
from scipy.special import gamma, factorial
from sympy.combinatorics import Permutation
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay 

class Point:
    def __init__(self, _coords):
        self.coords = np.array(_coords).tolist()
        self.connected = []
        self.lives = 0
        self.dead = False
        self.hyper = False

    def __eq__(self, other):
        return self.coords==other.coords

class Simplex:
    def __init__(self, _num_dimensions, _points, _triangulator):
        self.triangulator = _triangulator
        self.num_dimensions = _num_dimensions

        self.points = []
        for p in _points:
            self.points = self.points + [Point(p)]

        for p in range(len(self.points)):
            self.points[p].connected = [self.points[a] for a in range(len(self.points)) if a != p]

        self.lines = self.generate_lines()
        self.getVolume()

    def generate_lines(self):
        ls = []
        for p in range(len(self.points)-1):
            p1 = self.points[(p+1)].coords
            ls = ls + [np.array(p1) - np.array(self.points[0].coords)]
        return ls

    def getVolume(self):
        return abs(np.linalg.det(np.matrix(self.lines))/math.factorial(self.num_dimensions))

    def intersectWithHyperplane(self, dim_index, dim):
        eps = 0.00000000001
        p_outs = []

        lower = [p for p in self.points if p.coords[dim_index] < dim - eps]
        upper = [p for p in self.points if p.coords[dim_index] > dim + eps]

        for p0 in lower:
            for p1 in upper:
                t = (dim - p0.coords[dim_index]) / (p1.coords[dim_index]-p0.coords[dim_index])
                p_out = Point(np.array(p0.coords) + ((np.array(p1.coords)-np.array(p0.coords)) * t))
                p_out.connected = [p0,p1]
                p0.connected = [p_out if a == p1 else a for a in p0.connected]
                p1.connected = [p_out if a == p0 else a for a in p1.connected]
                p_out.hyper = True
                p_outs = p_outs + [p_out]

        # print('pouts:', len(p_outs))

        if len(p_outs) == 0:
            simplices = [Simplex(self.num_dimensions, [a.coords for a in self.points], self.triangulator)]
            greater_than = [s for s in simplices if np.all([p.coords[dim_index] >= dim for p in s.points])]
            less_than = [s for s in simplices if np.all([p.coords[dim_index] <= dim for p in s.points])]
            return [less_than,greater_than]


        index = 0
        i_less = [i + index for i in range(len(lower))]
        index += len(lower)

        i_greater = [i + index for i in range(len(upper))]
        index += len(upper)

        i_hyp = [i + index for i in range(len(p_outs))]
        index += len(p_outs)

        p_equal = [p for p in self.points if p.coords[dim_index] <= dim + eps and p.coords[dim_index] >= dim - eps]
        i_hyp = i_hyp + [i + index for i in range(len(p_equal))]

        p_total = lower + upper + p_outs + p_equal
        simplices = self.triangulator.chooseTriangulation(p_total, i_less, i_greater, i_hyp)

        # DRAW TRIANGULATION
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # for s in simplices:
        #     for p1 in s.points:
        #         for p2 in s.points:
        #             p1_red = [element for i, element in enumerate(p1.coords) if i in [0,1,2]]
        #             p2_red = [element for i, element in enumerate(p2.coords) if i in [0,1,2]]
        #
        #             line = np.array(p1_red)
        #             line = np.vstack((line, np.array(p2_red)))
        #             line = line.transpose()
        #             ax.plot(line[0], line[1], line[2], color='g')
        #
        # for p in p_outs:
        #     p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
        #     ax.scatter(p_red[0],p_red[1],p_red[2], color='b')
        #
        # for p in p_equal:
        #     p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
        #     ax.scatter(p_red[0],p_red[1],p_red[2], color='r')
        #
        # for p in upper:
        #     p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
        #     ax.scatter(p_red[0],p_red[1],p_red[2], color='k')
        #
        # for p in lower:
        #     p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
        #     ax.scatter(p_red[0],p_red[1],p_red[2], color='y')
        #
        # plt.show()
        # DEBUGGING

        greater_than = [s for s in simplices if np.all([p.coords[dim_index] >= dim for p in s.points])]
        less_than = [s for s in simplices if np.all([p.coords[dim_index] <= dim for p in s.points])]

        # Verify everything worked by summing the volume of the new simplices
        # vol_check = sum([s.getVolume() for s in simplices])
        # print([s.getVolume() for s in simplices])
        # print('Vol Check : parts = {}, whole = {}'.format(vol_check, self.getVolume()))
        return [less_than,greater_than]

class Cell:
    def __init__(self, _coords, _num_dimensions, _points, _triangulator):
        self.grid_coords = _coords
        self.num_dimensions = _num_dimensions
        self.triangulator = _triangulator
        self.points = _points
        self.simplices = self.generate_simplices(self.num_dimensions)
        self.hyps = self.calculateAAHyperplanes()

    def generate_simplices(self, dims):
        ### Based on js code by Mikola Lysenko (2014)
        ### https://github.com/mikolalysenko/triangulate-hypercube

        if dims < 0:
            return []

        if dims == 0:
            return [0]

        dfactorial = int(round(gamma(dims+1)))|0
        result = []

        for i in range(dfactorial):
            perm = Permutation(dims)
            p = perm.unrank_lex(dims,i)
            cell = [0]
            v = 0
            for j in range(p.size):
                v += (1 << (j^p))
                cell = cell + [v]
            if p.signature() < 1:
                cell[0] = v
                cell[dims] = 0

            result = result + [Simplex(dims, [self.points[a].coords for a in cell], self.triangulator)]

        return result

    def getVolume(self):
        sum = 0.0
        for s in self.simplices:
            sum = sum + s.getVolume()

        return sum

    def calculateAAHyperplanes(self):
        hyps = []
        for d in range(self.num_dimensions):
            hyps = hyps + [(d,min([a.coords[d] for a in self.points]),max([a.coords[d] for a in self.points]))]

        return hyps

    def drawCellInPlot(self, axis, col, dims, check_edges = False):
        for p1 in self.points:
            for p2 in self.points:
                p1_red = [element for i, element in enumerate(p1.coords) if i in dims]
                p2_red = [element for i, element in enumerate(p2.coords) if i in dims]
                if check_edges:
                    sim_count = [i for i, e in enumerate(p1_red) if e != p2_red[i]]
                    if len(sim_count) != 1:
                        continue
                line = np.array(p1_red)
                line = np.vstack((line, np.array(p2_red)))
                line = line.transpose()
                axis.plot(line[0], line[1], line[2], color=col)

    def intersectWith(self, other):
        # vol_eps = 0.000001
        vol_eps = 0.0000000000001
        simplices = [Simplex(self.num_dimensions, [p.coords for p in s.points], self.triangulator) for s in self.simplices]
        for (d,mn,mx) in other.hyps:
            next_simplices = []
            for s in simplices:
                if s.getVolume() < vol_eps:
                    continue
                next_simplices = next_simplices + s.intersectWithHyperplane(d,mn)[1]
            simplices = next_simplices
            next_simplices = []
            for s in simplices:
                if s.getVolume() < vol_eps:
                    continue
                next_simplices = next_simplices + s.intersectWithHyperplane(d,mx)[0]
            simplices = next_simplices

        # DRAW TRIANGULATION
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # for s in simplices:
        #     for p1 in s.points:
        #         for p2 in s.points:
        #             p1_red = [element for i, element in enumerate(p1.coords) if i in [0,1,2]]
        #             p2_red = [element for i, element in enumerate(p2.coords) if i in [0,1,2]]
        #             line = np.array(p1_red)
        #             line = np.vstack((line, np.array(p2_red)))
        #             line = line.transpose()
        #             ax.plot(line[0], line[1], line[2], color='g')
        # plt.show()
        # DEBUGGING

        return sum([s.getVolume()/self.getVolume() for s in simplices])

class Triangulator:
    def __init__(self, _num_dimensions):
        self.num_dimensions = _num_dimensions
        self.dimensions = [1.0 for i in range(self.num_dimensions)]
        self.resolution = [1 for i in range(self.num_dimensions)]
        self.base = [0.0 for i in range(self.num_dimensions)]
        self.points = self.generate_points([], self.base, self.dimensions, self.resolution)
        self.cells = self.generate_cells([],self.resolution)
        self.triangulations = self.generateTriangulationLookup()

    def generate_points(self, total, base, dim, res):
        dh, *dt = dim
        rh, *rt = res

        if len(dt) == 0:
            ps = []
            for d in range(rh+1):
                ps = ps + [Point((np.array(base) + np.array(total + [d * (dh / rh)])).tolist())]
            return ps

        ps = []
        for d in range(rh+1):
            ps = ps + [self.generate_points(total + [d*(dh/rh)], base, dt, rt)]
        return ps

    def int_to_bin_list(self, i):
        bstring = str(bin(i))
        return [int(a) for a in bstring[2:].zfill(self.num_dimensions)]

    def cell_index_list(self):
        return [self.int_to_bin_list(a) for a in range(2**self.num_dimensions)]

    def get_cell_indices(self, cell):
        return [(np.array(a) + np.array(cell)).tolist() for a in self.cell_index_list()]

    def getCellCornerPoint(self, coords):
        check = self.points
        for c in coords:
            check = check[c]
        return check

    def generate_cells(self, cell, res):
        rh, *rt = res

        cells = []
        if len(rt) == 0:
            for r in range(rh):
                cells = cells + [Cell(cell + [r], self.num_dimensions, [self.getCellCornerPoint(a) for a in self.get_cell_indices(cell + [r])], self)]
            return cells

        for d in range(rh):
            cells = cells + [self.generate_cells(cell + [d], rt)]

        return cells

    def flatten(self, cs, cells):
        for c in cells:
            if len(np.array(c).shape) == 1:
                cs = cs + c
            else:
                cs = self.flatten(cs, c)

        return cs

    def generateTriangulationLookup(self):
        print('Precalculating possible simplex triangulations...')
        simps = {}
        num_points = self.num_dimensions + 1
        # list possible combinations of points above and below the hyperplane
        # number of hyperplane points will be above*below
        combs = [(a+1, num_points - (a+1), (a+1)*(num_points - (a+1))) for a in range(int(num_points/2))]
        combs = combs + [(num_points - (a+1), a+1, (a+1)*(num_points - (a+1))) for a in range(int(num_points/2))]

        cs = []
        for (a,b,c) in combs:
            for i in range(a):
                for j in range(b):
                    if (a-i,b-j,c) not in cs:
                        cs = cs + [(a-i,b-j,c)]
        combs = cs

        # place points on a unit (hyper) cube and grab a simplex
        unit_cell = self.generate_cells([],[1 for i in range(self.num_dimensions)])
        unit_cell = self.flatten([], unit_cell)[0]
        unit_simplex = unit_cell.simplices[0]

        # split simplex into upper and lower points
        for (a,b,c) in combs:
            # If there are some points sitting on the hyperplane, then
            # we need to handle that
            num_points_on_hyp = num_points - (a+b)
            hyps = []
            if a not in simps:
                simps[a] = {}
            for lo in range(a):
                if b not in simps[a]:
                    simps[a][b] = {}
                for hi in range(b):
                    hi = hi + a
                    p = np.array(unit_simplex.points[lo].coords) + (np.array(unit_simplex.points[hi].coords) - np.array(unit_simplex.points[lo].coords)) / 2.0
                    p = p.tolist()
                    hyps = hyps + [Point(p)]
            for extra in range(num_points_on_hyp):
                hyps = hyps + [Point(unit_simplex.points[(a+b)+extra].coords)]

            all_points = [Point(a.coords) for a in (unit_simplex.points[:(a+b)] + hyps)]
            tri = Delaunay([a.coords for a in all_points])

            # DRAW UNIT CUBE TRIANGULATIONS
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            
            #for s in tri.simplices:
            #    for p1 in s:
            #        for p2 in s:
            #            p1_red = [element for i, element in enumerate(all_points[p1].coords) if i in [0,1,2]]
            #            p2_red = [element for i, element in enumerate(all_points[p2].coords) if i in [0,1,2]]
            #            line = np.array(p1_red)
            #            line = np.vstack((line, np.array(p2_red)))
            #            line = line.transpose()
            #            ax.plot(line[0], line[1], line[2], color='g')
            
            #for p in hyps:
            #    p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
            #    ax.scatter(p_red[0],p_red[1],p_red[2], color='b')
            
            
            #for p in unit_simplex.points:
            #    p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
            #    ax.scatter(p_red[0],p_red[1],p_red[2], color='k')
            
            #plt.show()
            # DEBUGGING

            # DRAW UNIT SQUARE TRIANGULATIONS
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            
            #for s in tri.simplices:
            #    for p1 in s:
            #        for p2 in s:
            #            p1_red = [element for i, element in enumerate(all_points[p1].coords) if i in [0,1,2]]
            #            p2_red = [element for i, element in enumerate(all_points[p2].coords) if i in [0,1,2]]
            #            line = np.array(p1_red)
            #            line = np.vstack((line, np.array(p2_red)))
            #            line = line.transpose()
            #            ax.plot(line[0], line[1], color='g')
            
            #for p in hyps:
            #    p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
            #    ax.scatter(p_red[0],p_red[1], color='b')
            
            
            #for p in unit_simplex.points:
            #    p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
            #    ax.scatter(p_red[0],p_red[1], color='k')
            
            #plt.show()
            # DEBUGGING

            simps[a][b][len(hyps)] = tri.simplices
        return simps

    def gen_simplices_for_cell(self, num_dims):
        ### Based on js code by Mikola Lysenko (2014)
        ### https://github.com/mikolalysenko/triangulate-hypercube

        dims = num_dims
        result = []

        if dims < 0:
            result = []
        elif dims == 0:
            result = [0]
        else:
            dfactorial = int(round(gamma(dims+1)))|0
            
            for i in range(dfactorial):
                perm = Permutation(dims)
                p = perm.unrank_lex(dims,i)
                cell = [0]
                v = 0
                for j in range(p.size):
                    v += (1 << (j^p))
                    cell = cell + [v]
                if p.signature() < 1:
                    cell[0] = v
                    cell[dims] = 0

                result = result + [cell]

        return result

    def generate_cell_simplices(self):
        cpp  = 'std::vector<Simplex> Triangulator::generateCellSimplices(unsigned int num_dimensions, std::vector<NdPoint>& points) {\n'
        cpp += '\tswitch(num_dimensions) {\n'
        for i in range(self.num_dimensions-1):
            i = i + 2
            cpp += '\tcase {}: {{\n'.format(i)
            indices = self.gen_simplices_for_cell(i)
            cpp += '\t\tstd::vector<Simplex> simplices;\n'
            for j in range(len(indices)):
                cpp += '\t\tstd::vector<NdPoint> ps_{}({});\n'.format(j,len(indices[j]))
                for k in range(len(indices[j])):
                    cpp += '\t\tps_{}[{}] = points[{}];\n'.format(j,k, indices[j][k])
                cpp += '\t\tsimplices.push_back(Simplex(num_dimensions,ps_{},*this));\n'.format(j)
            cpp += '\t\treturn simplices;\n'
            cpp += '\t}\n'
        
        cpp += '\tdefault: {\n'
        cpp +='\t\treturn std::vector<Simplex>();\n'
        cpp += '\t}\n'
        cpp += '\t}\n'
        cpp += '}\n'

        return cpp

    def generate_cell_cube_unit_points(self):     
        cpp  = 'std::vector<NdPoint> Triangulator::generateUnitCubePoints(unsigned int num_dimensions) {\n'
        cpp += '\tswitch(num_dimensions) {\n'
        for i in range(self.num_dimensions-1):
            i = i + 2
            cpp += '\tcase {}: {{\n'.format(i)
            tri = Triangulator(i)
            unit_cell = tri.generate_cells([],[1 for a in range(tri.num_dimensions)])
            unit_cell = tri.flatten([], unit_cell)[0]
            cpp += '\t\tstd::vector<NdPoint> points({});\n'.format(2**i)
            for j in range(len(unit_cell.points)):
                cpp += '\t\tstd::vector<double> coords_{}({});\n'.format(j,i)
                for k in range(i):
                    cpp += '\t\tcoords_{}[{}] = {};\n'.format(j,k,unit_cell.points[j].coords[k])
                cpp += '\t\tpoints[{}] = NdPoint(coords_{});\n'.format(j,j)
            cpp += '\t\treturn points;\n'
            cpp += '\t}\n'
        
        cpp += '\tdefault: {\n'
        cpp +='\t\treturn std::vector<NdPoint>();\n'
        cpp += '\t}\n'
        cpp += '\t}\n'
        cpp += '}\n'

        return cpp

    def generate_triangulations_cpp(self):
        cpp  = '#include "NdPoint.hpp"\n'
        cpp += '#include "NdCell.hpp"\n'
        cpp += '#include "Simplex.hpp"\n'
        cpp += '#include "Triangulator.hpp"\n'
        cpp += '\n\n'
        cpp += 'std::vector<Simplex> Triangulator::chooseTriangulation(unsigned int num_dimensions, std::vector<NdPoint>& points, std::vector<unsigned int>& lower_inds, std::vector<unsigned int>& upper_inds, std::vector<unsigned int>& hyper_inds) {\n'
        cpp += '\tif (lower_inds.size() == 0){\n'
        cpp += '\t\tstd::vector<Simplex> out;\n'
        cpp += '\t\tstd::vector<NdPoint> ps(upper_inds.size());\n'
        cpp += '\t\tfor (unsigned int i=0; i<upper_inds.size(); i++)\n'
        cpp += '\t\t\tps[i] = points[upper_inds[i]];\n'
        cpp += '\t\tout.push_back(Simplex(num_dimensions, ps, *this));\n'
        cpp += '\t\treturn out;\n'
        cpp += '\t}\n'
        cpp += '\tif (upper_inds.size() == 0){\n'
        cpp += '\t\tstd::vector<Simplex> out;\n'
        cpp += '\t\tstd::vector<NdPoint> ps(lower_inds.size());\n'
        cpp += '\t\tfor (unsigned int i=0; i<lower_inds.size(); i++)\n'
        cpp += '\t\t\tps[i] = points[lower_inds[i]];\n'
        cpp += '\t\tout.push_back(Simplex(num_dimensions, ps, *this));\n'
        cpp += '\t\treturn out;\n'
        cpp += '\t}\n'
        cpp += '\tstd::vector<std::vector<unsigned int>> tris = transitions[lower_inds.size()][upper_inds.size()][hyper_inds.size()];\n'
        cpp += '\tstd::vector<Simplex> out;\n'
        cpp += '\tfor (unsigned int t=0; t <tris.size(); t++) {\n'
        cpp += '\t\tstd::vector<NdPoint> ps(tris[t].size());\n'
        cpp += '\t\tfor (unsigned int i=0; i<tris[t].size(); i++)\n'
        cpp += '\t\t\tps[i] = points[tris[t][i]];\n'
        cpp += '\t\tout.push_back(Simplex(num_dimensions, ps, *this));\n'
        cpp += '\t}\n'
        cpp += '\treturn out;\n'
        cpp += '}\n'
        cpp += '\n'

        return cpp

    def generate_triangulations_hpp(self):
        cpp  = '#ifndef APP_ND_GRID_TRIANGULATOR\n'
        cpp += '#define APP_ND_GRID_TRIANGULATOR\n\n'
        cpp += '#include <map>\n'
        cpp += '#include <vector>\n'
        cpp += '\n\n'
        cpp += 'class NdPoint;\n'
        cpp += 'class Simplex;\n'
        cpp += '\n'
        cpp += 'class Triangulator {\n'
        cpp += 'public:\n'
        cpp += '\tstd::vector<std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>>> transitions;\n'
        cpp += '\tTriangulator() {\n'
        
        transitions_dims = [Triangulator(n).triangulations for n in range(self.num_dimensions+1) if n >= 2]
        
        cpp += '\t\ttransitions = std::vector<std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>>>();\n'
        cpp += '\t\tstd::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>> t;\n'
        for transitions in transitions_dims:
            cpp += '\t\tt = std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>>();\n'
            for ih,it in transitions.items():
                cpp += '\t\tt[{}] = std::map<unsigned int, std::map<unsigned int, std::vector<std::vector<unsigned int>>>>();\n'.format(ih) 
                for jh,jt in it.items():
                    cpp += '\t\tt[{}][{}] = std::map<unsigned int, std::vector<std::vector<unsigned int>>>();\n'.format(ih, jh) 
                    for kh,kt in jt.items():
                        cpp += '\t\tt[{}][{}][{}] = std::vector<std::vector<unsigned int>>({});\n'.format(ih, jh, kh, len(kt)) 
                        for l in range(len(kt)):
                            cpp += '\t\tt[{}][{}][{}][{}] = std::vector<unsigned int>({});\n'.format(ih, jh, kh, l, len(kt[l]))
                            for m in range(len(kt[l])):
                                cpp += '\t\tt[{}][{}][{}][{}][{}] = {};\n'.format(ih, jh, kh, l, m, kt[l][m])
            cpp += '\t\ttransitions.push_back(t);\n'
            
        cpp += '\t}\n'
        cpp += '\n\n'
        cpp += '\tstd::vector<Simplex> chooseTriangulation(unsigned int num_dimensions, std::vector<NdPoint>& points, std::vector<unsigned int>& lower_inds, std::vector<unsigned int>& upper_inds, std::vector<unsigned int>& hyper_inds);\n'
        cpp += '\tstd::vector<Simplex> generateCellSimplices(unsigned int num_dimensions, std::vector<NdPoint>& points);\n'
        cpp += '\tstd::vector<NdPoint> generateUnitCubePoints(unsigned int num_dimensions);\n'
        cpp += '};\n'
        cpp += '\n#endif\n'

        return cpp


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Please provide a maximum number of dimensions as an argument.')
    else:
        t = Triangulator(int(sys.argv[1]))
        with open('Triangulator.cpp', 'w') as f:
            f.write(t.generate_triangulations_cpp())
            f.write(t.generate_cell_simplices())
            f.write(t.generate_cell_cube_unit_points())
        with open('Triangulator.hpp', 'w') as f:
            f.write(t.generate_triangulations_hpp())


