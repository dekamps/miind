import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import math
from scipy.special import gamma, factorial
from sympy.combinatorics import Permutation
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

import miind.miind_api as api

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

class Grid:
    def __init__(self, _base, _dimensions, _resolution, _func, _threshold_v, _timestep=0.001):
        self.timestep = _timestep
        self.num_dimensions = len(_dimensions)
        self.threshold_v = _threshold_v
        self.triangulator = Triangulator(self.num_dimensions)
        # Reverse all our resolution parameters so it works out that the x dimension
        # is always the first dimension in the grid
        self.dimensions = _dimensions
        self.dimensions.reverse()
        self.resolution = _resolution
        self.resolution.reverse()
        self.base = _base
        self.base.reverse()
        print('Generating the regular and transformed grids...')
        self.points = self.generate_points([], self.base, self.dimensions, self.resolution)
        self.points_trans = self.generate_points_trans(_func, _timestep, 0.0001, [], self.base, self.dimensions, self.resolution)
        self.cells = self.generate_cells([],self.resolution)
        self.cells_trans = self.generate_cells_trans([],self.resolution)

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

    def generate_points_trans(self, func, timestep, tolerance, total, base, dim, res):
        tspan = np.linspace(0, timestep, 11)
        dh, *dt = dim
        rh, *rt = res

        if len(dt) == 0:
            ps = []
            for d in range(rh+1):
                t_1 = odeint(func, (np.array(base) + np.array(total + [d * (dh / rh)])).tolist(), tspan, atol=tolerance, rtol=tolerance)
                next = t_1[-1]
                ps = ps + [Point(next)]
            return ps

        ps = []
        for d in range(rh+1):
            ps = ps + [self.generate_points_trans(func, timestep, tolerance, total + [d*(dh/rh)], base, dt, rt)]
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

    def getCellTransCornerPoint(self, coords):
        check = self.points_trans
        for c in coords:
            check = check[c]
        return check

    def generate_cells(self, cell, res):
        rh, *rt = res

        cells = []
        if len(rt) == 0:
            for r in range(rh):
                cells = cells + [Cell(cell + [r], self.num_dimensions, [self.getCellCornerPoint(a) for a in self.get_cell_indices(cell + [r])], self.triangulator)]
            return cells

        for d in range(rh):
            cells = cells + [self.generate_cells(cell + [d], rt)]

        return cells

    def generate_cells_trans(self, cell, res):
        rh, *rt = res

        cells = []
        if len(rt) == 0:
            for r in range(rh):
                cells = cells + [Cell(cell + [r], self.num_dimensions, [self.getCellTransCornerPoint(a) for a in self.get_cell_indices(cell + [r])], self.triangulator)]
            return cells

        for d in range(rh):
            cells = cells + [self.generate_cells_trans(cell + [d], rt)]

        return cells

    def findCellCoordForPoint(self, point):
        rel_point = (np.array(point.coords) - np.array(self.base)).tolist()
        coords = []
        for i in range(len(rel_point)):
            c = int(rel_point[i] / (self.dimensions[i]/self.resolution[i]))
            if c > (self.resolution[i]-1):
                c = self.resolution[i]-1
            if c < 0:
                c = 0
            coords = coords + [c]
        return coords

    def calculateCellRange(self, tcell):
        mxs = [c for c in tcell.points[0].coords]
        mns = [c for c in tcell.points[0].coords]
        for p in tcell.points:
            for i in range(len(p.coords)):
                if mxs[i] < p.coords[i]:
                    mxs[i] = p.coords[i]
                if mns[i] > p.coords[i]:
                    mns[i] = p.coords[i]
        return [self.findCellCoordForPoint(Point(mns)), self.findCellCoordForPoint(Point(mxs))]

    def calculateTransitionForCell(self, tcell, cells, target, cell_range):
        min_head, *min_tail = cell_range[0]
        max_head, *max_tail = cell_range[1]

        if len(min_tail) == 0:
            transitions = []
            for c in range((max_head - min_head)+1):
                prop = tcell.intersectWith(cells[min_head + c])
                # print(target + [c], prop)
                if prop > 0:
                    transitions = transitions + [[target + [min_head + c],prop]]

            return transitions

        transitions = []
        for c in range((max_head - min_head)+1):
            transitions = transitions + self.calculateTransitionForCell(tcell, cells[min_head + c], target + [min_head + c], [min_tail, max_tail])
        return transitions


    def calculateTransitionMatrix(self):
        cells_to_check = self.flatten([], self.cells_trans)
        num_cells_to_check = len(cells_to_check)
        current_cell = 0
        percent_complete = 0
        cells_in_ten_percent = num_cells_to_check / 10
        transitions = []
        threshold_coord = self.findCellCoordForPoint(Point(np.zeros(self.num_dimensions-1).tolist() + [self.threshold_v]))[-1]
        for tcell in cells_to_check:
            current_cell += 1
            if current_cell % cells_in_ten_percent == 0:
                percent_complete += 10
                print(str(percent_complete) + '% complete.')

            tcell_transitions = self.calculateTransitionForCell(tcell, self.cells, [], self.calculateCellRange(tcell))
            
            # if there were no transitions, the cell was entirely outside everything
            if len(tcell_transitions) == 0:
                tcell_transitions = [[tcell.grid_coords, 1.0]]

            # if cell ends up bove threshold, send it all back to threshold
            tcell_transitions = [[coord, prop] if coord[-1] <= threshold_coord else [coord[:-1] + [threshold_coord], prop] for [coord, prop] in tcell_transitions ]

            # If there's any mass left over, spread it among
            # all cells : this is a hacky solution to maintaining mass
            # at the edge of the grid: the actual behaviour of the mass near the
            # edge shouldn't matter too much so hopefully we can get away with
            # this.
            total_mass = sum([a[1] for a in tcell_transitions])
            extra_mass_share = (1.0-total_mass) / len(tcell_transitions)
            transitions = transitions + [[tcell.grid_coords, [[coord, mass + extra_mass_share] for [coord, mass] in tcell_transitions]]]
        return transitions

    def plotGrid(self):
        cs = self.flatten([],self.cells)
        csts = self.flatten([],self.cells_trans)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c in cs:
            c.drawCellInPlot(ax, 'k', [0,1,2], True)
        for c in csts:
            c.drawCellInPlot(ax, 'r', [0,1,2])
        plt.show()

    def flatten(self, cs, cells):
        # just a single array
        if len(np.array(cells).shape) == 1:
            cs = cells
        else:
            for c in cells:
                cs = cs + self.flatten([], c)
        return cs

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
        points = self.generate_points([], [0.0 for i in range(self.num_dimensions)], [1.0 for i in range(self.num_dimensions)], [1 for i in range(self.num_dimensions)])
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
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # for s in tri.simplices:
            #     for p1 in s:
            #         for p2 in s:
            #             p1_red = [element for i, element in enumerate(all_points[p1].coords) if i in [0,1,2]]
            #             p2_red = [element for i, element in enumerate(all_points[p2].coords) if i in [0,1,2]]
            #             line = np.array(p1_red)
            #             line = np.vstack((line, np.array(p2_red)))
            #             line = line.transpose()
            #             ax.plot(line[0], line[1], line[2], color='g')
            #
            # for p in hyps:
            #     p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
            #     ax.scatter(p_red[0],p_red[1],p_red[2], color='b')
            #
            #
            # for p in unit_simplex.points:
            #     p_red = [element for i, element in enumerate(p.coords) if i in [0,1,2]]
            #     ax.scatter(p_red[0],p_red[1],p_red[2], color='k')
            #
            # plt.show()
            # DEBUGGING

            simps[a][b][len(hyps)] = tri.simplices
        return simps

    def chooseTriangulation(self, points, lower_inds, upper_inds, hyper_inds):
        if len(lower_inds) == 0:
            return Simplex(self.num_dimensions, [p.coords for p in [points[i] for i in upper_inds]], self)

        if len(upper_inds) == 0:
            return Simplex(self.num_dimensions, [p.coords for p in [points[i] for i in lower_inds]], self)

        tris = self.triangulations[len(lower_inds)][len(upper_inds)][len(hyper_inds)]
        total_inds = lower_inds + upper_inds + hyper_inds

        return [Simplex(self.num_dimensions, [p.coords for p in [points[total_inds[i]] for i in s]], self) for s in tris]

class MiindGrid3DFileGenerator:
    def __init__(self, _grid, _basename, _timescale, _reset_v, _threshold_v, _relative_reset_w, _reset_func=None, _reset_w_func=None):
        self.grid = _grid
        self.basename = _basename
        self.timescale = _timescale
        self.reset_v = _reset_v
        self.threshold_v = _threshold_v
        self.relative_reset_w = _relative_reset_w
        self.reset_func = _reset_func
        self.reset_w_func = _reset_w_func

    def generateTMatFile(self):
        transitions = self.grid.calculateTransitionMatrix()
        print(transitions)

        with open(self.basename + '.tmat', 'w') as tmat_file:
            tmat_file.write('0	0\n')
            for c in transitions:
                line = '1000000000;' + str(c[0][0]) + ',' + str(c[0][1]) + ';'
                for t in c[1]:
                    line += str(t[0][0]) + ',' + str(t[0][1]) + ':' + '{:.9f}'.format(t[1]) + ';'
                line += '\n'
                tmat_file.write(line)


class MiindGrid2DFileGenerator:
    def __init__(self, _grid, _basename, _timescale, _reset_v, _threshold_v, _relative_reset_w, _reset_func=None, _reset_w_func=None):
        self.grid = _grid
        self.basename = _basename
        self.timescale = _timescale
        self.reset_v = _reset_v
        self.threshold_v = _threshold_v
        self.relative_reset_w = _relative_reset_w
        self.reset_func = _reset_func
        self.reset_w_func = _reset_w_func

    def generateRevFile(self):
        with open(self.basename + '.rev', 'w') as rev_file:
            rev_file.write('<Mapping Type="Reversal">\n')
            rev_file.write('</Mapping>\n')

    def generateStateFile(self):
        with open(self.basename + '.stat', 'w') as stat_file:
            stat_file.write('<Stationary>\n')
            stat_file.write('</Stationary>\n')

    def generateMeshFile(self):
        with open(self.basename + '.mesh', 'w') as mesh_file:
            mesh_file.write('ignore\n')
            mesh_file.write('{}\n'.format(self.grid.timestep*self.timescale))

            for strip in self.grid.cells:
                sus_1 = [strip[0].points[0].coords[0]]
                svs_1 = [strip[0].points[0].coords[1]]
                sus_2 = [strip[0].points[2].coords[0]]
                svs_2 = [strip[0].points[2].coords[1]]
                for cell in strip:
                    sus_1 = sus_1 + [cell.points[1].coords[0]]
                    svs_1 = svs_1 + [cell.points[1].coords[1]]
                    sus_2 = sus_2 + [cell.points[3].coords[0]]
                    svs_2 = svs_2 + [cell.points[3].coords[1]]

                for s in svs_1:
                    mesh_file.write('{}\t'.format(s))
                mesh_file.write('\n')
                for s in sus_1:
                    mesh_file.write('{}\t'.format(s))
                mesh_file.write('\n')
                for s in svs_2:
                    mesh_file.write('{}\t'.format(s))
                mesh_file.write('\n')
                for s in sus_2:
                    mesh_file.write('{}\t'.format(s))
                mesh_file.write('\n')
                mesh_file.write('closed\n')

            mesh_file.write('end\n')

    def generateModelFile(self):
        self.generateRevFile()
        self.generateStateFile()
        self.generateMeshFile()
        api.MeshTools.buildModelFileFromMesh(self.basename, self.reset_v, self.threshold_v)
        self.generateResetMapping()

    def generateResetMapping(self):
        total_string = ''
        with open(self.basename + '.model', 'r') as model_file:
            lines = model_file.readlines()
            for l in range(len(lines)-1): # ignore last line </Model>
                if l != 3:
                    total_string += lines[l]

        mapping_string  = '<Mapping type="Reset">\n'
        reset_index = int((self.reset_v - self.grid.base[1]) / (self.grid.dimensions[1] / self.grid.resolution[1]))
        threshold_index = int((self.threshold_v - self.grid.base[1]) / (self.grid.dimensions[1] / self.grid.resolution[1]))
        w_int_offset = int(self.relative_reset_w / (self.grid.dimensions[0] / self.grid.resolution[0]))
        signed_offset = (self.relative_reset_w / (self.grid.dimensions[0] / self.grid.resolution[0]))
        w_offset_1 = abs((self.relative_reset_w / (self.grid.dimensions[0] / self.grid.resolution[0])) - w_int_offset)
        w_offset_2 = 1.0 - w_offset_1
        for strip_num in range(len(self.grid.cells)):
            if self.reset_func:
                reset_index = int((self.reset_func(self.grid.cells[strip_num][0].points[0].coords[0]) - self.grid.base[1]) / (self.grid.dimensions[1] / self.grid.resolution[1]))

            if self.reset_w_func:
                w_int_offset = int(self.reset_w_func(self.grid.cells[strip_num][0].points[0].coords[0]) / (self.grid.dimensions[0] / self.grid.resolution[0]))
                signed_offset = (self.reset_w_func(self.grid.cells[strip_num][0].points[0].coords[0]) / (self.grid.dimensions[0] / self.grid.resolution[0]))
                w_offset_1 = abs((self.reset_w_func(self.grid.cells[strip_num][0].points[0].coords[0]) / (self.grid.dimensions[0] / self.grid.resolution[0])) - w_int_offset)
                w_offset_2 = 1.0 - w_offset_1

            strip = self.grid.cells[strip_num]

            strip_index_lower = strip_num + w_int_offset
            if signed_offset >= 0:
                strip_index_upper = strip_index_lower + 1
            else:
                strip_index_upper = strip_index_lower - 1

            if strip_index_upper > len(self.grid.cells) - 1:
                strip_index_lower = len(self.grid.cells) - 1
                strip_index_upper = strip_index_lower
                mapping_string += str(strip[threshold_index].grid_coords[0]) + ',' + str(strip[threshold_index].grid_coords[1]) + '\t' + str(strip_index_lower) + ',' + str(reset_index) + '\t' + str(1.0) + '\n'
            elif strip_index_lower < 1:
                strip_index_lower = 0
                strip_index_upper = strip_index_lower
                mapping_string += str(strip[threshold_index].grid_coords[0]) + ',' + str(strip[threshold_index].grid_coords[1]) + '\t' + str(strip_index_lower) + ',' + str(reset_index) + '\t' + str(1.0) + '\n'
            else:
                mapping_string += str(strip[threshold_index].grid_coords[0]) + ',' + str(strip[threshold_index].grid_coords[1]) + '\t' + str(strip_index_lower) + ',' + str(reset_index) + '\t' + str(w_offset_2) + '\n'
                mapping_string += str(strip[threshold_index].grid_coords[0]) + ',' + str(strip[threshold_index].grid_coords[1]) + '\t' + str(strip_index_upper) + ',' + str(reset_index) + '\t' + str(w_offset_1) + '\n'
        mapping_string += '</Mapping>\n'

        total_string += mapping_string
        total_string += '</Model>\n'

        with open(self.basename + '.model', 'w') as model_file:
            model_file.write(total_string)

    def generateTMatFile(self):
        transitions = self.grid.calculateTransitionMatrix()

        with open(self.basename + '.tmat', 'w') as tmat_file:
            tmat_file.write('0	0\n')
            for c in transitions:
                line = '1000000000;' + str(c[0][0]) + ',' + str(c[0][1]) + ';'
                for t in c[1]:
                    line += str(t[0][0]) + ',' + str(t[0][1]) + ':' + '{:.9f}'.format(t[1]) + ';'
                line += '\n'
                tmat_file.write(line)


#5D?!?
# g = Grid([-40.0,0.0,0.0,0.0,0.0], [1.0,1.0,1.0,1.0,1.0], [3,3,3,3,3])
# print([a.coords for a in g.cells[0][0][0][0][0].simplices[0].points])
# print([a.coords for a in g.cells_trans[0][0][0][0][0].simplices[0].points])
# print('Intersect with hyperplane at x = 0.5')
# print(g.cells[0][0][0][0][0].simplices[0].intersectWithHyperplane(1,0.5))
# print('Cell volume...')
# print(g.cells[0][0][0][0][0].getVolume())
# print('Cells overlap: ')
# whole_vol = g.cells_trans[0][0][0][0][0].getVolume()
# print('Volume = ', g.cells_trans[1][1][1][1][1].intersectWith(g.cells[1][1][1][1][1]))
# print('Whole volume = ', whole_vol)
# transition = g.calculateTransitionForCell(g.cells_trans[1][1][1][1][1], g.cells, [], g.calculateCellRange(g.cells_trans[1][1][1][1][1]))
# print(transition)
# print('total: ', sum([a[1] for a in transition]))

# 4D
# g = Grid([-120,-0.4,-0.4,-0.4], [100.0,1.4,1.4,1.4], [10,10,10,10])
# g = Grid([-40.0,0.0,0.0,0.0], [1.0,1.0,1.0,1.0], [3,3,3,3])
# print([a.coords for a in g.cells[0][0][0][0].simplices[0].points])
# print([a.coords for a in g.cells_trans[0][0][0][0].simplices[0].points])
# print('Intersect with hyperplane at x = 0.5')
# print(g.cells[0][0][0][0].simplices[0].intersectWithHyperplane(1,0.5))
# print('Cell volume...')
# print(g.cells[0][0][0][0].getVolume())
# print('Cells overlap: ')
# whole_vol = g.cells_trans[0][0][0][0].getVolume()
# print('Volume = ', g.cells_trans[1][1][1][1].intersectWith(g.cells[1][1][1][1]))
# print('Whole volume = ', whole_vol)
# transition = g.calculateTransitionForCell(g.cells_trans[1][1][1][1], g.cells, [], g.calculateCellRange(g.cells_trans[1][1][1][1]))
# print(transition)
# print('total: ', sum([a[1] for a in transition]))
# g.plotGrid()

# 3D
# g = Grid([-40.0,0.0,0.0], [1.0,1.0,1.0], [3,3,3])
# print([a.coords for a in g.cells[0][0][0].simplices[0].points])
# print([a.coords for a in g.cells_trans[0][0][0].simplices[0].points])
# print('Intersect with hyperplane at x = 0.5')
# print(g.cells[0][0][0].simplices[0].intersectWithHyperplane(1,0.5))
# print('Cell volume...')
# print(g.cells[0][0][0].getVolume())
# print('Cells overlap: ')
# whole_vol = g.cells_trans[0][0][0].getVolume()
# print('Volume = ', g.cells_trans[1][1][1].intersectWith(g.cells[1][1][1]))
# print('Whole volume = ', whole_vol)
# transition = g.calculateTransitionForCell(g.cells_trans[1][1][1], g.cells, [], g.calculateCellRange(g.cells_trans[1][1][1]))
# print(transition)
# print('total: ', sum([a[1] for a in transition]))
# g.plotGrid()

def cond3d(y, t):
    tau_m = 20e-3
    E_r = -65e-3
    E_e = 0.0
    tau_s = 5e-3
    tau_t = 5e-3
    
    v = y[2]
    w = y[1]
    u = y[0]
    
    v_prime = (-(v-E_r) - w*(v-E_e) - u*(v-E_e))/tau_m
    w_prime = -w/tau_s
    u_prime = -u/tau_t

    return [u_prime, w_prime, v_prime]

g = Grid([-2.0,-2.0,-66e-3], [2.2,2.2,12e-3], [50,100,100], cond3d, -55e-3, 0.00001)
three_d_gen = MiindGrid3DFileGenerator(g, 'cond3D', 1, -65e-3, -55e-3, 0.0)
three_d_gen.generateTMatFile()

# 2D

#def rybak(y, t):
#    g_nap = 0.25 #mS
#    g_na = 30
#    g_k = 11.2 # Taken from Butera, Rinzel, and Smith 1999 # In the Rybak paper, this is quoted as g_k = 1 but we get no oscillations with that value
#    # this is the only change that needs to be made to make it all work.
#    E_na = 55 #mV
#    E_k = -80
#    g_l = 0.1 #mS
#    E_l = -64.0 #mV
#    I = 0.0 #
#    I_h = 0

#    v = y[1]
#    h_na = ((1 + (np.exp((v + 55)/7)))**(-1)) #0.62
#    h_nap = y[0]
#    m_k = ((1 + (np.exp(-(v + 28)/15)))**(-1))

#    I_nap = -g_nap * h_nap * (v - E_na) * ((1 + np.exp(-(v+47.1)/3.1))**-1)
#    I_l = -g_l*(v - E_l)
#    I_na = -g_na * h_na * (v - E_na) * (((1 + np.exp(-(v+35)/7.8))**-1)**3)
#    I_k = -g_k * ((m_k)**4) * (v - E_k)

#    v_prime = I_nap + I_na + I_k + I_l + I

#    part_1 = ((1 + (np.exp((v + 59)/8)))**(-1)) - h_nap
#    part_2 = (1200/np.cosh((v + 59)/(16)))
#    h_nap_prime = part_1 / part_2

#    return [h_nap_prime, v_prime]

#def resetFunc(w):
#    return min(-40,-55 + (w * 50.0))

#def resetFuncW(w):
#    return -((1.0 - w) * 0.01)

#g = Grid([-180.0,-0.4], [160.0,1.4], [300,200], rybak, -29.0, 0.01)
#two_d_gen = MiindGrid2DFileGenerator(g, 'rybakND', 0.001, -57.0, -29.0, -0.01, _reset_func=None)
#two_d_gen.generateModelFile()
#two_d_gen.generateTMatFile()
