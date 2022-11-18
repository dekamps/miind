#!/usr/bin/env python

import sys
#import ROOT
import numpy as np
import uuid
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from miind.bary3 import isinsidequadrilateral
from miind.bary3 import isinsidetriangle
from miind.bary3 import areaoftriangle
from miind.bary3 import areaofquadrilateral
from miind.bary3 import splitquadrilateral
from miind.bary3 import distance_to_line_segment
from miind.bary3 import generate_random_triangle_points
from miind.bary3 import generate_random_quadrilateral_points
from scipy.spatial import KDTree
from scipy.spatial import distance
from matplotlib.path import Path


# ROOT.gROOT.SetBatch(True)
# ROOT.gStyle.SetOptStat(0)

#enlarge if there are missing points
MAX_NEIGHBOURS = 128


def mergeQuads(a,b):
    newCell = Quadrilateral([a.points[0][0],b.points[1][0],b.points[2][0],a.points[3][0]],[a.points[0][1],b.points[1][1],b.points[2][1],a.points[3][1]])
    return newCell


from itertools import tee, islice, chain

def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return izip(prevs, items, nexts)

def np_perp( a ) :
    '''a is list like object of floats of length 2, corresponding to a 2D vector.
    Returns the same object, representing the perpendicular vector.'''
    b = np.empty_like(a)
    b[0] = a[1]
    b[1] = -a[0]
    return b

def np_cross_product(a, b):
    '''a and b are list like objects of length 2. Returns the cross product of a and b, when interpreted as 2D vectors.'''
    return np.dot(a, np_perp(b))


def np_seg_intersect(a, b):
    ''' a and b are line segments, defined by begin point a[0] (b[0]) and end point a[1] (b[1]). This function returns the intersection point
    of the two line segments or None, if there is none.'''
    r = np.subtract(a[1], a[0])
    s = np.subtract(b[1], b[0])
    v = np.subtract(b[0], a[0])
    num = np_cross_product(v, r)
    denom = np_cross_product(r, s)

    # if r x s = 0 and (q-p)xr=0, then the two lines are collinear.
    if np.isclose(denom, 0) and not np.isclose(num, 0):
	# Parallel and non-intersecting
        return None
    u = num / denom
    t = np_cross_product(v, s) / denom

    if u >= 0 and u <= 1 and t >=0 and t <=1:
        res = b[0] + (s*u)
        return res
	# Otherwise, the two line segments are not parallel but do not intersect.
    return None

def create_from_xml(filename):
    m = Mesh(None)
    m.FromXML(filename)
    return m

class CellProperty:

    def __init__(self, i, j, translation = [0.,0.], color = 2, fill = False):
        self.i = i
        self.j = j
        self.translation = translation
        self.color = color
        self.fill = fill


class Cell:
    '''Cell is a part of a mesh, most often a quadrilateral, but there are exceptions.'''

    def __init__(self, vs, ws):
        ''' assumption is that of 2D neural dynamics. One variable will be the membrane
        potential  v, and the other one will be called w here.'''
        self.vs = vs
        self.ws = ws

        self.area   = 0.
        self.points = []
        self.centroid = [0., 0.]

    def isPointInside(self,point):
        '''Every cell can tell whether a point is inside it or not.'''
        return False

    def draw(self, plotlist, translation = [0., 0.]):
        '''Add ROOT primitives to draw this cell. No-op for the base class.'''

    def generate_points_in_cell(self,N):
        return

class Quadrilateral(Cell):
    ''' This Cell takes exactly 4 mesh points, to be presented as an iterable of 4 vs and an iterable of 4 ws.
    It is the user's responsibility to ensure that the quadrilateral is not degenerate and self-crossing.'''
    def __init__(self, vs, ws):
        if len(vs) != 4:
            raise
        if len(ws) != 4:
            raise
        Cell.__init__(self,vs,ws)

        self.__points__(vs, ws)
        self.__set_reversal__(vs, ws)
        self.__area__()
        self.__centroid__()
        self.__bbox__()

        triangle1, triangle2 = splitquadrilateral(self.points)
        self.triangle1 = triangle1
        self.triangle2 = triangle2

    def __set_reversal__(self, vs, ws):
        if vs == 4*[0] and ws == 4*[0]:
            self.reversal = True
        else:
            self.reversal = False


    def __area__(self):
        self.area = areaofquadrilateral(self.points)


    def __centroid__(self):
        '''Use the vertex centroid, as it is always defined and not too wrong in a fine enough mesh.'''
        x = (self.points[0][0] + self.points[1][0] + self.points[2][0] + self.points[3][0])/4
        y = (self.points[0][1] + self.points[1][1] + self.points[2][1] + self.points[3][1])/4
        self.centroid = np.array([x,y])

    def __points__(self,vs, ws):
        ''' arange points, as one array of vs, and one array of ws.'''
        points = []
        for i in range(4):
            point = [vs[i],ws[i]]
            points.append(point)
        self.points = np.array(points)

    def __bbox__(self):
        '''determine bounding box.'''
        vs = self.points.transpose()[0]
        ws = self.points.transpose()[1]
        self.box = [[min(vs),min(ws)],[max(vs),max(ws)]]

    def bbox(self):
        '''Return cell bounding box.'''
        return self.box

    def isPointInside(self, point):
        ''' Returns True if the point is inside the cell, False otherwise.'''
        return isinsidetriangle(self.triangle1,point) or isinsidetriangle(self.triangle2,point)

    def draw(self, plotlist, translation = [0., 0.], c = 1):
        '''Quadrilateral adds TLine elements representing its boundary to the plot list.'''
        lx = [ p[0] + translation[0] for p in self.points ]
        ly = [ p[1] + translation[1] for p in self.points ]
        # close area
        lx.append(lx[0])
        ly.append(ly[0])

        x=np.array(lx)
        y=np.array(ly)
        line=ROOT.TPolyLine(len(x),x,y)
        line.SetFillColor(0)
        line.SetLineColor(c)
        line.Draw('Fill')

        plotlist.append(line)


    def generate_points_in_cell(self, N):
        '''Generate N points inside the quadrilateral.'''
        return generate_random_quadrilateral_points(self.points, N)

    def isSimple(self):
        diag_1=[self.points[0].tolist(),self.points[2].tolist()]
        diag_2=[self.points[1].tolist(),self.points[3].tolist()]
        if np_seg_intersect(diag_1,diag_2) is None: # adapted to prevent elementwise comparison (MdK:7/4/2017)
            return False
        else:
            return True

    def isSelfIntersecting(self):
        ''' Returns True if the 4 points defining the quadilateral define a self-intersecting shape, False otherwise.'''
        edge_1 = [self.points[0].tolist(), self.points[1].tolist()]
        edge_2 = [self.points[1].tolist(), self.points[2].tolist()]
        edge_3 = [self.points[2].tolist(), self.points[3].tolist()]
        edge_4 = [self.points[3].tolist(), self.points[0].tolist()]

        if np_seg_intersect(edge_1, edge_3) is None and np_seg_intersect(edge_2, edge_4) is None: # adapted to prevent elementwise comparison (MdK:7/4/2017)
            return False
        else:
            return True

    def isTooSmall(self, threshold = 1e-8):
        ''' Returns True if the Quadrilateral area is below a certain limit, False otherwise.'''
        if self.area < threshold:
            return True
        else:
            return False

class Negative(Cell):
    ''' Create a cell that is defined by the bins it is not in. '''

    def __init__(self, cells, quadrilateral, N = 100, fiducial_distance = 0, list_of_segments = []):
        ''' Give the list of other cells and a rectangular perimeter that is guaranteed to cover this
        cell, and is covered ENTIRELY by the other cells of the Mesh. Quadrilateral is a an np array of four points. A point is a list of two coordinates: v and w.
        Spurious points may very rarely arise close to the perimeter, due to the way the fiducial volume within the paparmeter is cut off from the rest of the mesh.
        It is not worth to code around this, at the moment. It is recommended to do a visual inspection to see where and how this happens. If a fiducial
        distance > 0 is defined, all points within that distance from the perimeter will be labeled as not in the bin. It is possible to exclude sides from
        this cut, which is necessary when the perimeter directly bounds the reset bin. list_of_segments is a list of indices that indicates which side of the
        quadrilaterals should be left out of this cut. The side defined by qudrilateral[0] and quadrilateral[1] is indexed by 0, etc.'''

        self.cells          = cells
        self.perimeter      = quadrilateral

        triangle1, triangle2  = splitquadrilateral(quadrilateral.points)
        self.triangle1        = triangle1
        self.triangle2        = triangle2

        self.N              = N

        self.fiducial_distance  = fiducial_distance
        self.list_of_segments   = list_of_segments
        self.__prepare_fiducial_cut__()

        self.points     = []
        self.cell_list  = []

        self.__cells_within_perimeter__()
        self.__generate_filtered__()

        self.__determine_bbox__()
        self.__area__()


    def __prepare_fiducial_cut__(self):
        '''Prepare the list of segments that the fiducial cut should applied on, i.e those sides of the perimeter that
        are not excluded by list of segments.'''
        self.test_segments = []
        for i in range(4):
            if i not in self.list_of_segments and self.fiducial_distance > 0:
                self.test_segments.append([self.perimeter[i], self.perimeter[(i+1)%4]])

    def __area__(self):
        self.area = areaofquadrilateral(self.perimeter.points)*float(self.N)/float(self.total)

    def __is_close_to_perimeter(self, point):
        for segment in self.test_segments:
            if self.fiducial_distance > 0 and distance_to_line_segment(segment,point) < self.fiducial_distance:
                return True
            else:
                return False

    def __is_point_inside__(self, point):
        ''' Test whether a point is inside the perimeter, but in one of the cells.'''
        if isinsidetriangle(self.triangle1, point) == False and isinsidetriangle(self.triangle2, point) == False:
            return False

        if self.__is_close_to_perimeter(point) == True:
            return False

        for coord in self.cell_list:
            if self.cells[coord[0]][coord[1]].isPointInside(point) == True:
                return False

        return True

    def __generate_filtered__(self):
        ''' The quadrilateral that was given is supposed to contain this bin. We use Monte Carlo integration to find the area and
        for visualization of the bin.'''
        filtered = []
        total = 0
        p = 0
        while p < self.N:
            total += 1
            point = generate_random_quadrilateral_points(self.perimeter.points,1)[0]
            if self.__is_point_inside__(point):
                filtered.append(point)
                p += 1

        self.total    = total
        self.filtered = filtered

        return

    def __determine_bbox__(self):
        ''' There are self.N points generated within the perimeter. There is a larger number of points that must be generated,
        which can be used to calculate the bin area. This function returns the total number of points that had to be generated
        to reach self.N points in the bin area.'''

        vs = [ point[0] for point in self.filtered ]
        ws = [ point[1] for point in self.filtered ]

        self.centroid =  np.array([float(np.sum(vs)), float(np.sum(ws))])/self.N

        minv  = np.min(vs)
        maxv  = np.max(vs)

        minw = np.min(ws)
        maxw = np.max(ws)

        self.box = [[minv, minw],[maxv, maxw]]


    def __add_cell_list(self,i,j):
        el = [i,j]
        if not el in self.cell_list:
            self.cell_list.append(el)

    def __cells_within_perimeter__(self):
        ''' Add cells with at least one point within the perimeter. These cells with be checked to wether a point is inside this cell or not.'''
        for i, cells in enumerate(self.cells):
            for j, cell in enumerate(cells):
                for point in cell.points:
                    # if someone somehow tries to insert a Negative in a mesh that already has a Negative (and not a Cell),
                    # disaster would strike, so we test
                    if (i != 0) and isinsidequadrilateral(self.perimeter.points,point):
                        self.__add_cell_list(i,j)

                # Now also add the cells that the perimeter points themseves are in. They can be missed if this cell has corner points
                # that are not within the perimeter
                if [i,j] not in self.cell_list:
                        for p in self.perimeter.points:
                            if cell.isPointInside(p):
                                self.__add_cell_list(i,j)

    def isPointInside(self, point):
        return self.__is_point_inside__(point)

    def bbox(self):
        ''' Return bounding box.'''
        return self.box

    def generate_points_in_cell(self,N):
        '''Generate N points, uniformly distributed inside the cell.'''

        # It is assumed that enough points already have been generated.
        if len(self.filtered) >= N:
            return self.filtered[:N]
        else:
            raise ValueError

    def update(self,perimeter,filtered, total):
        '''Allows the use of a list of points generated within this, and that was stored elsewhere in a file. total is the total number of points
        used to generate this list, which is required for the calculation of the area. This function
        will replace self.filtered by filtered, and recalculate the bin area and centroid based on this point. Not attempt will
        be made to test whether the points are actually within the cell volume. It is the callers responsibility to ensure or this
        or to do this post hoc in visualization. This routine is created to allow the use of data produced by parallel jobs.'''

        self.perimeter = np.array(perimeter)
        self.filtered  = filtered
        self.N         = len(filtered)
        self.total     = total

        self.__area__()
        self.__determine_bbox__()

    def draw(self,plotlist, translation = [0., 0.], color = 3):

        for point in self.filtered:
            m=ROOT.TMarker()
            m.SetMarkerColor(color)
            m.DrawMarker(point[0] + translation[0],point[1] + translation[1])
            plotlist.append(m)

class Mesh:

    def __init__(self, filename, kdTree=False):
        ''' A mesh has already been generated on a file.'''
        if filename is None:
            self.vs = []
            self.ws = []
            self.cells = []
            self.neighbours = {}
        else:
            self.filename = filename
            f = open(filename)
            lines = f.readlines()
            if len(lines) == 0:
                raise

            self.dt = float(lines[1])
            # preamble is two lines
            self.n_char = len(lines) - 2
            self.vs     = []
            self.ws     = []
            self.cells  = []

            # Only one inversion may occur in a meshfile
            self.inversion = False
            blocks = self.__split_blocks__(lines)

            # it is now assumed that self.cells is a list of lists of quadrilaterals.
            # this is not enforced yet
            self.cells = []

        # We create a place holder for strip 0. This cell has area 0; this can be used to test whether
        # real stationary cells have been added or not
            self.cells.append([Cell([0.0],[0.0])])

            for block in blocks:
                vs, ws = self.__build_arrays__(block)
                self.vs.append(vs)
                self.ws.append(ws)
                self.__build_grid__(vs,ws)

            if (kdTree == True): self.__build_tree__()
            self.neighbours = {}
            self.__build_neighbours__()
            self.are_labels_drawn = True

    def __split_blocks__(self,lines):

        blocks   = []
        newblock = []
        for line in lines[2:]:

            if 'end' in line:
                return blocks

            if 'closed' in line or 'inversion' in line:
                if len(newblock)%2 != 0:
                    raise ValueError
                blocks.append(list(newblock))
                newblock = []


                if 'inversion' in line:
                    if self.inversion == True:
                        raise ValueError
                    self.inversion = True
                    # demarcate where the inversion happens, so that later the number
                    # of strips before the inversion can be calculated
                    bound = len(blocks)
            else:
                newblock.append(line)
        blocks.append(newblock)

        # self bound is the upper boundary of the regular mesh. From this strip number onwards,
        # strips are made from 'inversions'.
        self.bound = 1
        for  i, block in enumerate(blocks):
            if self.inversion == True and i == bound: break
            nr_strips = len(block)/2 - 1
            self.bound += nr_strips

        return blocks


    def __build_arrays__(self,block):
        '''Builds arrays of characteristics,
        the  v points and the w points of a characteristic
        each have their own list.'''
        vs = []
        ws = []

        for line in block[0::2]:
            items = line.split()
            data = [ float(el) for el in items ]
            vs.append(data)
        for line in block[1::2]:
            items = line.split()
            data = [ float(el) for el in items ]
            ws.append(data)
        return vs, ws

    def __build_tree__(self):
        ''' build a KDTree out of the characteristic arrays. The grid points are stored
        in self.data. Thhe same point may occur multiple times, as'''
        points = []

        tandem = zip(self.vs, self.ws)
        for block in tandem:
            blocks = zip(block[0],block[1])
            for el in blocks:
                coords = zip(el[0],el[1])
                for coord in coords:
                    point = [coord[0], coord[1]]
                    points.append(point)

        self.data = np.array(points)
        self.tree = KDTree(self.data)

    def __build_grid__(self,vs,ws):
        ''' Builds a list of lists of cells.'''

        for i,v in enumerate(vs):
            if i < len(vs) - 1:
                m = min(len(vs[i]),len(vs[i+1]))
                cell = []
                for j in range(m-1):
                    cellv = [vs[i][j],vs[i][j+1],vs[i+1][j+1],vs[i+1][j]]
                    cellw = [ws[i][j],ws[i][j+1],ws[i+1][j+1],ws[i+1][j]]
                    try:
                        quad =Quadrilateral(cellv,cellw)
                    except ValueError:
                        print('An error, probably a degeneracy in strip: ',i, ' cell: ', j)
                    cell.append(quad)
                self.cells.append(cell)

    def __build_neighbours__(self):
        '''Builds a dictionary of lists of cells, keyed by a tuple representing a point. The list  contains all cells that
        a meshpoint is part of. Only the quadrilaterals should be included in this neighbours list.'''

        for i, cellist in enumerate(self.cells):
            for j, cell in enumerate(cellist):
                # this test ensures that only quadrilaterals are included in the neighbour list,
                # because cell[0][0] is initialized with a Cell instances that has no area.
                # After initialization cell[0][0] can have an area. i,j must run over the entire
                # cell list as they are to become coordinates.
                if cell.area > 0:
                    for p in range(4):
                        point = (cell.points[p][0],cell.points[p][1])
                        if point in self.neighbours:
                            self.neighbours[point].append([i,j])
                        else:
                            self.neighbours[point] = []
                            self.neighbours[point].append([i,j])
        return

    def __add_label__(self, cell, i, j, labelsize):
        point = cell.centroid
        t=ROOT.TText(point[0], point[1],str(i) + ',' + str(j))
        t.SetTextAlign(22)
        t.SetTextSize(labelsize)
        return t

    def insert_negative(self, perimeter, N = 10000, fiducial_distance = 0.0, exclusion_list = []):
        '''Replace the place holder for a stationary point by a Negative instance. If more than one stationary point is present,
        extend the list.'''
        cell = Negative(self.cells,perimeter, N, fiducial_distance, exclusion_list)

        if self.cells[0][0].area > 0:
            self.cells[0].append(cell)
        else:
            self.cells[0][0] = cell

    def insert_stationary(self, quadrilateral):
        '''Quadrilateral needs to be a cell of type Quadrilateral.'''

        if self.cells[0][0].area > 0:
            self.cells[0].append(quadrilateral)
        else:
            self.cells[0][0] = quadrilateral

    def deltat(self):
        '''Time step used in building the grid.'''
        return self.dt

    def draw(self,plotlist,labelsize):
        ''' Draw all cells as list of TLine elements.'''
        #  ignore reversal bin
        for i,celllist in enumerate(self.cells):
            for j, cell in enumerate(celllist):
                cell.draw(plotlist)
                #i+1 because the reversal bin is not included, but enumerate makes i start at 0
                #this only affects the text label
                if self.are_labels_drawn == True:
                    t=self.__add_label__(cell,i,j,labelsize)
                    plotlist.append(t)

        #make sure stationary bins comes on top
        for cell in self.cells[0]:
            cell.draw(plotlist)

    def dimensions(self):
        '''Returns minv, maxv, minw, maxw'''
        vs =  [point[0]  for celllist in self.cells for cell in celllist for point in cell.points]
        ws =  [point[1]  for celllist in self.cells for cell in celllist for point in cell.points]
        return np.array([[min(vs), max(vs)], [min(ws), max(ws)]])

    def bbox(self,i,j):
        ''' Give the bounding box of a given cell '''
        return self.cells[i][j].bbox()

    def query(self,point):
        ''' Give the result of a KDTree query'''
        ret=self.tree.query(point)
        return ret

    def bin_from_point(self, point):
        '''Give the bin the point belongs to. Return None if there is none.'''

        # first check the stationary bins
        for i, cell in enumerate(self.cells[0]):
            if self.isinbin(0,i,point):
                return [0,i]

        # else walk the tree

        close = self.tree.query(point,MAX_NEIGHBOURS)
        nearests=self.data[close[1]]
        for nearest in nearests:
            ns = self.neighbours[(nearest[0],nearest[1])]
            for n in ns:
                if self.isinbin(n[0],n[1],point):
                    return n

        return None

    def isinbin(self, i,j,point):
        ''' True if point(v,w) is in bin i,j. False otherwise.'''
        return self.cells[i][j].isPointInside(point)


    def findvs(self,v):
        ''' Give a list of all bins that contain potential V.'''
        l = []
        for i, cells in enumerate(self.cells):
            for j, cell in enumerate(cells):
                box = cell.bbox()

                if v >= box[0][0] and v < box[1][0]:
                    l.append([i,j])
        return l


    def checkSimple(self):
        '''Checks whether Quadrilaterals are Simple'''
        chksum=0
        for i, cells in enumerate(self.cells):
            for j, cell in enumerate(cells):
                chkCell=self.cells[i][j]
                if chkCell.__class__.__name__ == 'Quadrilateral':
                    if chkCell.isSimple() == False:
                        chksum += 1
                        print(i,j)
        return chksum

    def checkSelfIntersection(self):
        '''Checks for concave and self-intersecting Quadrilaterals'''
        chkConcave=0
        chkSelfIntersect=0
        x_coords=[]
        y_coords=[]
        for i, cells in enumerate(self.cells):
            for j, cell in enumerate(cells):
                chkCell=self.cells[i][j]
                if chkCell.__class__.__name__ == 'Quadrilateral':
                    if chkCell.isSimple() == False:
                        if chkCell.isSelfIntersecting() == False:
                            chkConcave += 1
                        else:
                            chkSelfIntersect += 1
                            print(i,j)
                            x_coords=np.append(x_coords,chkCell.centroid[0])
                            y_coords=np.append(y_coords,chkCell.centroid[1])
        plt.scatter(x_coords,y_coords)
        if chkSelfIntersect > 0:
            plt.show()
        return chkConcave,chkSelfIntersect

####under construction
    def removeBadBins(self):

        new_mesh = Mesh(None)
        new_mesh.dt = self.dt # Added (MdK): 6/04/2017

        for i, cells in enumerate(self.cells):
            if i == 0:
                new_mesh.cells.append(self.cells[0])
            else:
                for j, cell in enumerate(cells):
                    chkCell = self.cells[i][j]
                    if j == 0 and chkCell.isSelfIntersecting():
                        new_mesh.cells.append([Cell([0.],[0.])])
                    elif j == 0 and not chkCell.isSelfIntersecting():
                        new_mesh.cells.append([chkCell])
                    elif not chkCell.isSelfIntersecting():
                        new_mesh.cells[i].append(chkCell)
                    elif chkCell.isSelfIntersecting():
                        break

        return new_mesh

    def removeSmallBins(self,thresh = 1e-5):

        new_mesh = Mesh(None)
        new_mesh.dt = self.dt
        new_mesh.filename = self.filename
        for i, cells in enumerate(self.cells):
            if i == 0:
                new_mesh.cells.append(self.cells[0])
            else:
                for j, cell in enumerate(cells):
                    chkCell = self.cells[i][j]
                    if j == 0 and chkCell.isTooSmall(threshold = thresh):
                        new_mesh.cells.append([Cell([0.],[0.])])
                    elif j == 0 and not chkCell.isTooSmall(threshold = thresh):
                        new_mesh.cells.append([chkCell])
                    elif not chkCell.isTooSmall(threshold = thresh):
                        new_mesh.cells[i].append(chkCell)
                    elif chkCell.isTooSmall(threshold = thresh):
                        break
        return new_mesh

    def mergeSmallBins(self, thresh = 2e-3):
        new_mesh = Mesh(None)
        new_mesh.dt = self.dt
        new_mesh.filename = self.filename
        for i,cells in enumerate(self.cells):
            if i==0:
                new_mesh.cells.append(self.cells[0])
            else:
                for j, cell in enumerate(cells):
                    chkCell = self.cells[i][j]
                    if j==0:
                        new_mesh.cells.append([chkCell])
                    elif not chkCell.isTooSmall(threshold = thresh):
                        new_mesh.cells[i].append(chkCell)
                    elif chkCell.isTooSmall(threshold = thresh):
                        new_mesh.cells[i].append(mergeQuads(chkCell,self.cells[i][-1]))
                        break
        return new_mesh

    def removeWithRenewal(self, threshold = 1e-8):
        new_mesh = Mesh(None)
        revname = self.filename.split('.')[0] + '.rev'

        f = open(revname, 'w')
        f.write('<Mapping Type=\"Reversal\">\n')
        flagged_cells = []

        for i, cells in enumerate(self.cells):
            if i == 0 or i == 1:
                new_mesh.cells.append(self.cells[i])
            else:
                for j, cell in enumerate(cells):
                    chkCell = self.cells[i][j]

                    if j == 0 and (chkCell.isSelfIntersecting() or chkCell.isTooSmall(threshold)):
                        new_mesh.cells.append([Cell([0.],[0.])])
                        print (i,j)
                    elif j == 0 and not (chkCell.isSelfIntersecting() or chkCell.isTooSmall(threshold)):
                        new_mesh.cells.append([chkCell])
                    elif not (chkCell.isSelfIntersecting() or chkCell.isTooSmall(threshold)):
                        new_mesh.cells[i].append(chkCell)

                    if j == len(self.cells[i]) - 1:
                        ind = np.argmin(distance.cdist([C.centroid for C in self.cells[1]], [chkCell.centroid],'euclidean'))
                        f.write(repr(i)+ ',0\t1,' + repr(ind)+'\t1.0\n')
                    elif (chkCell.isSelfIntersecting() or chkCell.isTooSmall(threshold)):
                        if j != 0:
                            flagged_cells.append([i,j])
                        else:
                            print(i,j)
                            break
#	f.write('</Mapping>')

#	for i,xy in enumerate(end_strips):
#	    C = self.cells[xy[0]][xy[1]].centroid
#	    D = self.cells[1][limit_coords[i]].centroid
#	    plt.scatter(C[0],C[1])
#	    plt.scatter(D[0],D[1], color = 'r')

        for coords in flagged_cells:
            ind = np.argmin(distance.cdist([x.centroid for y in (new_mesh.cells) for x in y], [self.cells[coords[0]][coords[1]].centroid], 'euclidean'))
            i,j = new_mesh.cellIndex(ind)
            f.write(repr(coords[0])+',0\t'+repr(i)+','+repr(j)+'\t1.0\n')
            f.write('</Mapping>')

#	plt.show()
        return new_mesh

    def cellIndex(self,index):
        test_index = 0
        for i, cells in enumerate(self.cells):
            for j, cell in enumerate(cells):
                if test_index == index:
                    return i,j
                else:
                    test_index += 1

    def ToXML(self,fn):
        with open(fn,'w') as f:
            f.write('<Mesh>\n')
            f.write('<TimeStep>')
            f.write(str(self.dt))
            f.write('</TimeStep>\n')

            for i, cells in enumerate(self.cells):
                f.write('<Strip>')
                for j, cell in enumerate(cells):
                    for p in cell.points:
                        f.write("{:.12f}".format(p[0]))
                        f.write(' ')
                        f.write("{:.12f}".format(p[1]))
                        f.write(' ')
                f.write('</Strip>\n')
            f.write('</Mesh>\n')

    def ToStat(self, fn):
        with open(fn, 'w') as f:
            f.write('<Stationary>\n')
            for i, cells in enumerate(self.cells):
                for j, cell in enumerate(cells):
                    if len(cell.points) == 4:
                        f.write('<Quadrilateral><vline>')
                        for p in cell.points:
                            f.write("{:.12f}".format(p[0]))
                            f.write(' ')
                        f.write('</vline><wline>')
                        for q in cell.points:
                            f.write("{:.12f}".format(q[1]))
                            f.write(' ')
                        f.write('</wline></Quadrilateral>\n')
            f.write('</Stationary>')

    def FromXML(self, fn, fromString = False):
        '''Constructs a mesh from eithe a file, or a string if fromString == True.'''
        if not fromString:
            self.filename = fn
            tree = ET.parse(fn)
            root = tree.getroot()
        else:
            self.filename=""
            root = ET.fromstring(fn)

        for ts in root.iter('TimeStep'):
            self.dt = float(ts.text)

        for str in root.iter('Strip'):

            l = []

            # An empty strip should be allowed for example as a place holder for stationary cells
            if str.text is not None:

                coords = [ float(x) for x in str.text.split() ]
                if len(coords)%8 != 0:
                    raise ValueError
                n_chunck = len(coords)/8

                for i in range(0,int(n_chunck)):
                    vs=[]
                    ws=[]
                    vs.append(coords[8*i])
                    vs.append(coords[8*i+2])
                    vs.append(coords[8*i+4])
                    vs.append(coords[8*i+6])
                    ws.append(coords[8*i+1])
                    ws.append(coords[8*i+3])
                    ws.append(coords[8*i+5])
                    ws.append(coords[8*i+7])
                    quad=Quadrilateral(vs,ws)
                    l.append(quad)

            self.cells.append(l)
        self.__build_neighbours__()

def draw_shifted_bin(cell, plotlist, translation, fill = False, color = 2):
        x = np.array([ p[0] + translation[0] for p in cell.points ])
        y = np.array([ p[1] + translation[1] for p in cell.points ])
        line=ROOT.TPolyLine(len(x),x,y)
        if fill == True:
            line.SetFillColor(color)
        else:
            line.SetFillColor(0)

        line.Draw('Fill')
        line.SetLineColor(2)
        line.Draw('f')
        plotlist.append(line)

def draw_curves(curvelist):
    ''' Expects a list of lists: list element [0] is an array of v points, list element [1] is an array of w-points, list element [2] is the line color.'''
    gs=[]
    for curve in curvelist:
        n=len(curve[0])

        g=ROOT.TGraph(n,curve[0],curve[1])
        g.SetLineColor(curve[2])
        g.SetLineWidth(3)
        g.Draw('L')
        gs.append(g)
    return gs

def display_mesh(m, bbox, label = False, xtitle = 'V (mV)', ytitle = '', perimeter = [],  propertylist = [], labelsize = 0.01, curvelist = [] ):
    ''' Displays mesh. If i,j (>0=, ;existing in the grid, not tested)
    are given, the corresponding cell is drawn over the grid, translated by translation.'''

    m.are_labels_drawn=label
    d=m.dimensions()

    c=ROOT.TCanvas('c1','',0,0,500,500)
    c.Draw()

    h=ROOT.TH2F("h","",1000,bbox[0][0],bbox[0][1],1000,bbox[1][0],bbox[1][1])
    a=ROOT.TGaxis
    a.SetMaxDigits(2)
    h.GetYaxis().SetTitleOffset(1.4)
    h.SetXTitle(xtitle)
    h.SetYTitle(ytitle)
    h.Draw()

    p=ROOT.TMarker(-57.3,0.73,0)

    p.SetMarkerColor(3)
    p.Draw()

    plotlist=[]
    m.draw(plotlist,labelsize)

    if perimeter != []:
        vs = [point[0] for point in perimeter]
        ws = [point[1] for point in perimeter]
        pericell=Quadrilateral(vs,ws)
        pericell.draw(plotlist,[0.,0.],2)

    for property in propertylist:
        if property.i > 0 and property.j >= 0:
            print(property.i, property.j, property.translation)
            cell  = m.cells[property.i][property.j]
            print('shifting')
            draw_shifted_bin(cell,plotlist,property.translation,fill = property.fill, color = property.color)

            if property.i == 0:
                tr=property.translation
                m.cells[0][property.j].draw(plotlist,translation=tr)

    for el in plotlist:
        el.Draw()

    gs=draw_curves(curvelist)

    c.Update()
    items = m.filename.split('.')

    c.Print(items[0] + '.pdf')
    return c

def divide_perimeter(quad, N):
    quads = []

    N_div = N + 1

    points = quad.points
    leftv = np.linspace(points[0][0], points[1][0],N_div)
    leftw = np.linspace(points[0][1], points[1][1],N_div)

    rightv = np.linspace(points[3][0],points[2][0],N_div)
    rightw = np.linspace(points[3][1],points[2][1],N_div)

    for i in range(N):
        vs = [leftv[i], leftv[i+1], rightv[i+1], rightv[i]]
        ws = [leftw[i], leftw[i+1], rightw[i+1], rightw[i]]
        quad = Quadrilateral(vs,ws)
        quads.append(quad)
    return quads


if __name__ == "__main__":
    if len(sys.argv) == 2:
        m=Mesh(sys.argv[1])
        display_mesh(m,m.dimensions())
    else:
        print('Usage in main program: python mesh.py <filename>')
