import numpy as np
import ROOT
import miind.palette as palette
import miind.ode2dsystem as ode2dsystem 
import miind.mesh3 as mesh
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import xml.etree.ElementTree as ET
from itertools import count

ROOT.gROOT.SetBatch(False)

AXISCOLOR=1
DEFAULT_COLOR_LEGEND=[1e-6,1,100]
MAINFRAC = 0.9

class Visualizer:

    _ids = count(0)
    '''Wrapper class for ROOT visualization. For generation of visualization plots, use PlotGenerator.'''
    def __init__(self, sys, bbox=[], canvas = None, textpad = True):

        self.colorbar = []
        self.colorbarplanes = []
        self.id = self._ids.next()
        '''If a reversal bin is required, it must already have been inserted in the Mesh before using this class.'''

        self.sys  = sys
        if canvas == None:
            self.c = ROOT.TCanvas("c","",0,0,800,800)
            self.p = ROOT.TPad("p","",0.,1-MAINFRAC,MAINFRAC,1.)
            self.p.Draw()
            self.l = ROOT.TPad("l","",MAINFRAC,0.,1.,1.)
            self.l.Draw()
        else:
            self.c = canvas
            self.p = ROOT.TPad("p","",0.,1-MAINFRAC,MAINFRAC,1.)
            self.p.Draw()
            self.l = ROOT.TPad("l","",MAINFRAC,0.,1.,1.)
            self.l.Draw()

        self.c.SetFillColor(0)
        self.c.Draw()

        self.dimensions = self.sys.m.dimensions()

        if bbox == []:
            self.h = ROOT.TH2F("h_" + str(self.id) ,"",500, self.dimensions[0][0], self.dimensions[0][1],500, self.dimensions[1][0], self.dimensions[1][1])
        else:
            self.h = ROOT.TH2F("h_" + str(self.id) ,"",500,bbox[0][0],bbox[0][1],500,bbox[1][0],bbox[1][1])

        self.h.GetXaxis().SetAxisColor(AXISCOLOR)
        self.h.GetYaxis().SetAxisColor(AXISCOLOR)
        self.h.GetXaxis().SetLabelColor(AXISCOLOR)
        self.h.GetYaxis().SetLabelColor(AXISCOLOR)
        self.h.GetXaxis().SetLabelSize(0.05)
        self.h.GetXaxis().SetLabelOffset(0.02)
        self.h.GetYaxis().SetLabelSize(0.05)
        self.h.GetYaxis().SetLabelOffset(0.02)
        self.h.GetXaxis().SetNdivisions(5)
        
        self.p.cd()
        self.h.Draw()

        if textpad == True:
            self.txtpad = ROOT.TPad("tp","",0.4,0.9,0.6,0.95)
            self.txtpad.SetFillColor(1)
            self.txtpad.Draw()
            self.have_text = True
        else:
            self.have_text = False

        ROOT.gStyle.SetPalette(53);

        self.__create_color_objects()
        self.__initialize_geom_objects()

        self.txt = ROOT.TText()
        self.txt.SetTextColor(68)
        self.auxhists = []
        return


    def __create_color_objects(self):
        self.ncolors =   ROOT.TColor.GetNumberOfColors()
        self.colors  = [ ROOT.TColor.GetColorPalette(i) for i in range(self.ncolors) ]

    def __initialize_geom_objects(self):
        self.p.cd()
        print(len(self.sys.m.cells[0]))
        self.geom_list = []
        for i, cells in enumerate(self.sys.m.cells):
            for j, cell in enumerate(cells):
                if i == -1:
                    if cell.area > 0:
                      print ('yes')
                      m=ROOT.TMarker(cell.centroid[0],cell.centroid[1],20)
                      m.SetMarkerStyle(20)
                      m.SetMarkerColor(2)
                      m.SetMarkerSize(0.5)
                      m.Draw()
                      self.geom_list.append(m)
                    else:
                      print( 'no')
                      m=ROOT.TMarker(0.,0.,20)
                      m.SetMarkerColor(2)
                      m.SetMarkerSize(0.01)
                      m.Draw()
                      self.geom_list.append(m)

                else:
                    x=np.array(cell.vs, dtype=float)
                    y=np.array(cell.ws, dtype=float)
                    line=ROOT.TPolyLine(len(x),x,y)
                    line.SetFillColor(0)
                    line.Draw('Fill')
                    self.geom_list.append(line)

    def __draw_colorbar__(self,colorlegend, mi, ma):
        ''' Draws a color legend to the side of the plot. '''

        self.l.cd()
        xmin = 0.
        xmax = 0.75
        ymin = 1-MAINFRAC
        ymax = MAINFRAC
        ndiv = colorlegend[2]
        ysize = 1./float(ndiv)
        ycur = ymin
        x = np.zeros(4)
        y = np.zeros(4)

        n_text_pos = 5
        textpos = np.linspace(ymin, ymax, n_text_pos)
        xtext = 0.8
        labels = []
        for ytext in textpos:
            labels.append(ROOT.TText(xtext,ytext,'kadaver'))
            labels[-1].Draw()

        self.labels = labels

        # generate a pseudodensity based on color legend

        if len(self.colorbarplanes) == 0:
            for i in range(ndiv):
                x[0] = xmin
                y[0] = ycur
                x[1] = xmax
                y[1] = ycur
                x[2] = xmax
                y[2] = ycur+ysize
                x[3] = xmin
                y[3] = ycur+ysize
                ycur += ysize
                l=ROOT.TPolyLine(len(x),x,y)

                self.colorbarplanes.append(l)

        for i in range(ndiv):
            value = float(i)/float(ndiv)
            self.colorbarplanes[i].SetFillColor(self.color(value))
            self.colorbarplanes[i].Draw('Fill')
        self.c.cd()

    def color(self,x):
        ''' x between 0 and 1.'''
        if x > 1:
            return self.colors[-1]
        else:
            return self.colors[int(x*(self.ncolors-1))]

    def density(self):
        density=[]
        for i, cells in enumerate(self.sys.m.cells):
            for j, cell in enumerate(cells):
                if cell.area == 0:
                    cell.area = 1e-10
                    print('{} {} cell area 0'.format(i,j))
                if cell.area < 0:
                    print('{} {} negative cell area'.format(i,j))
                if self.sys.mass[self.sys.map(i,j)] < 0:
                    print('{} {} negative mass'.format(i,j))
                density.append(self.sys.mass[self.sys.map(i,j)]/cell.area)
        print (len(density))
        return np.array(density)

    def SetMarkerStyle(self,style):
        self.h.SetMarkerStyle(style)

    def SetMarkerSize(self,size):
        self.h.SetMarkerSize(size)

    def SetXTitle(self, title):
        self.h.SetXTitle(title)

    def SetYTitle(self, title):
        self.h.SetYTitle(title)

    def colscale(self,dens,colorlegend):
        fl = np.array(len(dens)*[colorlegend[0]])
        cols = np.log10(dens+fl)
        ma = np.max(cols)
        mi = np.min(cols)
        vals = (cols - mi)/(ma - mi)
        return mi, ma, vals

    def demo(self,dens,xlabel,ylabel,pdfname, points, pointcolor,runningtext,colorlegend):
        self.p.cd()
        self.SetXTitle(xlabel)
        self.SetYTitle(ylabel)

        mi, ma, vals = self.colscale(dens,colorlegend)

#        self.__draw_colorbar__(colorlegend, mi, ma)


        # at this stage, there should be a 1-1 correspondence between norm elements and self_geom elements
        print (str(len(dens)) + ',' + str(len(self.geom_list)))
        if len(dens) != len(self.geom_list):
            raise ValueError

        for i, cells in enumerate(self.sys.m.cells):
            for j, cell in enumerate(cells):

                col = self.color(vals[self.sys.map(i,j)])
                if i == -1:
                    self.geom_list[self.sys.map(i,j)].SetMarkerColor(col)
                else:
                    self.geom_list[self.sys.map(i,j)].SetFillColor(col)

        # if no copy, the markers are buried under the geom objects,
        # even if they are drawn later

        self.auxhists.append(self.h.Clone());

        self.auxhists[-1].SetMarkerColor(pointcolor)
        self.auxhists[-1].SetMarkerStyle(2)
        self.auxhists[-1].SetMarkerSize(0.1)
        for point in points:
            self.auxhists[-1].Fill(point[0],point[1])
        self.auxhists[-1].Draw('same')


        if self.have_text == True:
            self.txtpad.SetFillColor(1)
            self.txtpad.Draw()
            self.txtpad.cd()
            self.txt = ROOT.TText(0,0,runningtext) # need to create a new object or text will overwrite
            self.txt.SetTextColor(68)
            self.txt.SetTextSize(0.5)
            self.txt.Draw()


        self.c.Modified()
        self.c.Update()
        if pdfname != '':
            self.c.SaveAs(pdfname +'.pdf','pdf')

    def show(self,xlabel='',ylabel='',pdfname='',points=[],pointcolor=3, runningtext='', colorlegend=DEFAULT_COLOR_LEGEND):
        dens=self.density()
        self.demo(dens,xlabel,ylabel,pdfname,points,pointcolor,runningtext,colorlegend)

    def showfile(self,filename, xlabel = '', ylabel = '', pdfname = '', points = [], pointcolor = 3, runningtext = '', colorlegend=DEFAULT_COLOR_LEGEND):
        f=open(filename)
        line=f.readline()
        data = [ float(x) for x in line.split()[2::3] ]
        for x in data:
            if x < 0:
                print ('Negative density')
                raise ValueError

        self.density = np.array(data)
        self.demo(self.density,xlabel,ylabel,pdfname,points,pointcolor,runningtext,colorlegend)

class PlotGenerator:
    '''Able to visualize a dump from an Ode2DSystem (either generated by the Python or the C++ version). Requires a config file that is no longer in the C++ workflow.
    Consider ModelVisualizer for C++ output.'''

    def __init__(self, meshfilename, meshconfigfilename):
        ''' The mesh file is needed for the Mesh, obviously. The config file is necessary to find the
        stationary points that were added to the simulation, as bins will be present in the system dump.'''
        self.mesh   = mesh.Mesh(meshfilename)
        self.config = open(meshconfigfilename)
        self.__process_config_file__()
        self.sys = Ode2DSystem(self.mesh,[],[])
        self.v   = Visualizer(self.sys)

    def __process_config_file__(self):
        lines = self.config.readlines()
        stop = 0
        for i, line in enumerate(lines):
            if 'Reversal' in line:
                start = i
            if 'Fiducial' in line and stop == 0:
                stop = i

        for j in range(start+1,stop,2):
            vs = [ float(item) for item in lines[j].split() ]
            ws = [ float(item) for item in lines[j+1].split() ]
            q=mesh.Quadrilateral(vs,ws)
            if len(self.mesh.cells[0]) == 1:
                self.mesh.cells[0][0]=q
            else:
                self.mesh.cells[0].append(q)

    def showfile(self,filename, xlabel = '', ylabel='', pdfname='', points = [], pointcolor = 3, runningtext = ''):
        self.v.showfile(filename, xlabel, ylabel, pdfname, points, pointcolor, runningtext)

class ModelVisualizer:
    '''Able to read a Model file and visualize a Ode2DSystem dump. A Model file contains a Mesh with the stationary points already in place. This should
    be used on C++ output. 1D model strips, such as LIF and QIF should be handled by Model1DVisualizer.'''
    def __init__(self,modelfilename, canvas=None, bbox=[], textpad = False):
        self.modelfilename = modelfilename
        tree = ET.parse(modelfilename)
        root = tree.getroot()
        for child in root:
            if child.tag == 'Mesh':
                meshstr=ET.tostring(child)

        base = modelfilename.split('.')[0]
        with open(base + '.mesh.bak','w') as fmesh:
            fmesh.write(meshstr)

        self.mesh = mesh.Mesh(None)
        self.mesh.FromXML(base +'.mesh.bak')
        self.sys = Ode2DSystem(self.mesh,[],[])
        self.v   = Visualizer(self.sys,canvas=canvas,bbox=bbox,textpad=textpad)

    def showfile(self,filename, xlabel = '', ylabel='', pdfname='', points = [], pointcolor = 3, runningtext = '', colorlegend=DEFAULT_COLOR_LEGEND):
        self.v.showfile(filename, xlabel, ylabel, pdfname, points, pointcolor,runningtext,colorlegend)


class Model1DVisualizer:
    ''' If the Mesh is guaranteed to be of a 1D neuron, use this class instead of ModelVisualizer.'''
    def __init__(self,modelfilename,canvas=None, bbox=[]):
        self.modelfilename = modelfilename
        self.bbox = bbox
        tree = ET.parse(modelfilename)
        root= tree.getroot()
        for child in root:
            if child.tag == 'Mesh':
                meshstr=ET.tostring(child)
        base = modelfilename.split(' ')[0]
        with open(base + '.mesh.bak','w') as fmesh:
            fmesh.write(meshstr)

        self.mesh = mesh.Mesh(None)
        self.mesh.FromXML(base +'.mesh.bak')
        self.sys = Ode2DSystem(self.mesh,[],[])

        self.__init_interpretation_array__()
        self.canvas = canvas

    def __init_interpretation_array__(self):
        inter  = []
        areas  = []
        for i, cells in enumerate(self.sys.m.cells):
            for j, cell in enumerate(cells):
                center_point = (cell.points[0][0] + cell.points[1][0])/2.
                inter.append(center_point)
                area = cell.area
                areas.append(area)
        self.areas = np.array(areas)
        self.interpretation = np.array(inter)
        # determine the width of the strip as this needs to be divided out of the density
        cell = self.sys.m.cells[1][0] # should be there in a 1D model

        d = np.fabs(cell.points[1][1] - cell.points[0][1])
        if d == 0:
            d = np.fabs(cell.points[0][1] - cell.points[2][1])

        if d != 0:
            self.w = d
        else:
            raise ValueError
        print ('w: ' + str(self.w))


    def graphfromfile(self, filename):
        f=open(filename)
        line=f.readline()
        data = [ float(x) for x in line.split()[2::3] ]
        for x in data:
            if x < 0:
                print ('Negative density')
                raise ValueError
        if len(data) != len(self.areas):
            raise ValueError
        self.density = np.zeros(len(data))
        sum_mass = 0.

        for i in range(len(data)):
            sum_mass += data[i]*self.areas[i]
            self.density[i] = data[i]*self.w

        return self.interpretation, self.density

    def showfile(self,filename, xlabel='', ylabel='', pdfname='', colorlegend = DEFAULT_COLOR_LEGEND):

        interpretation, density = self.graphfromfile(filename)
        g=ROOT.TGraph(len(interpretation), interpretation, density)

        if self.bbox != []:
            self.h = ROOT.TH2F("h_" ,"",500,self.bbox[0][0],self.bbox[0][1],500,self.bbox[1][0],self.bbox[1][1])
            self.h.Draw()
            g.Draw('L')
        else:
            g.Draw('AL')

        self.canvas.Update()

class MatrixVisualizer:
    ''' Show if probability is lost in transitions and where.'''
    def __init__(self,modelfile, matrixfile):

        self.modelviz = ModelVisualizer(modelfile)
        self.__parse_matrix_file__(matrixfile)

    def __parse_matrix_file__(self,matrix_file):
        with open(matrix_file) as f:
            lines = f.readlines()
            for line in lines[1:]:
                orig = line.split(';')[1]
                corig = [ int(c) for c in orig.split(',') ]

                tolist   = line.split(';')[2:]
                s = 0
                for to in tolist:
                    if len(to.split(':')) > 1:
                        cont = float(to.split(':')[1])
                        s += cont
                if np.fabs(s - 1.0) > 1e-6:
                    print ('Leak cell:' + str(corig[0]) + ',' + str(corig[1]) + ',' + str(self.model.viz.v.sys))
                    self.modelviz.v.sys.mass[self.modelviz.v.sys.map(corig[0],corig[1])] = 0.1
        self.modelviz.v.sys.mass[0] = 1.0
        self.modelviz.v.show()

class ResetVisualizer:
    '''Show the reset mapping.'''

    def __init__(self,modelfile):
        tree = ET.parse(modelfile)
        root = tree.getroot()
        mrep = root.findall('Mesh')
        mstr = ET.tostring(mrep[0]) # there only should be one mesh
        self.m = Mesh(None)
        self.m.FromXML(mstr,fromString=True)

        mres = root.findall('Mapping')
        for mp in mres:
            if mp.attrib['type'] == 'Reset':
                mapstr= mp.text
                self.resetmap = self.__construct_map__(mapstr)

        self.connections = self.__construct_connections__()

    def __construct_map__(self, mapstr):
        '''Convert the mapping string into an actual mapping.'''
        l = mapstr.split('\n')
        self.resmap = [ [[int(y) for y in x.split()[0].split(',')], [int(y) for y in x.split()[1].split(',')], float(x.split()[2])]  for x in l if len(x.split()) == 3]


    def __construct_connections__(self):
        self.connects = []
        for line in self.resmap:
            fr=line[0]
            to=line[1]

            self.connects.append( [self.m.cells[fr[0]][fr[1]].centroid, self.m.cells[to[0]][to[1]].centroid] )


    def plot_connects(self):
        for el in self.connects:
            plt.plot([el[0][0],el[1][0]],[el[0][1],el[1][1]])

            delta_x = (el[1][0] - el[0][0])
            delta_y = (el[1][1] - el[0][1])
            grad = delta_y/delta_x
            if grad > 0:
                print ('BONG' + ',' + str(grad))

        plt.show()
