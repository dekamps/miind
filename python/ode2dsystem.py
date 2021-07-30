import numpy as np
import mesh3
#import ROOT

class ScoopList:
    '''Create a list of scoops. A scoop specifies from which bin at any given time probability must be taken and
    which bin it will be moved towards.'''


    def __init__(self, l):
        ''' Expects a list of cell coordinate pairs. A cell coordinate itself is of a list of two integers.
        A cell coordinate pait is a list of two cell ccordinates.
        The first cell coordinate in a cell coordinate pair indicates where the probability will be taken from, the second
        one indicates the cell where the removed probability will end up.'''
        self.l=l


    def append(self, coordinate_pair):
        ''' Append a cell coordinate pair.'''
        self.l.append(coordinate_pair)


class Ode2DSystem:

    '''Create a system where mass is constant, but its position is being updated. If a reset bin is intended, it must already have been
    inserted at this stage.'''

    def __init__(self, m, reversal_list, threshold_list):

        self.reversal_list  = reversal_list
        self.threshold_list = threshold_list

        self.m = m
        self.__reserve_mass_space__()
        self.__define_representation_bins__()

        self.t       = 0.0
        self.mass[1] = 1.0


    def __reserve_mass_space__(self):
        ''' Define the mass array and fill an array with the cumulation of strip lengths.'''
        n = 0
        self.cumulative = [0]
        self.lengths = []
        for cell in self.m.cells:
            l = len(cell)
            n += l
            self.lengths.append(l)
            self.cumulative.append(n)

        self.mass = np.zeros(n)


    def __define_representation_bins__(self):

        self.x=np.array([cell.centroid[0] for cells in self.m.cells for cell in cells])
        self.y=np.array([cell.centroid[1] for cells in self.m.cells for cell in cells])


    def map(self,i,j):
        '''Convert a (v, w) tuple to an internal mass index. Consult the output pdf generated
        by mesh.py for the ordering of the bins'''
        return int(self.cumulative[i] + (j-self.t)%self.lengths[i])

    def redistribute_probability(self):
        '''Implements scoops: probability that has moved through threshold, for example, needs to reappear somewhere else.'''
        self.f = 0
        for pair in self.reversal_list:

            to   = self.map(pair[1][0],pair[1][1])
            fr   = self.map(pair[0][0],pair[0][1])

            self.mass[to] += self.mass[fr]
            self.mass[fr] = 0

        oldfr = []
        for triplet in self.threshold_list:
            to   = self.map(triplet[1][0],triplet[1][1])
            fr   = self.map(triplet[0][0],triplet[0][1])


            alpha = triplet[2]

            self.f += alpha*self.mass[fr]
            self.mass[to] += alpha*self.mass[fr]

            if fr != oldfr:
                if oldfr != []:
                    self.mass[oldfr] = 0.
                oldfr = fr

        self.mass[fr] = 0.


    def initialize(self,i,j):
        '''Set initial density as a delta peak in bin i,j '''
        self.mass.fill(0.0)
        self.mass[self.map(i,j)] = 1.0


    def evolve(self):
        '''Step one unit time step. Update the current time step and thereby the mapping between phase space bin
        and density index. Adapt the density when crossing the threshold. Keep track of the amount of density
        crossing the the threshold.'''
        self.f = 0
        self.t += 1

    def grid(self):
        ''' Produce a tuple suitable for ROOT.TGraph2D from the density.'''

        z = [ self.mass[self.map(i,j)]/cell.area for i, cells in enumerate(self.m.cells) for j, cell in enumerate(cells)  ]
        return self.x, self.y, np.array(z)


    def dump(self,fn):
        ''' Write grid results into a file with name fn. Overwrites earlier results for the same name.'''
        f=open(fn,'w')
        for i, cells in enumerate(self.m.cells):
            for j, cell in enumerate(cells):
                dens=self.mass[self.map(i,j)]/cell.area
                f.write(str(i) + '\t' + str(j) + '\t' + str(dens) + ' ')

        f.close()

if __name__ == "__main__":
    print ('not designed for single use. import as a module.')
