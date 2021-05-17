import miind.mesh3 as mesh3
import miind.writemesh as writemesh
import numpy as np
import uuid


class QifMeshGenerator:

    def __init__(self, basename, tau = 10e-3, V_min = -10., V_max = 10., dt = 1e-4, I = 1.):
        self.basename = basename
        self.tau = tau
        self.V_max = V_max
        self.V_min = V_min
        self.I = I
        self.dt = dt

        self.epsilon  = 0.001   # padding as fraction of the threshold potential
        self.labda    = 0.005   # fiducial bin size as a fraction of (V_threshold - V_rev)
        self.w        = 0.005   # arbitrary value for strip width


    def VQifPositive(self, V0, t):
        I=self.I
        sqi = np.sqrt(I)
        v = sqi*np.tan(sqi*t/self.tau + np.arctan(V0/sqi))
        return v

    def __generateQifMeshPositive__(self):

        I = self.I

        with open(self.basename + '.mesh','w') as meshfile:

            meshfile.write('ignore\n')
            bin=[]
            meshfile.write(str(self.dt) +'\n')

            # just one strip


            V_posneg = [self.V_max*(1 + self.epsilon), self.V_max]
            while V_posneg[-1] > self.V_min:
                v_new = self.VQifPositive(V_posneg[-1],-self.dt)
                V_posneg.append(v_new)

            V_pos = V_posneg[::-1]

            for v in V_pos:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_pos:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            for v in V_pos:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_pos:
                meshfile.write(str(0.0) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')
            meshfile.write('end')


    def VQifNegative(self, V0, t):
        I = -self.I
        sqi = np.sqrt(I)
        ep = np.exp(-2*sqi*t/self.tau)
        pl = sqi + V0
        mi = sqi - V0

        v = sqi*(pl*ep - mi)/(pl*ep + mi)
        return v


    def __generateQifMeshNegative__(self):
        ''' Generates a mesh for QIF neurons. It adds a little padding on the positive strip, so that the
        threshold can be taken close, but above the neuron threshold value.'''

        I = -self.I
        with open(self.basename + '.mesh','w') as meshfile:

            meshfile.write('ignore\n')
            bin=[]
            meshfile.write(str(self.dt) +'\n')

            #   first handle negative strip
            v_stable = -np.sqrt(I) - self.labda
            V_negrev = [v_stable]
            while (V_negrev[-1] > self.V_min):
                v_new = self.VQifNegative(V_negrev[-1],-self.dt)
                V_negrev.append(v_new)
            V_neg = V_negrev[::-1]
   
            for v in V_neg:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_neg:
                meshfile.write(str(0) + '\t')
            meshfile.write('\n')
            for v in V_neg:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')

            for v in V_neg:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')

            v_unstable = np.sqrt(I) - self.labda

            V_inter = [v_unstable]
            while V_inter[-1] > -np.sqrt(I) + self.labda:
                v_new = self.VQifNegative(V_inter[-1], self.dt)
                V_inter.append(v_new)
    

            for v in V_inter:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_inter:
                meshfile.write(str(0) + '\t')
            meshfile.write('\n')
            for v in V_inter:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_inter:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')


            V_posneg = [self.V_max*(1 + self.epsilon), self.V_max]
            while V_posneg[-1] > np.sqrt(I) + self.labda:
                v_new = self.VQifNegative(V_posneg[-1],-self.dt)
                V_posneg.append(v_new)

            V_pos = V_posneg[::-1]

            for v in V_pos:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_pos:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            for v in V_pos:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in V_pos:
                meshfile.write(str(0.0) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')
            meshfile.write('end')

    def generateQifMesh(self):
        ''' Generates a mesh for QIF neurons. '''
        if self.I < 0:
            self.__generateQifMeshNegative__()
        if self.I > 0:
            self.__generateQifMeshPositive__()

    def generateQifStationary(self):
        '''Must be called only when the mesh already has been generated.'''
        meshname = self.basename + '.mesh'
        statname = self.basename + '.stat'
        
        if self.I < 0:

            m=mesh3.Mesh(self.basename + '.mesh')
            points=m.cells[1][-1].points # cells[0] is reserved
            vstat1 = max([ p[0] for p in points])
            points=m.cells[2][-1].points
            vstat2 = min( [p[0] for p in points])

            points=m.cells[2][0].points
            vstat3 = max( [p[0] for p in points])
            points=m.cells[3][0].points
            vstat4 = min( [p[0] for p in points])

            with open( statname,'w') as fstat:
                fstat.write('<Stationary>\n')
                format = "%.9f"
                fstat.write('<Quadrilateral>')
                fstat.write('<vline>' +  str(vstat1) + ' ' + str(vstat1) + ' ' +  str(vstat2) + ' ' + str(vstat2) + '</vline>')
                fstat.write('<wline>' +  str(0)      + ' ' + str(self.w) + ' ' +  str(self.w) + ' ' + str(0)      + '</wline>')
                fstat.write('</Quadrilateral>\n')
                fstat.write('<Quadrilateral>')
                fstat.write('<vline>' +  str(vstat3) + ' ' + str(vstat3) + ' ' +  str(vstat4) + ' ' + str(vstat4) + '</vline>')
                fstat.write('<wline>' +  str(0)      + ' ' + str(self.w) + ' ' +  str(self.w) + ' ' + str(0)      + '</wline>')
                fstat.write('</Quadrilateral>\n')
                fstat.write('</Stationary>')

        if self.I > 0:
            with open( statname,'w') as fstat:
                fstat.write('<Stationary>\n')
                format = "%.9f"
                fstat.write('</Stationary>')


    def generateQifReversal(self):
        '''Requires the base name of a mesh file. It will add the extension '.mesh', try to open               
        the mesh file, and on the assumption that the mesh has been generated by cond.py 
        write a file bn + '.rev', which contains mappings to the reversal bin, which is assumed to be 0,0.      
        A user that later wants to associate  the '.rev file with the mesh needs to make sure that                                                                        
        a bin 0,0 has been created, for example by using InsertStationary.'''

        revname = self.basename  + '.rev'
    
        with open( revname,'w') as f:

            if self.I < 0:
                f.write('<Mapping type=\"Reversal\">\n')
                for i in range(1,4):
                    f.write(str(i) + ',' + str(0))
                    f.write('\t')
                    f.write(str(0) + ',' + str(0))
                    f.write('\t')
                    f.write(str(1.0) + '\n')
                f.write('</Mapping>')
            if self.I > 0:
                f.write('<Mapping type=\"Reversal\">\n')
                f.write(str(1) + ',' + str(0))
                f.write('\t')
                f.write(str(1) + ',' + str(0))
                f.write('\t')
                f.write(str(1.0) + '\n')
                f.write('</Mapping>')

    
if __name__ == "__main__":
    g=QifMeshGenerator('qif')
    g.generateQifMesh()
    g.generateQifReversal()
    g.generateQifStationary()
