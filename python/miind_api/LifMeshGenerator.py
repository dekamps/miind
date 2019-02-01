import mesh3 as mesh
import writemesh
import numpy as np

class LifMeshGenerator:
    def __init__(self, basename):
        self.tau             = 10     # membrane time constant in s
        self.V_threshold     = -35.0     # threshold in V
        self.epsilon         = 0.001     # padding as fraction of the threshold potential
        self.labda           = 0.0001      # fiducial bin size
        self.V_rest          = -57.5     # reversal/rest potential (also the reset potential)
        self.V_min           = -70.0       # guaranteed minimum value of the grid
        self.V_max           = -34.99     # guaranteed maximum value of the grid
        self.N_grid          = 150       # number of points in the interval [V_res, self.V_threshold); e.g if self.V_min = self.V_threshold, the grid holds double this number of bins
        self.dt              = 0.0001     # timestep for each bin
        self.strip_w         = 0.005     # arbitrary value for strip width
        self.basename        = basename

    def generateLifMesh(self):

        if self.V_min > self.V_rest:
            raise ValueError ("self.V_min must be less than V_rev.")
        if self.V_max < self.V_threshold+self.epsilon:
            raise ValueError ("self.V_max must be greater than or equal to self.V_threshold.")

        with open(self.basename + '.mesh','w') as meshfile:
            meshfile.write('ignore\n')
            meshfile.write('{}\n'.format(self.dt))

            ts = self.dt * np.arange(self.N_grid)
            pos_vs = self.V_rest + (self.V_threshold-self.V_rest)*np.exp(-ts/self.tau)
            pos_vs = np.insert(pos_vs, 0, self.V_max)
            neg_vs = self.V_rest + (self.V_min-self.V_rest)*np.exp(-ts/self.tau)

            if len(neg_vs) > 0:
                for v in neg_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in neg_vs:
                    meshfile.write(str(0.0) + '\t')
                meshfile.write('\n')
                for v in neg_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in neg_vs:
                    meshfile.write(str(self.strip_w) + '\t')
                meshfile.write('\n')
                meshfile.write('closed\n')

                for v in pos_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in pos_vs:
                    meshfile.write(str(self.strip_w) + '\t')
                meshfile.write('\n')
                for v in pos_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in pos_vs:
                    meshfile.write(str(0.0) + '\t')
                meshfile.write('\n')
                meshfile.write('closed\n')
                meshfile.write('end')

        return self.basename + '.mesh'

    def generateLifStationary(self):
        statname = self.basename + '.stat'

        v_plus = self.V_rest + self.labda
        self.V_min  = self.V_rest - self.labda

        with open(statname,'w') as statfile:
            statfile.write('<Stationary>\n')
            format = "%.9f"
            statfile.write('<Quadrilateral>')
            statfile.write('<vline>' +  str(self.V_min) + ' ' + str(self.V_min) + ' ' +  str(v_plus) + ' ' + str(v_plus) + '</vline>')
            statfile.write('<wline>' +  str(0)     + ' ' + str(self.strip_w)     + ' ' +  str(self.strip_w)      + ' ' + str(0)      + '</wline>')
            statfile.write('</Quadrilateral>\n')
            statfile.write('</Stationary>')

    def generateLifReversal(self):
        revname = self.basename  + '.rev'
        m=mesh.Mesh(self.basename + '.mesh')

        with open(revname,'w') as revfile:
            revfile.write('<Mapping type=\"Reversal\">\n')
            for i in range(1,len(m.cells)):
                revfile.write(str(i) + ',' + str(0))
                revfile.write('\t')
                revfile.write(str(0) + ',' + str(0))
                revfile.write('\t')
                revfile.write(str(1.0) + '\n')
            revfile.write('</Mapping>')
