import miind.mesh3 as mesh
import miind.writemesh as writemesh
import numpy as np

class LifMeshGenerator:
    '''This class helps to generate a leaky-integrate-and-fire mesh. The neural parameters are:
    tau_m:        :membrane time constant
    V_rest:       :resting potential
    V_threshold:  :threshold potential
    Here there is :no need to set a refractive period, this can be done elsewhere.
    The following :parameters need to be set to define the grind and require user input.
    dt            :the step size of the grid, which is the main determinant of the bin size
    self.N_grid   : determines how close the grid hugs V_rest. A large value results in an exponential pile up of the grid close to V_rest. A small value leaves a big gap.

    The following parameters will be set to define the grid, and often don't require user setting

    V_max         :must be larger than V_threshold, but not by much. MIIND needs a threshold cell, rather than a boundary and V_max create a cell boundary
                  :above threshold for this purpose. The default value is V_max + epsilon
    epsilon:      :just a small value
    self.strip_w  : an arbitrary value for the strip width.
    self.lambda   : an arbitrary small value for the reset bin. Should not touch other grid cellss.'''


    def __init__(self, basename, tau = 10e-3, V_threshold = -50., V_rest = -65.0, V_min = -80., dt = 0.0001, N_grid=300):
        self.tau             = tau                               # membrane time constant in s
        self.V_threshold     = V_threshold                       # threshold in V
        self.epsilon         = 0.001                             # padding as fraction of the threshold potential
        self.labda           = 0.0001                            # fiducial bin size
        self.V_rest          = V_rest                            # reversal/rest potential (also the reset potential)
        self.V_min           = V_min                             # guaranteed minimum value of the grid
        self.V_max           = self.V_threshold + self.epsilon   # guaranteed maximum value of the grid
        self.N_grid          = N_grid                            # number of points in the interval (V_rest, self.V_threshold);
                                                                 # note that this parameter controls how close to V_rest the grid extends, it does NOT control the bindwidth,
                                                                 # which is determined by self.dt. Also note that the grid may extend to negative values
                                                                 # e.g if V_min = 2*V.rest - V_threshold, the grid holds double this number of bins
        self.dt              = dt                                # timestep for each bin
        self.strip_w         = 0.005                             # arbitrary value for strip width
        self.basename        = basename
        self.pos_vs = []
        self.neg_vs = []

    def generateLifMesh(self):

        if self.V_min > self.V_rest:
            raise ValueError ("self.V_min must be less than V_rev.")
        if self.V_max < self.V_threshold+self.epsilon:
            raise ValueError ("self.V_max must be greater than or equal to self.V_threshold.")

        with open(self.basename + '.mesh','w') as meshfile:
            meshfile.write('ignore\n')
            meshfile.write('{}\n'.format(self.dt))

            ts = self.dt * np.arange(self.N_grid)
            self.pos_vs = self.V_rest + (self.V_threshold-self.V_rest)*np.exp(-ts/self.tau)
            self.pos_vs = np.insert(self.pos_vs, 0, self.V_max)
            self.neg_vs = self.V_rest + (self.V_min-self.V_rest)*np.exp(-ts/self.tau)

            if len(self.neg_vs) > 0:
                for v in self.neg_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in self.neg_vs:
                    meshfile.write(str(0.0) + '\t')
                meshfile.write('\n')
                for v in self.neg_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in self.neg_vs:
                    meshfile.write(str(self.strip_w) + '\t')
                meshfile.write('\n')
                meshfile.write('closed\n')

                for v in self.pos_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in self.pos_vs:
                    meshfile.write(str(self.strip_w) + '\t')
                meshfile.write('\n')
                for v in self.pos_vs:
                    meshfile.write(str(v) + '\t')
                meshfile.write('\n')
                for v in self.pos_vs:
                    meshfile.write(str(0.0) + '\t')
                meshfile.write('\n')
                meshfile.write('closed\n')
                meshfile.write('end')

        return self.basename + '.mesh'

    def generateLifStationary(self):
        statname = self.basename + '.stat'

        v_plus = self.pos_vs[-1]
        v_min  = self.neg_vs[-1]

        with open(statname,'w') as statfile:
            statfile.write('<Stationary>\n')
            format = "%.9f"
            statfile.write('<Quadrilateral>')
            statfile.write('<vline>' +  str(v_min) + ' ' + str(v_min) + ' ' +  str(v_plus) + ' ' + str(v_plus) + '</vline>')
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

if __name__ == "__main__":
    g=LifMeshGenerator('lif')
    g.generateLifMesh()
    g.generateLifStationary()
    g.generateLifReversal()
