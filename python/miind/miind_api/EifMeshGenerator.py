import miind.mesh3 as mesh3
import miind.writemesh as writemesh
import numpy as np

class EifMeshGenerator:

    def __init__(self, basename, g_l = 0.3, v_l = -70, v_th = -56, delta_t = 1.48, V_min = -90., V_max = -51.5, dt = 0.1, epsilon = 0.01):
        self.basename = basename
        self.g_l = g_l
        self.v_l = v_l
        self.v_th = v_th
        self.delta_t = delta_t
        self.V_max = V_max
        self.V_min = V_min
        self.dt = dt

        self.epsilon  = epsilon   # padding as fraction of the threshold potential
        self.w        = 0.005   # arbitrary value for strip width
        
        self.calculated_threshold = self.find_threshold()
        print('Found threshold value: ' + str(self.calculated_threshold) )

    def time_step(self, v0, dt):
        v_prime = ((self.g_l * self.delta_t * np.exp((v0 - self.v_th)/self.delta_t)) - (self.g_l * (v0 - self.v_l)))
        
        
        return v0 + dt*v_prime
        
    def time_step_back(self, v0, dt):
        v_prime = ((self.g_l * self.delta_t * np.exp((v0 - self.v_th)/self.delta_t)) - (self.g_l * (v0 - self.v_l)))
       
        
        if v_prime > self.V_max - (self.v_th + self.epsilon):
            raise ValueError('V_max is set too high. Bring it further towards v_th or reduce dt.')
        
        return v0 - dt*v_prime
        
    # Run backward from reversal potential until we reach the threshold (where prime is very small)
    def find_threshold(self):
        v0 = self.v_l + self.epsilon
        while self.time_step_back(v0, self.dt) - v0 > 0.000001:
            v0 = self.time_step_back(v0, self.dt)
        return v0

    def generateEifMesh(self):
        ''' Generates a mesh for EIF neurons. '''
        with open(self.basename + '.mesh','w') as meshfile:

            meshfile.write('ignore\n')
            bin=[]
            meshfile.write(str(self.dt*0.001) +'\n')

            # strip below reversal potential (v_l)
            v0 = self.V_min
            
            vs = [v0]
            while v0 < self.v_l - self.epsilon: # WARNING: if your time step is small, this will produce a LOT of cells!
                v0 = self.time_step(v0, self.dt)
                vs = vs + [v0]

            for v in vs:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(0.0) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')
            
            # strip between threshold and reversale potentials (v_th -> v_l)
            v0 = self.calculated_threshold - self.epsilon
            
            vs = [v0]
            while v0 > self.v_l + self.epsilon: # WARNING: if your time step is small, this will produce a LOT of cells!
                v0 = self.time_step(v0, self.dt)
                vs = vs + [v0]

            for v in vs:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(0.0) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')
            
            # strip between vmax and threshold potentials (V_max -> v_l)
            v0 = self.V_max
            
            vs = [v0]
            while v0 > self.calculated_threshold + self.epsilon: # WARNING: if your time step is small, this will produce a LOT of cells!
                v0 = self.time_step_back(v0, self.dt)
                vs = vs + [v0]
                
            vs.reverse()
            
            for v in vs:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(self.w) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(v) + '\t')
            meshfile.write('\n')
            for v in vs:
                meshfile.write(str(0.0) + '\t')
            meshfile.write('\n')
            meshfile.write('closed\n')
            
            
            meshfile.write('end')

    def generateEifStationary(self):
        '''Must be called only when the mesh already has been generated.'''
        meshname = self.basename + '.mesh'
        statname = self.basename + '.stat'
        
        with open( statname,'w') as fstat:
            fstat.write('<Stationary>\n')
            format = "%.9f"
            fstat.write('<Quadrilateral>')
            fstat.write('<vline>' +  str(self.v_l-self.epsilon) + ' ' + str(self.v_l-self.epsilon) + ' ' +  str(self.v_l+self.epsilon) + ' ' + str(self.v_l+self.epsilon) + '</vline>')
            fstat.write('<wline>' +  str(0)      + ' ' + str(self.w) + ' ' +  str(self.w) + ' ' + str(0)      + '</wline>')
            fstat.write('</Quadrilateral>\n')
            fstat.write('<Quadrilateral>')
            fstat.write('<vline>' +  str(self.v_th-self.epsilon) + ' ' + str(self.v_th-self.epsilon) + ' ' +  str(self.v_th+self.epsilon) + ' ' + str(self.v_th+self.epsilon) + '</vline>')
            fstat.write('<wline>' +  str(0)      + ' ' + str(self.w) + ' ' +  str(self.w) + ' ' + str(0)      + '</wline>')
            fstat.write('</Quadrilateral>\n')
            fstat.write('</Stationary>')


    def generateEifReversal(self):
        revname = self.basename  + '.rev'
    
        with open( revname,'w') as f:
            f.write('<Mapping type=\"Reversal\">\n')
            f.write(str(1) + ',' + str(0))
            f.write('\t')
            f.write(str(0) + ',' + str(0))
            f.write('\t')
            f.write(str(1.0) + '\n')
            
            f.write(str(2) + ',' + str(0))
            f.write('\t')
            f.write(str(0) + ',' + str(0))
            f.write('\t')
            f.write(str(1.0) + '\n')
            
            f.write('</Mapping>')

    
if __name__ == "__main__":
    g=EifMeshGenerator('eif')
    g.generateEifMesh()
    g.generateEifReversal()
    g.generateEifStationary()
