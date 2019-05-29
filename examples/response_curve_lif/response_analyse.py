import ROOT
import sys
import os
import directories3 as directories
'''Analyse the response curve data.'''

# Don't want flashing canvasses.

ROOT.gROOT.SetBatch(True)

cwd = os.getcwd()
jobpath  = os.path.join(cwd,'response','joblist')

f = open(jobpath)
lines = f.readlines()

for line in lines:
    path = os.path.split(line.strip())
    print(path[-1])
    f=ROOT.TFile(os.path.join(line.strip() ,  path[-1] + '_0.root'))

    # Get the firing rate response of the LIF population
    g=f.Get('rate_0')

    # Fit and write out
    fun=ROOT.TF1('func1','pol1',0.8,1.0)
    g.Fit('func1','R')
    p = fun.GetParameters()

    with open('response.dat','a') as f:
        items= path[-1].split('_')
        print(items[2])
        print(items[1])
        f.write(items[2] + ' ' + items[1] + ' ' +  str(p[0]) + '\n')
    c=ROOT.TCanvas('c'+ path[-1])
    g.Draw('AL')
    c.SaveAs(path[-1] + '.png')
