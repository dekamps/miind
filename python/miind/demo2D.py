#!/usr/bin/env python

import ROOT
import os
import miind.visualize as visualize
import sys
import time
import subprocess
 
MONITOR=os.path.basename(os.getcwd())

def processlist(modelname):
    filelist=subprocess.check_output(["ls", modelname + '_mesh']).split()
    sortlist=sorted(filelist, key=lambda item: (float(item.split('_')[-2]) ))
    f=open('processlist.txt','w')
    for fi in sortlist:
        f.write('file \'' + fi + '.pdf\'\n' )

def display(modelname, mesh, step):
    m=visualize.ModelVisualizer(modelname)
    while(1):
        m.showfile( modelname + '_mesh' + '/mesh_' + mesh + '_' + step + '_1', colorlegend = [1e-6,1, 100])

def loop(modelname,option,batch=False):

    if batch == True:
        print 'switching to batch'	
        ROOT.gROOT.SetBatch(True) 

    m=visualize.ModelVisualizer(modelname)
    i = 0

    while(1):
    
        filelist=subprocess.check_output(["ls", modelname + '_mesh']).split()
        # nice, but filelist is ordered alphabetically. want sort on time which we expect (!) is the last field
        sortlist=sorted(filelist, key=lambda item: (float(item.split('_')[-2]) ))
        
        if (i == len(sortlist) ): break
        if (len(sortlist) > 0):
            timestring = sortlist[i].split('_')[-2]
            print sortlist[i], timestring
            if (option == ''):
                m.showfile( modelname + '_mesh' + '/' + sortlist[i], runningtext = 't = ' + timestring, colorlegend = [1e-6,1, 100])
            else:
                m.showfile( modelname + '_mesh' + '/' + sortlist[i], pdfname=sortlist[i], runningtext = 't = ' + timestring, colorlegend = [1e-6,1.,100])
                
            i+=1

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loop(sys.argv[1],'')
    if len(sys.argv) == 3:
        loop(sys.argv[1],'y')
    if len(sys.argv) == 4:
	    loop(sys.argv[1],'y',True)
    if len(sys.argv) == 5: # expect demo2Dpy <placeholder i.e disp> model, mesh, step
	    display(sys.argv[2],sys.argv[3],sys.argv[4])
