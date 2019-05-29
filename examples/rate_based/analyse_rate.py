import numpy as np
import ROOT
import sys
import os
import directories3 as directories
'''Analyse the response curve data.'''

# Don't want flashing canvasses.

ANALYTIC_DATAFILE    = 'analytic.dat'
POPULATION_DATAFILE  = 'pop.dat'

ROOT.gROOT.SetBatch(True)

cwd = os.getcwd()
jobpath  = os.path.join(cwd,'rate','joblist')

f = open(jobpath)
lines = f.readlines()

def ExtractPopulationResults():
	with open(POPULATION_DATAFILE,'w') as fpop:
		fpop.close()

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

		with open(POPULATION_DATAFILE,'a') as fpop:
			items= path[-1].split('_')
			print(items[2])
			print(items[1])
			fpop.write(items[2] + ' ' + items[1] + ' ' +  str(p[0]) + '\n')
		c=ROOT.TCanvas('c'+ path[-1])
		g.Draw('AL')
		c.SaveAs(path[-1] + '.png')


def ParseAnalyticResponse():
	s1=[]
	s2=[]
	s5=[]
	s7=[]

	with open(ANALYTIC_DATAFILE) as fan:
		lines=fan.readlines()
	for line in lines[1:]:
		vals = [float(x) for x in line.split()]
		if vals[1] == 0.001:
			s1.append([vals[0],vals[2]])
		if vals[1] == 0.002:
			s2.append([vals[0],vals[2]])
		if vals[1] == 0.005:
			s5.append([vals[0],vals[2]])
		if vals[1] == 0.007:
			s7.append([vals[0],vals[2]])
				 
	return s1, s2, s5, s7

def ConvertToGraph(s):
	mu=[]
	f=[]
	for pair in s:
		mu.append(pair[0])
		f.append(pair[1])
	g=ROOT.TGraph(len(mu),np.array(mu),np.array(f))
	return g


def DrawAnalytic(s1, s2, s5, s7):
	graphs=[]
	graphs.append(ConvertToGraph(s1))
	graphs[-1].Draw('L')
	graphs.append(ConvertToGraph(s2))
	graphs[-1].SetLineStyle(2)
	graphs[-1].Draw('L')
	graphs.append(ConvertToGraph(s5))
	graphs[-1].SetLineStyle(3)
	graphs[-1].Draw('L')
	return graphs

def ParsePopResponse():
	m=[]
	f=[]
	with open(POPULATION_DATAFILE) as fp:
		lines = fp.readlines()
		for line in lines:
			items = [float(x) for x in line.split()]
			m.append(items[0])
			f.append(items[2])
	g=ROOT.TGraph(len(m),np.array(m),np.array(f))
	return g

def DrawComparisonPlot():
	s1, s2, s5, s7 = ParseAnalyticResponse()
	g = ParsePopResponse()
	ROOT.gStyle.SetOptStat(0)
	c2=ROOT.TCanvas()
	hh=ROOT.TH2F('h','',500,16e-3,21e-3,500,0.,24.)
	hh.SetXTitle('#mu (mV)')
	hh.SetYTitle('f (spikes/s)')
	hh.Draw()


	graphs = DrawAnalytic(s1, s2, s5, s7)

	l=ROOT.TLegend(0.1,0.7,0.4,0.9)
	l.AddEntry(graphs[0], '#sigma = 1 mV','lp')
	l.AddEntry(graphs[1], '#sigma = 2 mV','lp')
	l.AddEntry(graphs[2], '#sigma = 5 mV','lp')
	l.Draw()

	g.SetMarkerStyle(3)
	g.SetMarkerSize(1)
	g.SetMarkerColor(2)
	g.Draw('P')
	c2.Print('anarate.pdf')
if __name__ == "__main__":
	ExtractPopulationResults()
	DrawComparisonPlot()
