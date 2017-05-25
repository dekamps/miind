import numpy as np
import ROOT
import sys
import os
import directories
import glob
import utilities as ut
import subprocess as sp

'''Analyse the response curve data.'''

# Don't want flashing canvasses.

ANALYTIC_DATAFILE    = 'analytic.dat'
POPULATION_DATAFILE  = 'pop.dat'

ROOT.gROOT.SetBatch(True)

def AnalyseROOTFile(fn):
	f=ROOT.TFile(fn)

	items = fn.split('_')
	mu    = float(items[2])
	sigma = float(items[1])

	# Get the firing rate response of the LIF population
	g=f.Get('rate_0')

	# Fit and write out
	fun=ROOT.TF1('func1','pol1',0.8,1.0)
	g.Fit('func1','R')
	p = fun.GetParameters()

	return p[0], mu, sigma


def ExtractPopResults():
	'''Assumes  the simulation results are in a directory called response, and that
	the simulation results are in subdirectories of that directory. Each subdirectory is
	called: 'response_[sigma]_[mu]_0.root'. '''

	res = [] 
	with ut.cd('response'):
		dirs = sp.check_output(['ls']).split()
		for f in dirs:
			with ut.cd(f):
				files = glob.glob("*.root")
				f, m, s = AnalyseROOTFile(files[0]) # glob returns  a list, even if it has only one member
				res.append([f,m,s])

	with open(POPULATION_DATAFILE,'w') as fpop:
		for r in res:
			fpop.write(str(r[0]) +'\t' + str(r[1]) + '\t' + str(r[2]) +'\n')


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
			m.append(items[1])
			f.append(items[0])
			print items[0], items[2]
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
	c2.Print('anapop.pdf')

if __name__ == "__main__":
#	ExtractPopulationResults()
	ExtractPopResults()
	DrawComparisonPlot()
