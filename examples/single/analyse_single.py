import ROOT
import sys
import os
import directories
import numpy as np

'''Analyse the response curve data.'''

# Don't want flashing canvasses.

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

jobpath  = os.path.join(directories.miind_root(),'build','jobs','single','joblist')


def nodecay():
	filepath = os.path.split(jobpath)[0]
	path = os.path.join(filepath,'no_decay_0.root')
	f=ROOT.TFile(path)
	c=ROOT.TCanvas('c')
	c.SetGrayscale()

	h = ROOT.TH2F("h","No leakage",500,0.01,1.,500,0.,40.)
	h.SetXTitle('V')
	h.SetYTitle('#rho')
	h.GetYaxis().SetLabelOffset(0.02)
	h.Draw()
	g2=f.Get('grid_0_1.1513041e-05')
	g2.SetLineColor(ROOT.kYellow+4)
	g2.SetFillColor(ROOT.kYellow+4)
	g2.SetFillStyle(1001)
	g3=f.Get('grid_0_3.2236514e-05')
	g3.SetLineColor(ROOT.kYellow-5)
	g3.SetFillColor(ROOT.kYellow-5)
	g4=f.Get('grid_0_0.00099012149')
	g4.SetLineColor(ROOT.kYellow-8)
	g4.SetFillColor(ROOT.kYellow-8)
	g4.Draw('BL)')
	g3.Draw('BL')
	g2.Draw('BL')
	c.SaveAs('no_decay.png')

def noinput():

	filepath = os.path.split(jobpath)[0]
	path = os.path.join(filepath,'no_input_0.root')
	f=ROOT.TFile(path)

	c=ROOT.TCanvas('c')
	c.SetGrayscale()

	h = ROOT.TH2F("h","No input",500,0.01,1.,500,0.,40.)
	h.SetXTitle('V')
	h.SetYTitle('#rho')
	h.GetYaxis().SetLabelOffset(0.02)
	
	h.Draw()
	g=f.Get('grid_0_0.0013843197')
	g.SetLineColor(ROOT.kCyan)
	g.Draw('L')
	g2=f.Get('grid_0_0.010151678')
	g2.SetLineColor(ROOT.kCyan+1)
	g2.Draw('L')
	g3=f.Get('grid_0_0.020303356')
	g3.SetLineColor(ROOT.kCyan+2)
	g3.Draw('L')
	g4=f.Get('grid_0_0.040145271')      
	g4.SetLineColor(ROOT.kCyan+3)
	g4.Draw('L')
	g5=f.Get('grid_0_0.080290542')      
	g4.SetLineColor(ROOT.kCyan+4)
	g5.Draw('L')

	p=ROOT.TPad("p","",0.5,0.5,0.90,0.85)
	p.Draw()
	p.cd()
	h2=ROOT.TH2F("h2","",500,0.,0.3,500,0.,0.000001)
	h2.Draw()
	g5.Draw('L')
	c.SaveAs('no_input.png')

def single_state():
	filepath = os.path.split(jobpath)[0]
	path = os.path.join(filepath,'response_0.root')
	f=ROOT.TFile(path)

	c = ROOT.TCanvas("c3")
	h = ROOT.TH2F("h","Density",500,0.01,1.,500,0.,2.)
	h.SetXTitle('V')
	h.SetYTitle('#rho')
	h.GetYaxis().SetLabelOffset(0.02)	
	h.Draw()

	g=f.Get('grid_0_0.29901305')
	g.Draw('L')
	c.SaveAs('steadystate.png')

	c2=ROOT.TCanvas("c4")
	h=ROOT.TH2F("h2","Rate",500,0.,0.3,500,0.,20.)
	h.SetYTitle('f (spikes/s)')
	h.SetXTitle('t (s)')
	h.GetYaxis().SetLabelOffset(0.02)	
	h.Draw()
	g=f.Get('rate_0')
	g.Draw('L')
	c2.SaveAs('frate.png')
	
def single():
	single_state()

def parsejoblist():
	with open(jobpath) as f:
		lines = f.readlines()
		for line in lines:
			job =  os.path.split(line.strip())[1]
			print job
			if  job == 'no_decay':
				nodecay()
			if job == 'no_input':
				noinput()	
			if job == 'response':
				single()

if __name__ == '__main__':
	parsejoblist()
