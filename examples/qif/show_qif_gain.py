import numpy as np
import os
import subprocess as sp
import ROOT
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT.gROOT.SetStyle('Plain')
ROOT.gStyle.SetOptStat(0)

dst_resp_dir = 'DST_response'
dst_spec_dir = 'DST_spectrum'

def process_file(rate_path):
    rate_file = rate_path.split('/')[-1]

    m = float(rate_file.split('_')[0])
    s = float(rate_file.split('_')[1])
    with open(rate_path) as fin:
        lines=fin.readlines()
    fs = lines[1].split()
    freq = float(fs[-1])
    return m, s, freq

def list_files(dir):
    ''' We know that different mu values for sigma = 1, 2, 5 mV have been generated. Plot these points'''
    su1 = []
    su2 = []
    su5 = []
    files= sp.check_output(['ls', dir]).split('\n')
    ratefiles = [ f for f in files if 'rate' in f]
    return ratefiles

def show_curve():
    ''' We know that different mu values for sigma = 1, 2, 5 mV have been generated. Plot these points'''
    su1 = []
    su2 = []
    su5 = []
    files= sp.check_output(['ls', dst_resp_dir]).split('\n')
    ratefiles = list_files(dst_resp_dir)
    
    
    for f in ratefiles:
        m, s, f = process_file(os.path.join(dst_resp_dir,f))
        if np.fabs(s - 0.158) < 1e-6:
            su1.append([m,f])
        if np.fabs(s - 0.316) < 1e-6:
            su2.append([m,f])
        if np.fabs(s - 0.791) < 1e-6:
            su5.append([m,f])

    s1 = sorted(su1, key = lambda entry: entry[0])
    s2 = sorted(su2, key = lambda entry: entry[0])
    s5 = sorted(su5, key = lambda entry: entry[0])

    print s2
    c=ROOT.TCanvas()
    h=ROOT.TH2F("h","Gain curve (QIF)",500, 0., 10., 500, 0., 25.)    
    h.SetXTitle('\mu (V)')
    h.SetYTitle('f (spikes/s)')
    h.Draw()

    m1 = [ p[0] for p in s1 ] 
    f1 = [ p[1] for p in s1 ] 
    g1 = ROOT.TGraph(len(m1), np.array(m1), np.array(f1))
    g1.SetMarkerStyle(2)
    g1.SetMarkerSize(0.5)
    g1.Draw('P')


    m2 = [ p[0] for p in s2 ] 
    f2 = [ p[1] for p in s2 ] 
    g2 = ROOT.TGraph(len(m2), np.array(m2), np.array(f2))
    g2.SetMarkerStyle(3)
    g2.SetMarkerSize(0.5)
    g2.Draw('P')


    m5 = [ p[0] for p in s5 ] 
    f5 = [ p[1] for p in s5 ] 

    g5 = ROOT.TGraph(len(m5), np.array(m5), np.array(f5))
    g5.SetMarkerStyle(4)
    g5.SetMarkerSize(0.5)
    g5.Draw('P')

    c.SaveAs('QIFGain.pdf')

def sinusoid(x, base, a, omega, delta):
    return a*np.sin(2*np.pi*omega*x+delta) + base

def process_spect_file(f):
    print f
    om = float(f.split('/')[1].split('_')[1])/(2*np.pi)
    with open(f) as fin:
        lines = fin.readlines()
        ts = [float(x) for x in lines[0].split()]
        fs = [float(x) for x in lines[1].split()]
    popt, pcov = curve_fit(sinusoid, ts[len(ts)/2:], fs[len(fs)/2:],bounds = ([0., 0, 0.9*om,-np.pi],[20., 10.,1.1*om,np.pi]) )
    plt.plot(ts,fs)


    fits = []
    for i, t in enumerate(ts):
        fits.append(sinusoid(ts[i],popt[0],popt[1],popt[2],popt[3]))
    plt.axis([5.0,10.,0.,19.])
    plt.plot(ts,fits)
    plt.show()


    return popt[1],popt[3]

def show_spectrum():
    oss=[]
    ass=[]
    ratefiles = list_files(dst_spec_dir)
    for f in ratefiles:
        print f 
        om = float(f.split('_')[1])
        oss.append(om)
        a, delta = process_spect_file(os.path.join(dst_spec_dir,f))
        ass.append(a)

    plt.loglog(oss,ass,'+')
    plt.show()

if __name__ == "__main__":
    show_curve()
    show_spectrum()


