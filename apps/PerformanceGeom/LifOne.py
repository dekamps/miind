import ROOT
import string
import numpy as np

#  NEST simulation results
TEST_PATH = '../../build/apps/PerformanceGeom/test/'
NEST_FILE = 'om_counts.dat'
ROOT_FILE = 'singlepoptest_0.root'
DENS_FILE = 'density_count.dat'

def parse_sim_res_dense():
    f = open(DENS_FILE)
    lines = f.readlines()

    meta = lines[0].split(';')
    n_bins = int(meta[0].split(':')[1]) - 1
    v_min =  float(meta[1].split(':')[1])
    v_max =  float(meta[2].split(':')[1])

    dv = (v_max - v_min)/n_bins
    vs = np.arange(v_min + 0.5*dv, v_max , dv)
    ve = np.zeros(len(vs))
    counts = np.array([ float(x) for x in lines[1:] ])
    error  = np.sqrt(counts)
    tot = sum(counts)
    area = tot*dv
    counts /= area
    error  /= area
    return vs, counts, ve, error

def parse_sim_res_spike():
    f = open(NEST_FILE)
    lines = f.readlines()

    meta = lines[0].split(';')
    nr_neurons = int(meta[0].split(':')[1])
    dt = float(meta[1].split(':')[1])
    t_end = float(meta[2].split(':')[1])

    counts = []
    times  = []
    t = 0.5*dt
    for line in lines[1:]:
        times.append(float(t)/1000.)
        counts.append(int(line))
        t += dt

    t = np.array(times)
    c = np.array(counts)
    e = np.sqrt(c)

    mult = (1./(dt*1e-3*float(nr_neurons)))
    print c*mult

    return t, c*mult, e*mult


def plot_rate_results():
    ROOT.gStyle.SetOptStat(0)
    type=111
    ps = ROOT.TSVG('rate.svg',type)
    ps.Range(12.0,5.0)
    k  = ROOT.TCanvas()
    f  = ROOT.TFile(TEST_PATH + 'singlepoptest_0.root')

    h = ROOT.TH2F('h','Jump response',500,0.,0.3,500,0.,20.)
    h.GetXaxis().SetTitle("t (s)")
    h.GetYaxis().SetTitle("f (spikes/s)")
    h.Draw()
    g  = f.Get('rate_1')
    g.Draw('L')

    t, c, e_r = parse_sim_res_spike()
    e_t = np.array(len(t)*[0.])
    ge = ROOT.TGraphErrors(len(t),t,c,e_t,e_r)
    ge.SetLineColor(2)
    ge.Draw('P')
    k.Update()
    ps.Close()

def plot_dens_results():
    ROOT.gStyle.SetOptStat(0)
    type=111
    ps = ROOT.TSVG('dense.svg',type)
    ps.Range(12.0,5.0)
    k  = ROOT.TCanvas()
    f  = ROOT.TFile(TEST_PATH + 'singlepoptest_0.root')
    h = ROOT.TH2F("h","Steady state density",500,0.,1.,500,0.,2.)
    h.Draw()
    h.GetXaxis().SetTitle("V (rescaled)")
    h.GetYaxis().SetTitle("#rho")
    g = f.Get('grid_1_10.000325')
    g.Draw("L");
    v, c, ev, ec =  parse_sim_res_dense()
    
    ge = ROOT.TGraphErrors(len(v), v, c, ev, ec)
    ge.SetLineColor(2)
    ge.Draw("P")
    ps.Close()
    
    return

if __name__ == "__main__":
    plot_rate_results()
    plot_dens_results()

