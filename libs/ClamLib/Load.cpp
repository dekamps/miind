#include <vector>
#include <iostream>
#include <TFile.h>
#include <TROOT.h>
#include "SimulationInfoBlock.h"
#include "WeightedLink.h"

using std::cout;
using std::endl;

void Load()
{
  gROOT->ProcessLine(".L Id.cpp+");
  gROOT->ProcessLine(".L WeightedLink.h+");
  gROOT->ProcessLine(".L CircuitInfo.cpp+");
  gROOT->ProcessLine(".L RootLayerDescription.cpp+");
  gROOT->ProcessLine(".L RootLayeredNetDescription.cpp+");
  gROOT->ProcessLine(".L SimulationInfoBlock.cpp+");
  gROOT->ProcessLine(".L Macro.cpp+");
}
