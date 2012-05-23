#include <vector>
#include <iostream>
#include <TFile.h>
#include <TROOT.h>
#include "SimulationInfoBlock.h"
using std::cout;
using std::endl;
using namespace ClamLib;
void Macro()
{

  TFile* p_file = new TFile("simulationorganizersmalldirect.root");

  SimulationInfoBlock* p_block = (SimulationInfoBlock*)p_file->Get("simulationorganizersmalldirect");
  
  cout << p_block->InfoVector().size() << endl;

  cout << "Static NodeId: " << 3 << endl;
  ClamLib:: Id id = (p_block->InfoVector())[3][0];

  cout << "Dynamic NodeId: " << id  << endl;
}
