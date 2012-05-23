// Copyright (c) 2005 - 2010 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://clam.sourceforge.net

#ifdef WIN32
#pragma warning (disable: 4267)
#pragma warning (disable: 4996)
#endif

#include <iostream>

using std::cout;
using std::endl;

#include <ClamLib/ClamLib.h>
using ClamLib::CircuitInfo;
using ClamLib::CircuitDescription;
using ClamLib::CircuitNodeRole;
using ClamLib::DynamicSubNetwork;
using ClamLib::SimulationInfoBlock;
using ClamLib::SimulationOrganizer;
using ClamLib::SimpleCircuitCreator;
using ClamLib::TrainedNet;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::WilsonCowanAlgorithm;
using DynamicLib::WilsonCowanParameter;
using StructnetLib::DenseOverlapLinkRelation;
using StructnetLib::D_OrientedPattern;
using StructnetLib::SpatialConnectionistNet;

int main(){
	cout << "Example for a 2D sheet of nodes" << endl;
	D_DynamicNetwork net;

	LayerDescription desc;
	desc._nr_x_pixels = 8;
	desc._nr_y_pixels = 8;
	desc._nr_features = 8;

	D_OrientedPattern pat(desc._nr_x_pixels,desc._nr_y_pixels,desc._nr_features);
	// convert this to an artificial neural network, so-called TrainedNet

	vector<LayerDescription> vec_desc;
	vec_desc.push_back(desc);
	DenseOverlapLinkRelation lr(vec_desc);

	SpatialConnectionistNet ann_net(&lr);
	vector<D_TrainingUnit> vec_tu;
	TrainedNet tn(ann_net,vec_tu);

	// Now we have to create a circuit at each node of the ANN. In this case the circuit is simple and consists of only one
	// node.

	ClamLib::AddTNToDN convert;
	WilsonCowanParameter par(10e-3,100,1.0);
	WilsonCowanAlgorithm alg(par);

	SpatialPosition pos(0.0,0.0,0.0,0.0);
	SimpleCircuitCreator creator(&alg,&alg,&net,pos);

	SimulationOrganizer org;
	org.Convert("Area_MT",tn,creator,pat,&net);


	for (DynamicSubNetwork::const_iterator iter = org[0].begin(); iter != org[0].end(); iter++){
		PhysicalPosition pos;
		iter.Position(&pos);
		cout << iter->NumberOfNodes() << " " << (*iter)[0] << " "	<< pos._position_x << " " 
																	<< pos._position_y << " "	
																	<< pos._position_z << " " 
																	<< pos._position_depth << endl;
	}

	cout << "OK signing off" << endl;
}