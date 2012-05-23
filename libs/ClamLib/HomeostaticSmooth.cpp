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
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "HomeostaticSmooth.h"
#include "../StructnetLib/StructnetLib.h"

using namespace ClamLib;
using namespace StructnetLib;

namespace {
	void CollectActivities(SpatialConnectionistNet& net, vector<double>* p_vec){
	const vector<LayerDescription>& vec_dim = net.Dimensions();
	vector<double>& vec_act = *p_vec;


	for (Layer l = 0; l < net.NumberOfLayers(); l++ ){
		int n_active = 0;
		for (Index ix = 0; ix < vec_dim[l]._nr_x_pixels; ix++){
			for (Index iy = 0; iy < vec_dim[l]._nr_y_pixels; iy++){
				for (Index ift = 0; ift < vec_dim[l]._nr_features; ift++){
					PhysicalPosition pos;
					pos._position_x = ix;
					pos._position_y = iy;
					pos._position_z = l;
					pos._position_depth = ift;

					NodeId id = net.Id(pos);
					double f_act = net.GetActivity(id);
	
					if (f_act != 0){
						n_active++;
						// ANN activity can be positive or negative
						vec_act[l] += fabs(f_act);
					}
				}
			}
		}
		if ( n_active > 0 )
			vec_act[l] /= n_active;
	}


	}

	vector<double> AverageLevel(TrainedNet& net){
		vector<double> vec_ret(net._net.NumberOfLayers());
		Number n_pat = net._net.NumberOfInputNodes();
		for (Index i = 0; i < n_pat; i++)
		{
			D_Pattern pat(n_pat);
			pat.Clear();
			pat[i] = 0.1; // TODO: replace literal; the precise value diesn't matter though
			net._net.ReadIn(pat);
			net._net.Evolve();
			CollectActivities(net._net,&vec_ret);
		}

		return vec_ret;
	}
}
void ClamLib::HomeostaticSmooth(TrainedNet* p_net, double scale)
{
	// determine the average level of the input
	vector<double> vec_act = AverageLevel(*p_net);

	// determine the average level of activity of active neurons per layer. By definition, the activity of the first layer,
	// layer 0 is the average level of inputs

	for (Layer l = 1; l < p_net->_net.NumberOfLayers(); l++)
		p_net->_net.ScaleWeights(l,scale*vec_act[l-1]/vec_act[l]);
}

