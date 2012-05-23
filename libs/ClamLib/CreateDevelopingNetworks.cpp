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
#include "CreateDevelopingNetworks.h"
#include "ClamLibException.h"
#include "JocnPattern.h"
#include "TestDefinitions.h"
#include "TrainedNet.h"

using namespace ConnectionismLib;
using namespace ClamLib;
using namespace StructnetLib;
using namespace std;

bool ClamLib::CreateDevelopingNetworks
(
	string						path,
	string*						p_ffd_name,
	string*						p_rev_name,
	TrainedNet**				p_tn, 
	SpatialConnectionistNet**	p_rev
)
{
	*p_ffd_name = NAME_FFD_DEVELOP;
	*p_rev_name = NAME_REV_DEVELOP;

	string absolute_path_ffd = path + string("/") + NAME_FFD_DEVELOP;
	string absolute_path_rev = path + string("/") + NAME_REV_DEVELOP;

	ifstream ist_ffd(absolute_path_ffd.c_str());
	ifstream ist_rev(absolute_path_rev.c_str());
	vector<D_TrainingUnit> vec_tus = CreateJOCNTrainingUnits();

	if ( ist_ffd && ist_rev )
	{
		*p_tn  = new TrainedNet(ist_ffd);
		*p_rev = new SpatialConnectionistNet(ist_rev);
		return true;
	}

	ist_ffd.close();
	ist_rev.close();

	ofstream ost_ffd(absolute_path_ffd.c_str());
	ofstream ost_rev(absolute_path_rev.c_str());

	if ( ! ost_ffd || ! ost_rev)
		throw ClamLibException("Couldn't open developping networks for creation");

	vector<LayerDescription> vec_desc;

	vec_desc.push_back(ClamLib::LAYER_0);
	vec_desc.push_back(ClamLib::LAYER_1);
	vec_desc.push_back(ClamLib::LAYER_2);
	vec_desc.push_back(ClamLib::LAYER_3);
	vec_desc.push_back(ClamLib::LAYER_4);


	SpatialConnectionistNet net =  
		CreateJOCNFFDNet
		(
			vec_desc,
			vec_tus,
			JOCN_ENERGY
		);

	SpatialConnectionistNet reverse_net =
		CreateJOCNREVNet
		(
			vec_desc,
			net,
			vec_tus
		);

	
	TrainedNet tn_fd(net,vec_tus);
	*p_tn  = new TrainedNet(tn_fd);
	*p_rev = new SpatialConnectionistNet(reverse_net);

	ost_ffd << tn_fd;
	ost_rev << reverse_net;

	return true;
}