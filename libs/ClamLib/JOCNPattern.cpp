// Copyright (c) 2005 - 2009 Marc de Kamps
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
#include "JocnPattern.h"
#include "TestDefinitions.h"

using namespace ConnectionismLib;
using namespace ClamLib;
using namespace StructnetLib;

D_OrientedPattern ClamLib::JOCNPattern(PatternShape type)
{
	D_OrientedPattern pat_ret(24,24,4);
	pat_ret.Clear();

	switch(type)
	{
	case SQUARE:

		pat_ret[149]  = pat_ret[150]  = pat_ret[151]  = 1.0;
		pat_ret[1278] = pat_ret[1302] = pat_ret[1326] = 1.0;
		break;

	case DIAMOND:

		pat_ret[701]  = pat_ret[726]  = pat_ret[751]  = 1.0;
		pat_ret[1855] = pat_ret[1878] = pat_ret[1901] = 1.0;
		break;

	case CROSS_DIAGONAL:

		pat_ret[702]  = pat_ret[725]  = pat_ret[727]  = pat_ret[750]  = 1.0;
		pat_ret[1854] = pat_ret[1877] = pat_ret[1879] = pat_ret[1902] = 1.0;
		break;

	case CROSS_HORIZONTAL:

		pat_ret[125]  = pat_ret[126]  = pat_ret[127]  = 1.0;
		pat_ret[173]  = pat_ret[174]  = pat_ret[175]  = 1.0;
		pat_ret[1277] = pat_ret[1279] = pat_ret[1301] = 1.0;
		pat_ret[1303] = pat_ret[1325] = pat_ret[1327] = 1.0;
		break;

	}
	return pat_ret;
}

namespace {

	void PushTranslatedVersionsBack
	(
		vector<D_TrainingUnit>*		p_vec,
		D_OrientedPattern			pat_in,
		const D_Pattern&	pat_out
	)
	{
		pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);

		p_vec->push_back(D_TrainingUnit(pat_in,pat_out));

		pat_in.TransX(TR_JOCN);
		p_vec->push_back(D_TrainingUnit(pat_in,pat_out));

		pat_in.TransY(TR_JOCN);
		p_vec->push_back(D_TrainingUnit(pat_in,pat_out));

		pat_in.TransX(-TR_JOCN);
		p_vec->push_back(D_TrainingUnit(pat_in,pat_out));
	}
}

vector<D_TrainingUnit> ClamLib::CreateJOCNTrainingUnits()
{
	vector<D_TrainingUnit> vec_ret;

	// first pattern
	D_Pattern pat_out(NR_JOCN_OUTPUTS);
	pat_out.Clear();
	pat_out[CROSS_HORIZONTAL] = MAX_LINEAR_STATIC_RATE;

	D_OrientedPattern pat_in = JOCNPattern(CROSS_HORIZONTAL);

	PushTranslatedVersionsBack
	(
		&vec_ret,
		pat_in,
		pat_out
	);

	pat_in = JOCNPattern(CROSS_DIAGONAL);

	pat_out.Clear();
	pat_out[CROSS_DIAGONAL] = MAX_LINEAR_STATIC_RATE;

	PushTranslatedVersionsBack
	(
		&vec_ret,
		pat_in,
		pat_out
	);

	pat_in = JOCNPattern(SQUARE);

	pat_out.Clear();
	pat_out[SQUARE] = MAX_LINEAR_STATIC_RATE;

	PushTranslatedVersionsBack
	(
		&vec_ret,
		pat_in,
		pat_out
	);

	pat_in = JOCNPattern(DIAMOND);

	pat_out.Clear();
	pat_out[DIAMOND] = MAX_LINEAR_STATIC_RATE;

	PushTranslatedVersionsBack
	(
		&vec_ret,
		pat_in,
		pat_out
	);

	return vec_ret;
}

SpatialConnectionistNet ClamLib::CreateJOCNFFDNet
(
	const vector<LayerDescription>&	vec_desc,
	const vector<D_TrainingUnit>&	vec_tus,
	double							f_energy
)
{

	DenseOverlapLinkRelation link(vec_desc);

	SpatialConnectionistNet net(&link);

	BackpropTrainingVector
	<
		D_LayeredReversibleSparseImplementation,
		D_LayeredReversibleSparseImplementation::WeightLayerIterator
	> vec_train(&net,ClamLib::NETLIB_TRAINING_PARAMETER);

	vec_train.AcceptTrainingUnitVector(vec_tus);

	while ( vec_train.ErrorValue()> f_energy )
	{
		vec_train.Train();
	}

	D_OrientedPattern pat = JOCNPattern(SQUARE);
	pat.ClipMax(MAX_LINEAR_STATIC_RATE);
	net.ReadIn(pat);
	net.Evolve();


	return net;
}

SpatialConnectionistNet ClamLib::CreateJOCNREVNet
(
	const vector<LayerDescription>&	vec_desc,
	SpatialConnectionistNet&		net,
	const vector<D_TrainingUnit>&	vec_tus
)
{
	ReverseDenseOverlapLinkRelation reverse_link(net);

	SpatialConnectionistNet reverse_net(&reverse_link);

	HebbianTrainingParameter par_train;
	par_train._train_threshold = false;
	par_train._scale = 20.0;
	SpatialConnectionistHebbian heb(par_train);
	reverse_net.SetTraining(heb);
	reverse_net.Initialize();


	D_Pattern pat_dummy(0);
	D_TrainingUnit tu_dummy(pat_dummy,pat_dummy);
	for (Index i  = 0; i < vec_tus.size(); i++ )
	{
		net.ReadIn(vec_tus[i].InPat());
		net.Evolve();
		reverse_net.ReverseActivities(net);
		reverse_net.Train(tu_dummy);
	}

	return reverse_net;
}