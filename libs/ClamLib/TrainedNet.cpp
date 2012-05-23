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

#include "ClamLibException.h"
#include "TrainedNet.h"

using namespace ClamLib;

TrainedNet::TrainedNet
(
	const SpatialConnectionistNet&	net,
	const vector<D_TrainingUnit>&	vec_pat
):
_net(net),
_vec_pattern(vec_pat)
{
}	

TrainedNet::TrainedNet
(
	istream& s
):
_net(NetFromInput(s))
{
	int n;
	s >> n;
	_vec_pattern.clear();


	D_TrainingUnit tu;
	for (int i = 0; i < n; i++)
	{
		tu.FromStream(s);
		_vec_pattern.push_back(tu);
	}
	string str;
	s >> str;
	if ( str != ToEndTag(Tag()) )
		throw ClamLibException("TrainedNet end tag expected");
}

TrainedNet::~TrainedNet()
{
}

bool TrainedNet::FromStream(istream& s)
{
	return true;
}

bool TrainedNet::ToStream(ostream& s) const
{
	s << Tag() << "\n";
	_net.ToStream(s);

	s << _vec_pattern.size() << "\n";
	ostream_iterator<D_TrainingUnit> it(s," ");
	copy
	(
		_vec_pattern.begin(),
		_vec_pattern.end(),
		it
	);

	s << "\n";
	s << "\n" << ToEndTag(Tag()) << "\n";

	return true;
}

SpatialConnectionistNet TrainedNet::NetFromInput(istream & s)
{
	string str;

	s >> str;

	if (str != this->Tag() )
		throw ClamLibException("TrainedNet tag expected");

	return SpatialConnectionistNet(s);
}

string TrainedNet::Tag() const
{
	return string("<TrainedNet>");
}

ostream& ClamLib::operator<<(ostream& s, const TrainedNet& net)
{
	net.ToStream(s);
	return s;
}