// Copyright (c) 2005 - 2007 Marc de Kamps
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

#include "NodeState.h"
#include "DynamicLibException.h"
#include "LocalDefinitions.h"
#include <cassert>
#include <functional>

using namespace DynamicLib;
using namespace UtilLib;
using namespace std;

NodeState::NodeState
			(
				const vector<double>& state
			):_state(state)
{
}

double NodeState::operator [](Index index) const
{
	assert( index < static_cast<Index>(_state.size()) );
	return _state[index];
}

bool NodeState::ToStream(ostream& s) const
{
	s << Tag() << "\n";
	s << static_cast<int>(_state.size()) << "\n";

	copy
	(
		_state.begin(),
		_state.end(),
		ostream_iterator<double>(s," ")
	);

	s << "\n" << ToEndTag(Tag()) << "\n";

	return true;
}


bool NodeState::FromStream(istream& s)
{
	string tag;
	s >> tag;
	if ( tag != Tag() )
		throw DynamicLibException(STR_STATE_PARSE_ERROR);

	int size;
	s >> size;

	if ( ! s.good() )
		return false;

	return StreamToVector<double>
		(
			s,
			size,
			&_state
		);


}

string NodeState::Tag() const
{
	return STR_NODESTATE_TAG;
}
