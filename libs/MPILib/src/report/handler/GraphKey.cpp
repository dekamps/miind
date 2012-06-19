// Copyright (c) 2005 - 2011 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include <sstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <MPILib/include/report/handler/GraphKey.hpp>
#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {
namespace report {
namespace handler {

GraphKey::GraphKey
(
	NodeId id,
	Time time
):
_id(id),
_time(time)
{
}

GraphKey::GraphKey()
{
}

GraphKey::GraphKey
(
	const std::string& key_string
)
{
	typedef boost::tokenizer<boost::char_separator<char> >
		tokenizer;
	boost::char_separator<char> sep("_");
	tokenizer tokens(key_string, sep);
	tokenizer::iterator tok_iter = tokens.begin();
	if (*tok_iter == std::string("grid") ){
		Index ind = boost::lexical_cast<Index>(*(++tok_iter));
		_id = NodeId(ind);
		_time = boost::lexical_cast<Time>(*(++tok_iter));
		_type = STATEGRAPH;
	}
	if (*tok_iter != std::string("rate") )
		return;
	Index ind = boost::lexical_cast<Index>(*(++tok_iter));
	_id = NodeId(ind);
	_time = 0.0;
	_type = RATEGRAPH;
}

std::string GraphKey::generateName() const
{
	std::ostringstream str;
	str.precision(KEY_PRECISION);
	if (_type == RATEGRAPH)
		str << "rate_" << _id;
	else
		str << "grid_" << _id << "_" << _time;
	return str.str();
}

}// end namespace of handler
}// end namespace of report
}// end namespace of MPILib
