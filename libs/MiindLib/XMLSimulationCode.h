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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_MIINDLIB_XMLSIMULATIONCODE_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_XMLSIMULATIONCODE_INCLUDE_GUARD

#include "XMLSimulation.h"

namespace MiindLib {

	template <class Weight>
	XMLSimulation<Weight>::XMLSimulation
	(
		const string&					connection_type_name,
		const XMLRunParameter&			par_run_xml,
		const SimulationRunParameter&	par_run,
		const algorithm_vector&			vec_alg,
		const node_vector&				vec_nodes,
		const connection_vector&		vec_con
	):
	_connection_type_name(connection_type_name),
	_par_run_xml(par_run_xml),
	_par_run(par_run),
	_vec_alg(vec_alg),
	_vec_nodes(vec_nodes),
	_vec_con(vec_con)
	{
	}

	template <class Weight>
	XMLSimulation<Weight>::~XMLSimulation()
	{
	}

	template <class Weight>
	bool XMLSimulation<Weight>::ToStream(ostream& s) const{
		s << this->Tag() <<"\n";
		s << "<WeightType>" << _connection_type_name << "</WeightType>\n";
		s << "<Algorithms>\n";
		BOOST_FOREACH(algorithm_pointer p_alg, _vec_alg){
			p_alg->ToStream(s);
		}	
		s << "</Algorithms>\n";
		s << "<Nodes>\n";
		BOOST_FOREACH(const XMLNode& node, _vec_nodes)
			node.ToStream(s);
		s << "</Nodes>\n";
		s << "<Connections>\n";
		BOOST_FOREACH(const XMLConnection<Weight>& connection, _vec_con)
			connection.ToStream(s);
		s << "</Connections>\n";
		_par_run_xml.ToStream(s);
		_par_run.ToStream(s);
		s << this->ToEndTag(this->Tag()) <<"\n";

		return true;
	}

	template <class Weight>
	bool XMLSimulation<Weight>::FromStream(istream& s)
	{
		return false;
	}

	template <class Weight>
	string XMLSimulation<Weight>::Tag() const 
	{
		return "<Simulation>";
	}
}

#endif // include guard
