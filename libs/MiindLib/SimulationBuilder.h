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
#ifndef _CODE_LIBS_MIINDLIB_SIMULATIONBUILDER_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_SIMULATIONBUILDER_INCLUDE_GUARD

#include <string>
#include <iostream>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include "../PopulistLib/PopulistLib.h"

#include "../DynamicLib/DynamicLib.h"
#include "XMLNodes.h"
#include "XMLConnection.h"
#include "XMLRunParameter.h"

using DynamicLib::SimulationRunParameter;
using PopulistLib::PopulationConnection;
using std::istream;
using std::string;

namespace MiindLib {

	//! This class builds and instantiates a complete simulation from its XML representation. It is also capable of generating
	//! template examples for different weight types.

	//! SimulationBuilder can be constructed with a stream argument. It is assumed that the stream contains a valid XML simulation structure.
	
	template <class Connection>
	class SimulationBuilder {
	public:

		typedef boost::shared_ptr<XMLRunParameter> xml_run_par_pointer;
		typedef boost::shared_ptr<SimulationRunParameter> run_par_pointer;
		typedef boost::shared_ptr<DynamicLib::AbstractAlgorithm<Connection> > algorithm_pointer;
		typedef vector<algorithm_pointer> algorithm_vector;
		typedef DynamicLib::DynamicNetwork< DynamicLib::DynamicNetworkImplementation<Connection> > Network;
		typedef boost::shared_ptr<Network> network_pointer;
		typedef DynamicLib::DynamicNode<Connection> DNode;

		//! Build and run a simulation from a stream. The XML file may contain an instruction to show simulation results, online, i.e. during
		//! simulation. The batch argument can overrule this setting and force batch mode.
		bool BuildSimulation
		(
			istream&,		//! file stream for simulation to build from
			bool			//! true if running in batch mode
		);

		bool GenerateExample(const string& ) const;

		//! A list of all nodes added to the canvas by the user. In figure processing mode this determines
		//! for which nodes figures are produced
		vector<NodeId> CanvasIds() const;

		//! Name of the simulation results file
		string SimulationFileName() const;

	private:

		bool ParseRunParameter	(istream&, bool);
		bool ParseAlgorithms	(istream&);
		bool ParseNodes			(istream&);
		bool ParseConnections	(istream&);

		network_pointer ConstructNetwork();

		bool NetToFile(network_pointer);

		typedef pair<NodeId,string> id_label;
		void AddNodes		(network_pointer);
		void AddConnections	(network_pointer);
		void AddCanvasNodes ();

		boost::shared_ptr<DynamicLib::RootReportHandler>		_p_handler;
		xml_run_par_pointer					_p_run_xml;
		run_par_pointer						_p_run_par;
		algorithm_vector					_vector_algorithm;
		vector<XMLNode >					_vec_nodes;
		vector<XMLConnection<Connection> >	_vec_connections;
		vector<id_label>					_vec_id_label;
		vector<NodeId>						_vec_canvas_id;
		string								_name_root;
	};

}

// 
namespace std {

	template <class Weight>
	bool operator==( boost::shared_ptr< DynamicLib::AbstractAlgorithm<Weight> >& alg1, const string& alg2)
	{
		return (alg1->GetName() == alg2 );
	}

	inline bool operator==( const std::pair<NodeId,string>& p , const string& s)
	{
		return (p.second == s);
	}
}




#endif // include guard
