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
#ifndef _CODE_LIBS_MIINDLIB_SIMULATIONBUILDERCODE_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_SIMULATIONBUILDERCODE_INCLUDE_GUARD

#include "SimulationBuilder.h"
#include "../DynamicLib/DynamicLib.h"
#include "XMLNodes.h"
#include "XMLConnectionCode.h"

namespace MiindLib {

	template <class Weight>
	bool SimulationBuilder<Weight>::ParseRunParameter(istream& s, bool b_batch){
		_p_run_xml = boost::shared_ptr<XMLRunParameter>(new XMLRunParameter(s));
		_name_root = _p_run_xml->SimulationName();
		_p_handler = 
			boost::shared_ptr<RootReportHandler>
			(
				new RootReportHandler
				(
					_name_root,
					b_batch ? false : _p_run_xml->OnScreen(),
					b_batch ? true  : _p_run_xml->InFile(),
					_p_run_xml->ParCanvas()
				)
			);
		_p_run_par = run_par_pointer(new SimulationRunParameter(*_p_handler,s));
		return true;
	}

	template <class Weight>
	bool SimulationBuilder<Weight>::ParseAlgorithms(istream& s){

		string dummy;
		s >> dummy;

		if (dummy != "<Algorithms>")
			return false;

		DynamicLib::AlgorithmBuilder<Weight> builder;

		algorithm_pointer p_alg;
		string offending_string;

		try {
			while(1){
				p_alg= builder.Build(s);
				if ( p_alg->IsValid() )
					_vector_algorithm.push_back(p_alg);
				else {
					offending_string = p_alg->BuildFailureReason();
					break;
				}
			}
		}
		catch(GeneralException& ex)
		{
			cerr << ex.Description();
			return false;
		}

		if (offending_string == "</Algorithms>" )
			return true;
		else
		{
			cerr << "Offending algorithm string: " << offending_string << endl;
			return false;
		}

	}

	template <class Weight>
	bool SimulationBuilder<Weight>::ParseNodes(istream& s){
		string dummy;
		s >> dummy;

		if (dummy != "<Nodes>")
			return false;
		//absorb newline char
		getline(s,dummy);

		string offending_string;
		try {
			while(1){
				XMLNode node(s);
				if (node.IsValid() )
					_vec_nodes.push_back(node);
				else
				{
					offending_string = node.BuildFailureReason();
					break;
				}
			}
		}
		catch(GeneralException& ex)
		{
			cerr << ex.Description();
			return false;
		}

		// test whether reason for invalidity is valid
		if ( offending_string == "/Nodes" )
			return true;
		else {
			cerr << "Offending node string: " << offending_string << endl;
			return false;
		}
	}

	template <class Weight>
	bool SimulationBuilder<Weight>::ParseConnections(istream& s){
		string dummy;
		s >> dummy;
		if (dummy != "<Connections>")
			return false;
		//absorb newline char
		getline(s,dummy);

		string offending_string;

		try {
			while(1){
				XMLConnection<Weight> connection(s);
				if (connection.IsValid() )
					_vec_connections.push_back(connection);
				else
				{
					offending_string = connection.BuildFailureReason();
					break;
				}
			}
		}
		catch(GeneralException& exc)
		{
			cerr << exc.Description();
		}

		if ( offending_string != "/Connections" ){
			cerr << "Offending connection string" << offending_string << endl;
			return false;
		}

		return true;
	}

	template <class Weight>
	bool SimulationBuilder<Weight>::NetToFile(network_pointer p_net)
	{
		ostringstream str;
		str << this->_p_run_xml->SimulationName() << ".net";

		ofstream ofst(str.str().c_str());
		if (! ofst)
			return false;
		p_net->ToStream(ofst);
		return true;
	}

	template <class Weight>
	bool SimulationBuilder<Weight>::BuildSimulation
	(
		istream&	s,
		bool		b_batch
	){

		if (! ParseAlgorithms(s) )
			return false;
		if (! ParseNodes(s) )
			return false;
		if (! ParseConnections(s) )
			return false;
		if (! ParseRunParameter(s,b_batch) )
			return false;

		boost::shared_ptr<Network> 
			p_net = ConstructNetwork();

		if (this->_p_run_xml->NetToFile() )
			NetToFile(p_net);
		
		// the canvas nodes must be added before network configuration,
		// it can be done once the network is constructed but before Configuration

		this->AddCanvasNodes();

		if (! p_net->ConfigureSimulation(*_p_run_par) )
			return false;

		p_net->Evolve();

		return true;
	}

	template <class Weight>
	void SimulationBuilder<Weight>::AddConnections	(network_pointer p_net)
	{
		BOOST_FOREACH(XMLConnection<Weight>& connection, _vec_connections)
		{
		
			string in  = connection.In();
			vector<id_label>::const_iterator iter_in =
				std::find(_vec_id_label.begin(),_vec_id_label.end(),in);
			if (iter_in == _vec_id_label.end() )
				throw MiindLibException("Input connection not found");

			string out = connection.Out();
			vector<id_label>::const_iterator iter_out =
				std::find(_vec_id_label.begin(),_vec_id_label.end(),out);
			if( iter_out == _vec_id_label.end() )
				throw MiindLibException("output connection not found");

			p_net->MakeFirstInputOfSecond(iter_in->first,iter_out->first,connection.Connection());
		}
	}


	template <class Weight>
	void SimulationBuilder<Weight>::AddNodes(network_pointer p_net)
	{
		BOOST_FOREACH(XMLNode& node, _vec_nodes)
		{
			// first find the algorithm that belongs to the network
			typename algorithm_vector::iterator it_alg = find(_vector_algorithm.begin(), _vector_algorithm.end(), node.AlgorithmName());
			if (it_alg != _vector_algorithm.end() ){
				id_label label;
				DynamicLib::NodeType type = FromValueToType(node.TypeName());
				NodeId id = p_net->AddNode(**it_alg,type);
				label.first = id;
				label.second = node.GetName();
				_vec_id_label.push_back(label);
			}
			else 
				throw MiindLibException("Can't locate algorithm");		}
	}

	template <class Weight>
	typename SimulationBuilder<Weight>::network_pointer SimulationBuilder<Weight>::ConstructNetwork()
	{
		network_pointer p_return = network_pointer( new DynamicNetwork<DynamicNetworkImplementation<Weight> >);
	
		AddNodes(p_return);
		AddConnections(p_return);

		return p_return;
	}

	template <class Weight>
	void SimulationBuilder<Weight>::AddCanvasNodes()
	{
	  //  Disabled BOOST_FOREACH, unpredictable behaviour. MdK: 4-4-2012
	  vector<string> vec_canvas_names = _p_run_xml->CanvasNames();
	  vector<string>::iterator iter;
	  for (iter = vec_canvas_names.begin(); iter != vec_canvas_names.end(); iter++){
	    string name = *iter;
	    vector<pair<NodeId,string> >::iterator iter_label = find(_vec_id_label.begin(), _vec_id_label.end(),name);
	     if (iter_label != _vec_id_label.end() ){
			 NodeId id = iter_label->first;
			 _p_handler->AddNodeToCanvas(id);
			 _vec_canvas_id.push_back(id);
	     } else
		  throw MiindLibException("Canvas node name is not defined in list of nodes");
	  }
	}

	template <class Weight>
	vector<NodeId> SimulationBuilder<Weight>::CanvasIds() const
	{
		return _vec_canvas_id;
	}

	template <class Weight>
	string SimulationBuilder<Weight>::SimulationFileName() const
	{
		return _name_root;
	}
}
#endif // include guard
