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
#ifndef _CODE_LIBS_MIINDLIB_SIMULATIONPARSER_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_SIMULATIONPARSER_INCLUDE_GUARD

#include <string>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include "Simulation.h"

using std::string;
using std::istream;
using std::ostream;

namespace DynamicLib {
	template <class Connection> class AbstractAlgorithm;
}

/*! Parses a configuration file and sets up a simulation.
*
*   If the configuration file already exists, the only operation required is to call ExecuteSimulation with the file name as argument
*	If no configuration files exist, the GenerateXMLFile method can generate a template which can then be extended in an (XML) editor.
*	
*/
namespace MiindLib { 

	class SimulationParser {
	public:

		//! default constructor
		SimulationParser();

		//! Runs the simulation as specified in the configuration file
		bool ExecuteSimulation
		(
			const string&,	//!< file name of the configuration file
			bool			//!< true if batch mode is forced
		);

		//! Generate a template configuration file. For type "double" this sets up a simple Wilson-Cowan simulation, for type "pop" a simple population density simulation
		bool 
			GenerateXMLFile
			(
				const string&,				//!< name of the configuration file to be generated
				const string& = "double"	//!< type of the connection weight required for the simulation, this is "pop" for networks  using population density techniques
			);

		//! Produce svg files of the simulation results
		bool
			Analyze
			(
				const string&				//!< name of the configuration file
			);

	private:

		string			_name_simulation_result;
		vector<NodeId>	_vec_canvas_ids;

	};
}

#endif // include guard
