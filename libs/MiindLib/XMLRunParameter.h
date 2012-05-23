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
#ifndef _CODE_LIBS_MIINDLIB_XMLRUNPARAMETER_INCLUDE_GUARD
#define _CODE_LIBS_MIINDLIB_XMLRUNPARAMETER_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "../UtilLib/UtilLib.h"

using DynamicLib::CanvasParameter;
using DynamicLib::DEFAULT_CANVAS;
using UtilLib::Streamable;

namespace MiindLib {

	class XMLRunParameter : public Streamable {
	public:

		XMLRunParameter(istream&);

		XMLRunParameter
		(
			const string&,								//!< name of the simulation
			bool,										//!< results shown on canvas?
			bool,										//!< smulation results stored in file
			bool,										//!< write out the network?
			const vector<string>& = vector<string>(0),	//!< vector of names to canvas
			const CanvasParameter& = DEFAULT_CANVAS	//!< specify the dimensions of histograms when a canvas is shown
		);

		virtual ~XMLRunParameter();

		virtual string Tag() const;

		virtual bool ToStream(ostream&) const;

		virtual bool FromStream(istream&);

		void AddNodeToCanvas(const string& name){ _vector_canvas_nodes.push_back(name); }

		bool OnScreen() const { return _b_canvas; }

		bool InFile() const { return _b_file; }

		bool NetToFile() const { return _b_write_net; }

		string SimulationName() const { return _handler_name; }

		CanvasParameter ParCanvas() const { return _par_canvas; }

		bool WriteNet() const { return _b_write_net; }

		vector<string> CanvasNames() const { return _vector_canvas_nodes; }

	private:

		string	GetHandlerName	(istream&)	const;
		bool	GetCanvasValue	(istream&)	const;
		bool	GetWriteState	(istream&)	const;
		bool	GetWriteNet		(istream&)	const;

		CanvasParameter GetCanvasParameter(istream&) const;

		void	DecodeCanvasVector(istream&);
		

		string			_handler_name;
		bool			_b_canvas;
		bool			_b_file;
		bool			_b_write_net;
		CanvasParameter _par_canvas;

		vector<string> _vector_canvas_nodes;
	};

}
#endif // include guard
