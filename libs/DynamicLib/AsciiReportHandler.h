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
#ifndef _CODE_LIBS_DYNAMICLIB_ASCIIREPORTHANDLER_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_ASCIIREPORTHANDLER_INCLUDE_GUARD

#include "AbstractReportHandler.h"

namespace DynamicLib
{

	//! A report handler that produces simulation results as MATLAB files
	//!
	//! At the moment the simulation results are dumped in a massive text file, so some processing is required
	//! to generate actual MATLAB files. For high volume data generation the RootReportHandler or RootHighThroughputHandler 
	//! are recommended

	class AsciiReportHandler : public AbstractReportHandler
	{
	public:

		//! Name of the simulation result file
		AsciiReportHandler(const string&);

		AsciiReportHandler(const AsciiReportHandler&);

		virtual ~AsciiReportHandler();

		virtual bool 
			WriteReport
			(
				const Report&
			);

		virtual bool Update();

		virtual AsciiReportHandler* Clone() const;

		virtual bool Close();

		virtual void InitializeHandler
			(
				const NodeInfo&
			);

		virtual void DetachHandler
			(
				const NodeInfo&
			){}

	private:

		ostream& Stream();

		ostream* _p_stream_ascii;
		bool     _b_owning;
	};

} // end of DynamicLib

#endif // include guard
