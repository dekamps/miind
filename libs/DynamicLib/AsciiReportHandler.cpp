// Copyright (c) 2005 - 2008 Marc de Kamps
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

#include "AsciiReportHandler.h"
#include <fstream>
#include "../UtilLib/UtilLib.h"
#include "DynamicLibException.h"

using namespace std;
using namespace DynamicLib;
using namespace UtilLib;

AsciiReportHandler::AsciiReportHandler
(
	const string& name
):
AbstractReportHandler(name),
_p_stream_ascii(new ofstream(name.c_str())),
_b_owning(true)
{
	// check for existence and opening of the stream
	if (! _p_stream_ascii || ! *_p_stream_ascii )
		throw DynamicLibException((STR_ASCIIHANDLER_EXCEPTION + name).c_str());
}

AsciiReportHandler::AsciiReportHandler
(
	const AsciiReportHandler& rhs
):
AbstractReportHandler(rhs.MediumName()),
_p_stream_ascii(rhs._p_stream_ascii),
_b_owning(false)
{
}

AsciiReportHandler::~AsciiReportHandler()
{
	if (_b_owning)
		delete _p_stream_ascii;

}

bool AsciiReportHandler::WriteReport	
(
		const Report& report
) 
{
	ConcreteStreamable streamable;

	Stream() << STR_REPORT			<< "\n";
	Stream() << STR_TIME			<< report._time			<< streamable.ToEndTag(STR_TIME);
	Stream() << STR_NODEID			<< report._id._id_value	<< streamable.ToEndTag(STR_NODEID);
	Stream() << STRING_NODEVALUE	<< report._rate			<< streamable.ToEndTag(STRING_NODEVALUE)
		     << "\n";
	
	Stream().precision(10);
	report._state.ToStream (Stream());     
	report._grid.ToStream  (Stream());

	Stream() << streamable.ToEndTag(STR_REPORT) << "\n";

	return true;
}


AsciiReportHandler* AsciiReportHandler::Clone() const
{
	return new AsciiReportHandler(*this);
}

ostream& AsciiReportHandler::Stream()
{
	return *_p_stream_ascii;
}

bool AsciiReportHandler::Close()
{
	return true;
}

bool AsciiReportHandler::Update()
{
	Stream().flush();
	return true;
}

void AsciiReportHandler::InitializeHandler
(
	const NodeInfo& info
)
{
	Stream()<< info._id			<< " " 
			<< info._position._x<< " "
			<< info._position._y<< " "
			<< info._position._z<< "\n";
}
