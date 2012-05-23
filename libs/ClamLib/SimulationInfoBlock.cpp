// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifdef WIN32
#pragma warning(disable: 4267 4996 4244)
#endif

#include "SimulationInfoBlock.h"
#include "DynamicSubNetwork.h"

using namespace ClamLib;

SimulationInfoBlock::SimulationInfoBlock():
_name(""),
_info(0),
_circ_desc(0)
{
}

SimulationInfoBlock::SimulationInfoBlock(const DynamicSubNetwork& net):
_name(net._block._name),
_net_desc(net._block._net_desc),
_info(net._block._info),
_circ_desc(net._block._circ_desc)
{
}

SimulationInfoBlock::SimulationInfoBlock
(
	const TString&						name,
	const RootLayeredNetDescription&	net_desc,
	const std::vector<CircuitInfo>&		info,
	const CircuitDescription&			circ_desc
):
_name(name),
_net_desc(net_desc),
_info(info),
_circ_desc(circ_desc)
{
}

SimulationInfoBlock::SimulationInfoBlock
(
	const SimulationInfoBlock& rhs
):
TNamed(rhs),
_name(rhs._name),
_net_desc(rhs._net_desc),
_info(rhs._info),
_circ_desc(rhs._circ_desc)
{
	this->SetName(_name);
}

TString SimulationInfoBlock::Name() const
{
	return _name;
}

const std::vector<CircuitInfo>&
	SimulationInfoBlock::InfoVector() const
{
	return _info;
}

const RootLayeredNetDescription& 
	SimulationInfoBlock::DescriptionVector() const
{
	return _net_desc;
}

const CircuitDescription&
	SimulationInfoBlock::DescriptionCircuit() const
{
	return _circ_desc;
}

