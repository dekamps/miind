// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifdef WIN32
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif

#include "InputCircuitCreator.h"
#include "RootConversions.h"

using namespace std;
using namespace ClamLib;

using StructnetLib::PhysicalPosition;
using DynamicLib::EXCITATORY;
using DynamicLib::INHIBITORY;
using DynamicLib::Rate;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::D_RateAlgorithm;
using DynamicLib::SpatialPosition;

InputCircuitCreator::InputCircuitCreator
(
	Rate*					input_field,
	D_DynamicNetwork*		p_dnet,
	const PhysicalPosition&	pos
):
AbstractCircuitCreator(p_dnet),
_input_field(input_field),
_p_dnet(p_dnet),
_pos(pos)
{
}

void InputCircuitCreator::AddNodes
(
	CircuitInfo* p_info
)
{
	D_RateAlgorithm alg(*_input_field);
	NetLib::NodeId id = _p_dnet->AddNode(alg, EXCITATORY);
	SpatialPosition pos;

	pos._x = static_cast<float>(_pos._position_x);
	pos._y = static_cast<float>(_pos._position_y);
	pos._z = static_cast<float>(_pos._position_z);
	_p_dnet->AssociateNodePosition(id,pos);

}