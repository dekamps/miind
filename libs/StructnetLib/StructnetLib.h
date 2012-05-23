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
#ifndef _CODE_LIBS_STRUCNET_STRUCNET_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_STRUCNET_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "../ConnectionismLib/ConnectionismLib.h"
#include "ActivityRect.h"
#include "DescriptionException.h"
#include "DenseOverlapColorLinkRelation.h"
#include "LinkRelation.h"
#include "FullyConnectedLinkRelation.h"
#include "NoLinkRelation.h"
#include "OrientedPatternCode.h"
#include "SparseLayeredNetCode.h"
#include "LayerDescription.h"
#include "PhysicalPosition.h"
#include "ReverseDenseOverlapLinkRelation.h"
#include "StructnetLibException.h"

 
using NetLib::Sigmoid;
using ConnectionismLib::BackpropTrainingVector;
using ConnectionismLib::Hebbian;
using SparseImplementationLib::D_LayeredReversibleSparseImplementation;

/*! 
 *
 * \page StructnetLib StructnetLib
 *
 * \section intro_sec Introduction
 * StructnetLib builds partly on SparseImplementationLib and it's partly stand alone. It provides
 * classes that provide a spatial interpretation for an existing network.
 *
 *
 * \section An example of a network trained by Backpropagation
 * \dontinclude StructnetLibTest.cpp
 * \skip Jogs
 * \until true
 *
 */

#endif // include guard
