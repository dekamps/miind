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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSENODE_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_SPARSENODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <utility>
#include <iostream>
#include "../NetLib/NetLib.h"
#include "../UtilLib/UtilLib.h"
#include "AbstractSparseNodeCode.h"
#include "BasicDefinitions.h"

using std::auto_ptr;
using std::pair;
using std::istream;
using std::ostream;
using NetLib::AbstractSquashingFunction;
using NetLib::NodeId;
using NetLib::NoSquashingFunction;
using NetLib::NetLibException;
using NetLib::SquashingFunctionFactory;
using UtilLib::Index;
using UtilLib::NumericException;

namespace SparseImplementationLib
{

	//! SparseNode is a concrete node class
	//! 
	//! SparseNode is a class for representing nodes ina network. Such nodes are
	//! collected in a SparseImplementation, which is the representation of a 
	//! SparseNetwork. So, users will typically deal with nodes indirectly via 
	//! the SparseImplementation. This is particularly true for creating networks.
	//! The connection list of a node will usually not accessed directly. In general
	//! this is the job of ScalarProduct objects. But sometimes one needs direct access
	//! to a node, for example, to read out its activity.

	template <class ActivityType_, class WeightType_>
	class SparseNode : public AbstractSparseNode<ActivityType_,WeightType_>
	{
	public:


		typedef ActivityType_ NodeValue;
		typedef WeightType_ WeightType;

		//! default constructor
		SparseNode();

		SparseNode(const SparseNode&);

		//! virtual destructor, required because of inheriting
		virtual ~SparseNode();

		SparseNode& operator=(const SparseNode&);

		//!
		void Update();

		//!
		virtual bool ToStream  (ostream& s) const;

		//!
		virtual bool FromStream(istream& s);

				
		template <class N, class W> 
		friend istream& operator>>
		(
			istream&,
			SparseNode<N,W>& 
		);


		//! Set a new squashing function object and destroy the old one
		void ExchangeSquashingFunction( const AbstractSquashingFunction* );

		//! Give the squashing function
		AbstractSquashingFunction& SquashingFunction() const;


	protected:
			

		virtual SparseNode<NodeValue, WeightType>* Address(std::ptrdiff_t);

		virtual std::ptrdiff_t Offset(AbstractSparseNode<NodeValue,WeightType>*) const;

	private:

		auto_ptr<AbstractSquashingFunction> _p_function;

	}; // end of SparseNode


	typedef SparseNode<double,double> D_SparseNode;
	// inserted by korbo
	typedef SparseNode<float,float> F_SparseNode;

	typedef std::pair<D_SparseNode*, double> D_Connection;

	typedef SparsePredecessorIterator<D_AbstractSparseNode> DWeightIterator;

} // end of SparseImplementationLib

#endif // include guard
