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

#ifndef _CODE_LIBS_DYNAMICLIB_ABSTRACTALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_ABSTRACTALGORITHM_INCLUDE_GUARD

#include <vector>
#include "../UtilLib/UtilLib.h"
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "AlgorithmGrid.h"
#include "NodeState.h"
#include "SimulationRunParameter.h"
#include "ReportValue.h"

using SparseImplementationLib::ConnectionProduct;
using SparseImplementationLib::AbstractSparseNode;

using UtilLib::Streamable;
using std::istream;
using std::ostream;
using std::string;
using std::vector;

namespace DynamicLib
{

	template <class WeightValue> class DynamicNode;

	//! Base class for Algorithms that run on nodes.
	//!
	//! 
	template <class WeightValue>
	class AbstractAlgorithm : public Persistant
	{
	public:

		typedef pair< AbstractSparseNode<double,WeightValue>*, WeightValue > Connection;
		typedef WeightValue WeightType;

		typedef typename AbstractSparseNode<double,WeightValue>::predecessor_iterator predecessor_iterator;
		//! standard constructor. The argument specifies the  number of state variables. This is not often used in
		//! derived classes anymore and is now set to 1 by default.
		AbstractAlgorithm(Number n = 1);

		//! copy constructor
		AbstractAlgorithm(const AbstractAlgorithm<WeightValue>&);

		//! enforce abstract base class
		virtual ~AbstractAlgorithm() = 0;
		
	
		//! Conceivably every algorithm must be able to deal with an InnerProduct	
		double InnerProduct
			(
				predecessor_iterator,
				predecessor_iterator 
			) const ;

		virtual bool IsSynchronous() const { return true; }

		virtual bool Values() const{ return false;}

		virtual vector<ReportValue> GetValues() const {return vector<ReportValue>(0);}

		//! Write out object to stream
		virtual bool ToStream(ostream&) const;

		//! Get object from stream
		virtual bool FromStream(istream&);

		//! Tag for serialization
		virtual string Tag() const;

		//! Copy internal grid, pure virtual
		//! because the representation of a state may determined by the particular algorithm
		virtual AlgorithmGrid Grid() const = 0;

		//! Copy NodeState, in general a much simpler object than the AlgorithmGrid,
		//! from which it is calculated. Usually something simple, like a rate
		virtual NodeState  State() const = 0;

		//! An algorithm saves its log messages, and must be able to produce them for a Report
		virtual string LogString() const = 0;

		//! Cloning operation, to provide each DynamicNode with its own 
		//! Algorithm instance. Clients use the naked pointer at their own risk.
		virtual AbstractAlgorithm<WeightValue>* Clone() const = 0;

		//! A complete serialization of the state of an Algorithm, so that
		//! it can be resumed from its disk representation
		virtual bool Dump(ostream&) const = 0;
		
		virtual bool Configure
		(
			const SimulationRunParameter&
		) = 0;

		//! An algorithm does not have to reimplement this function. The network will then be updated asynchronously.
		//! Where synchronous updating is imported, overloading this function allows input to be collected before
		//! any node is evolved, thereby allowing synchronous updating.
		virtual bool CollectExternalInput
		(
			predecessor_iterator,
			predecessor_iterator
		){ 
			return true;
		}


		//! Carry out the evolution step of the algorithm. If there is no overlaod of CollectExternalInput this is done asynchronously.
		virtual bool EvolveNodeState
		(
			predecessor_iterator,
			predecessor_iterator,
			Time
		) = 0;

		virtual Time CurrentTime() const = 0;

		virtual Rate CurrentRate() const = 0;

	protected:

		void ApplyBaseClassHeader(ostream& s, const string& type_name) const	{ s << "<AbstractAlgorithm type=\"" << type_name << "\">\n"; }
		void ApplyBaseClassFooter(ostream& s) const 							{ s << "</AbstractAlgorithm>\n"; }
		bool IsAbstractAlgorithmTag(const string& dummy) const					{ return (dummy == "<AbstractAlgorithm" ) ? true: false; }


		Number            StateSize           (const AlgorithmGrid&) const;
		Number&           StateSize           (AlgorithmGrid&) const;
		valarray<double>& ArrayState          (AlgorithmGrid&) const;
		valarray<double>& ArrayInterpretation (AlgorithmGrid&) const;

	private:


	}; // end of AbstractAlgorithm

	typedef AbstractAlgorithm<double> D_AbstractAlgorithm;

	template <class WeightValue>
	ostream& operator<<(ostream&, const AbstractAlgorithm<WeightValue>&);

	template <class WeightValue>
	istream& operator>>(istream&, AbstractAlgorithm<WeightValue>&);

} // end of namespace

#endif // include guard

