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
#ifndef _CODE_LIBS_DYNAMICLIB_DYNAMICNODECODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DYNAMICNODECODE_INCLUDE_GUARD

#include "DynamicNode.h"
#include "InactiveReportHandler.h"
#include "IterationNumberException.h"
#include "RateAlgorithmCode.h"
#include "Report.h"
#include "BasicDefinitions.h"
	
namespace DynamicLib
{
	template <class Weight>
	DynamicNode<Weight>::DynamicNode():
	AbstractSparseNode<double,Weight>(NodeId(0)),
	_number_iterations(0),
	_maximum_iterations(0),
	_type(NEUTRAL),
	_state(NodeState(vector<double>(0))),
	_p_algorithm(new RateAlgorithm<Weight>(0.0)),
	_p_handler(new InactiveReportHandler)
	{
		this->SetValue(0);
	}

	template <class Weight>
	DynamicNode<Weight>::DynamicNode
	(
		const AbstractAlgorithm<Weight>& algorithm,
		NodeType type
	):
	AbstractSparseNode<double,Weight>(NodeId(0)),
	_number_iterations(0),
	_maximum_iterations(0),
	_type(type),
	_state(algorithm.State()),
	_p_algorithm(algorithm.Clone()),
	_p_handler(new InactiveReportHandler)
	{
		this->SetValue(0);
	}

	template <class Weight>
	DynamicNode<Weight>::~DynamicNode()
	{
	}

	template <class Weight>
	DynamicNode<Weight>::DynamicNode
	(
		const DynamicNode<Weight>& rhs
	):
	AbstractSparseNode<double,Weight>(rhs),
	_number_iterations	(rhs._number_iterations	),
	_maximum_iterations	(rhs._maximum_iterations),
	_type				(rhs._type),
	_state				(rhs._state),
	_info				(rhs._info),
	_p_algorithm		(rhs._p_algorithm->Clone()),
	_p_handler			(rhs._p_handler->Clone())
	{
		this->SetValue(0);
	}

	template <class Weight>
	DynamicNode<Weight>& DynamicNode<Weight>::operator=
	(
		const DynamicNode<Weight>& rhs
	)
	{
		if ( &rhs == this )
			return *this;
		else
		{

			AbstractSparseNode<double,Weight>::operator=(rhs);

			_p_algorithm		= auto_ptr<AbstractAlgorithm<Weight> >( rhs._p_algorithm->Clone() );
			_p_handler			= auto_ptr<AbstractReportHandler>( rhs._p_handler->Clone() );
			_number_iterations	= rhs._number_iterations;
			_maximum_iterations	= rhs._maximum_iterations;
			_state				= rhs._state;
			_type				= rhs._type;
			_info				= rhs._info;

			return *this;
		}
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::CollectExternalInput(){

		_p_algorithm->CollectExternalInput
		(
			this->begin(),
			this->end()
		);

		return true;
	}

	template <class WeightValue>
	Time DynamicNode<WeightValue>::Evolve(Time time_to_achieve)
	{	

		
		while (_p_algorithm->CurrentTime() < time_to_achieve)
		{
			++_number_iterations;

		_p_algorithm->EvolveNodeState
				(
					this->begin(),
					this->end(),
					time_to_achieve
				);

		}

		// update state
		this->SetValue(_p_algorithm->CurrentRate());
		return _p_algorithm->CurrentTime();
	}

	template <class Weight>
	NodeType DynamicNode<Weight>::Type() const
	{
		return _type;
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::ConfigureSimulationRun
	(
		const SimulationRunParameter& parameter_simulation_run
	)
	{
		_maximum_iterations = parameter_simulation_run.MaximumNumberIterations();
		bool b_return = _p_algorithm->Configure(parameter_simulation_run);

		// Add this line or other nodes will not get a proper input at the first simulation step!
		this->SetValue(_p_algorithm->CurrentRate());

		_p_handler = auto_ptr<AbstractReportHandler>( parameter_simulation_run.Handler().Clone() );

		// by this time, the Id of a Node should be known
		// this can't be handled by the constructor because it is an implementation (i.e. a network)  property

		_info._id = this->MyNodeId();
		_p_handler->InitializeHandler(_info);

		return b_return;
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::ClearSimulation()
	{
		_p_handler->DetachHandler(_info);
		return true;
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::UpdateHandler()
	{
		return _p_handler->Update();
	}

	template <class WeightValue>
	Time DynamicNode<WeightValue>::CurrentTime() const
	{
		return _p_algorithm->CurrentTime();
	}

	template <class WeightValue>
	Number DynamicNode<WeightValue>::NumberMaximumIterations() const
	{
		return _maximum_iterations;
	}

	template <class WeightValue>
	string DynamicNode<WeightValue>::ReportAll(ReportType type) const
	{
		string string_return = _p_algorithm->LogString();

		vector<ReportValue> vec_values;
		if (_p_algorithm->Values())
			vec_values = _p_algorithm->GetValues();

		if ( type == RATE || type == STATE )
		{
				Report report
				(
					CurrentTime(),
					this->GetValue(),
					this->MyNodeId(),
					_state,
					_p_algorithm->Grid(),
					string_return,
					type,
					vec_values
				);

				_p_handler->WriteReport(report);
			}

		if ( type == UPDATE )
			_p_handler->Update();

		return string_return;
	}

	template <class WeightValue>
	template <typename StateType> StateType DynamicNode<WeightValue>::State() const
	{
		// for the time being, a node state is just its activity
		vector<double> vector_state( 1,this->GetValue() );
		StateType state_return(vector_state);
		return state_return;
	}

	template <class WeightValue>
	DynamicNode<WeightValue>* DynamicNode<WeightValue>::Address(std::ptrdiff_t offset)
	{
		return this + offset;
	}

	template <class WeightValue>
	std::ptrdiff_t DynamicNode<WeightValue>::Offset(AbstractSparseNode<double,WeightValue>* p_abstract_node) const
	{
		DynamicNode<WeightValue>* p_node = dynamic_cast<DynamicNode<WeightValue>*>(p_abstract_node);
		if (! p_node)
			throw DynamicLibException(OFFSET_ERROR);
		return p_node - this;
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::FromStream(istream& s)
	{
		string dummy;
		s >> dummy;

		if (dummy != this->Tag())
			throw DynamicLibException("DynamicNode tag expected");
		AbstractSparseNode<double,WeightValue>::FromStream(s);

		// AbsorbAlgorithm also absorbs the end tag of the node itself!
		_p_algorithm = AbsorbAlgorithm(s);

		return true;
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::ToStream(ostream& s) const
	{
		s << this->Tag() << endl;
		AbstractSparseNode<double,WeightValue>::ToStream(s);
		_p_algorithm->ToStream(s);
		s << this->ToEndTag(this->Tag()) << "\n";

		return true;
	}

	template <class WeightValue>
	string DynamicNode<WeightValue>::Tag() const
	{
		return STR_DYN_TAG;
	}

	template <class WeightValue>
	void DynamicNode<WeightValue>::AssociatePosition
	(
		const SpatialPosition& position
	)
	{
		_info._position = position;
	}

	template <class WeightValue>
	bool DynamicNode<WeightValue>::GetPosition
	(
		SpatialPosition* p_pos
	) const
	{
		if (_info._position == INVALID_POSITION)
		{
			*p_pos = INVALID_POSITION;
			return false;
		}
		else
		{
			*p_pos = _info._position;
			return true;
		}
	}

	template <class WeightValue>
	auto_ptr<AbstractAlgorithm<WeightValue> >
		DynamicNode<WeightValue>::CloneAlgorithm() const
	{
		return auto_ptr<AbstractAlgorithm<WeightValue> >(_p_algorithm->Clone());
	}

	template <class WeightValue>
	void DynamicNode<WeightValue>::SetNodeName(const std::string& name)
	{
			_name = name;
	}

	template <class WeightValue>
	string DynamicNode<WeightValue>::GetNodeName() const
	{
		return _name;
	}
	
	template <class WeightValue>
	auto_ptr<AbstractAlgorithm<WeightValue> > 
		DynamicNode<WeightValue>::AbsorbAlgorithm(istream& s){
			// for the moment do not try to recreate the algorithm, just absorb its description
			auto_ptr<AbstractAlgorithm<WeightValue> > p(0);
			string dummy;
			s >> dummy; // this now presumably contains an algorithm tag

			// there are still algorithms that are not represented, in the long run this must be phased out
			if ( dummy == this->ToEndTag(this->Tag()) )
				return p;

			if ( dummy.find("Algorithm") == string::npos)
				throw DynamicLibException("Algorithm tag found that does not contain the strong 'Algorithm'");

			string dummy_end = this->ToEndTag(dummy);
			do {
				s >> dummy;
			}
			while ( dummy != dummy_end);

			s >> dummy;
			if (dummy != this->ToEndTag(this->Tag()) )
				throw DynamicLibException("DynamicNode end tag expected");

			return p;
	}

} // end of DynamicLib

#endif // include guard
