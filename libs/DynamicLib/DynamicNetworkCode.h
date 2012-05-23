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
#ifndef _CODE_LIBS_DYNAMICLIB_DYNAMICNETWORKCODE_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DYNAMICNETWORKCODE_INCLUDE_GUARD

#include "DynamicNetwork.h"

namespace DynamicLib
{
	template <class WeightValue>
	DynamicNetwork<WeightValue>::DynamicNetwork
	(
	):
	_current_report_time	(0),
	_current_update_time	(0),
	_current_state_time		(0),
	_current_simulation_time(0),
	_parameter_simulation_run
	(
		InactiveReportHandler(),
		0,
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		""
	),
	_state_network			(0.0),
	_implementation			(),
	_stream_log				()
	{
	}

	template <class WeightValue>
	DynamicNetwork<WeightValue>::DynamicNetwork
	(
		const DynamicNetwork<WeightValue>& rhs
	):
	_current_report_time		(rhs._current_report_time),
	_current_update_time		(rhs._current_update_time),
	_current_state_time			(rhs._current_state_time),
	_current_simulation_time	(rhs._current_simulation_time ),
	_parameter_simulation_run	(rhs._parameter_simulation_run ),
								// default constructor for network state: the network is unconfigured and hasn't run
	_state_network				(0.0),	
								// default constructor for _stream_log
	_implementation				(rhs._implementation)
	{								
	}

	template <class WeightValue>
	DynamicNetwork<WeightValue>::~DynamicNetwork()
	{
	}

	template <class Implementation>
	NodeId DynamicNetwork<Implementation>::AddNode	
	(
		const AbstractAlgorithm<WeightType>& algorithm,
		NodeType type
	)
	{
		assert (type == EXCITATORY || type == INHIBITORY || type == NEUTRAL || type == EXCITATORY_BURST || type == INHIBITORY_BURST);
		DynamicNode<WeightType> node(algorithm,type);
		return _implementation.AddNode(node);
	}

	template<class Implementation>
	bool DynamicNetwork<Implementation>::IsDalesLawSet() const
	{
		return _implementation.IsDalesLawSet();
	}

	template<class Implementation>
	bool DynamicNetwork<Implementation>::SetDalesLaw(bool b_law)
	{
		_implementation.SetDalesLaw(b_law);
		return b_law;
	}
	template <class Implementation>
	bool DynamicNetwork<Implementation>::MakeFirstInputOfSecond
	(
		NodeId id_input,
		NodeId id_receiver,
		const WeightType& weight
	)
	{
		return 
			_implementation.MakeFirstInputOfSecond
			(
				id_input,
				id_receiver,
				weight
			);
	}

	template <class Weight>
	bool DynamicNetwork<Weight>::ToStream(ostream& s) const
	{
		s << Tag() << "\n";

		_implementation.ToStream(s);

		s << ToEndTag(Tag());

		return true;
	}

	template <class Weight>
	bool DynamicNetwork<Weight>::FromStream(istream& s)
	{
		string dummy;
		s >> dummy;

		if (dummy != this->Tag() )
			throw DynamicLibException("Unexpected DynamicNetwork tag");

		_implementation.FromStream(s);

		s >> dummy;

		if (dummy != this->ToEndTag(this->Tag()))
			throw DynamicLibException("Unexpected DynamicNetwork end tag");

		return true;
	}

	template <class Weight>
	string DynamicNetwork<Weight>::Tag() const
	{
		return STR_DYNAMIC_NETWORK_TAG;
	}

	template <class Weight>
	bool DynamicNetwork<Weight>::ConfigureSimulation
	(
		const SimulationRunParameter& parameter
	)
	{

		_current_report_time     = parameter.TReport();
		_current_update_time     = parameter.TUpdate();  
		_current_simulation_time = parameter.TBegin();


		_parameter_simulation_run = parameter;

		InitializePercentageQueue();

		InitializeLogStream(parameter.LogName());

		bool b_configure =_implementation.ConfigureSimulation(parameter);

		if (b_configure)
		{
			_stream_log << "Configuring network succeeded\n";
			_state_network.ToggleConfigured();
			return true;
		}
		else
		{
			_state_network.SetResult(CONFIGURATION_ERROR);
			return false;
		}
	}

	template <class WeightValue>
	bool DynamicNetwork<WeightValue>::Evolve()
	{
		string report; // string that collects parts of the report that need to be written into the log file

		// Only run when the network is configured
		if (_state_network.IsConfigured() )
		{
			_state_network.ToggleConfigured();
			_stream_log << "Starting simulation\n";
			_stream_log.flush();

			try
			{
				do
				{						
					do
					{
						// business as usual: keep evolving, as long as there is nothing to report
						// or to update
						UpdateSimulationTime();
						bool b_result = 
							_implementation.Evolve
							(
								CurrentSimulationTime()
							);
						// unless somthing goes wrong
						if (! b_result)
							return false;
			
					}
					while
					( 
						CurrentSimulationTime() < CurrentReportTime() &&
						CurrentSimulationTime() < CurrentUpdateTime() &&
						CurrentSimulationTime() < CurrentStateTime()
					);

					// now there is something to report or to update
					if ( CurrentSimulationTime() >= CurrentReportTime() )
					{
						// there is something to report
						CheckPercentageAndLog(CurrentSimulationTime());
						UpdateReportTime();
						report = _implementation.CollectReport(RATE);
						_stream_log << report;
					}
					// just a rate or also a state?
					if (CurrentSimulationTime() >= CurrentStateTime() )
					{
						// a rate as well as a state
						_implementation.CollectReport(STATE);
						UpdateStateTime();
					}
					// update?
					if ( CurrentReportTime() >= CurrentUpdateTime() )
					{					
						_implementation.CollectReport(UPDATE);
						UpdateUpdateTime();
					}
				}
				while ( CurrentReportTime() <  EndTime() );
				// write out the final state
				_implementation.CollectReport(STATE);
			}

			catch(IterationNumberException)
			{
				HandleIterationNumberException();
				return false;
			}

			_implementation.ClearSimulation(); 
			_stream_log << "Simulation ended, no problems noticed\n";
			_stream_log << "End time: " << CurrentSimulationTime() << "\n";
			_stream_log.close();
			return true;
		}
		else
			return false;
	}

	template <class WeightValue>
	void DynamicNetwork<WeightValue>::UpdateReportTime()
	{
		_current_report_time += _parameter_simulation_run.TReport();
	}

	template <class WeightValue>
	void DynamicNetwork<WeightValue>::InitializeLogStream(const std::string & name)
	{
		// resource will be passed on to _stream_log
		boost::shared_ptr<ostream> p_stream(new ofstream(name.c_str()));
		if (! p_stream)
			throw DynamicLibException("DynamicNetwork cannot open log file.");
		if (!_stream_log.OpenStream(p_stream))
			_stream_log << "WARNING YOU ARE TRYING TO REOPEN THIS LOG FILE\n";
	}

	template <class WeightValue>
	Time DynamicNetwork<WeightValue>::CurrentReportTime() const
	{
		return _current_report_time;
	}

	template <class WeightValue>
	Time DynamicNetwork<WeightValue>::EndTime() const
	{
		return _parameter_simulation_run.TEnd();
	}

	template <class WeightValue>
	Time DynamicNetwork<WeightValue>::CurrentUpdateTime() const
	{
		return _current_update_time;
	}

	template <class WeightValue>
	Time DynamicNetwork<WeightValue>::CurrentStateTime() const
	{
		return _current_state_time;
	}

	template <class WeightValue>
	void DynamicNetwork<WeightValue>::UpdateStateTime()
	{
		_current_state_time += _parameter_simulation_run.TState();
	}

	template <class WeightValue>
	void DynamicNetwork<WeightValue>::UpdateSimulationTime() 
	{
		_current_simulation_time += _parameter_simulation_run.TStep();
	}

	template <class WeightValue>
	Time DynamicNetwork<WeightValue>::CurrentSimulationTime() const
	{
		return _current_simulation_time;
	}

	template <class WeightValue>
	void DynamicNetwork<WeightValue>::UpdateUpdateTime()
	{
		_current_update_time += _parameter_simulation_run.TUpdate();
	}

	template <class WeightValue>
	NodeState DynamicNetwork<WeightValue>::State(NodeId id) const
	{
		return _implementation.State(id);
	}

	template <class Implementation>
		typename AbstractSparseNode<double,typename Implementation::WeightType_>::predecessor_iterator
		 DynamicNetwork<Implementation>::begin(NodeId id)
	{
		return _implementation.begin(id); 
	}

	template <class Implementation>
		typename AbstractSparseNode<double,typename Implementation::WeightType_>::predecessor_iterator
		DynamicNetwork<Implementation>::end(NodeId id)
	{
		return _implementation.end(id);
	}

	template <class Implementation>
	Number DynamicNetwork<Implementation>::NumberOfNodes() const
	{
		return _implementation.NumberOfNodes();
	}

	template <class Implementation>
	bool DynamicNetwork<Implementation>::AssociateNodePosition
	(
		NodeId id,
		const SpatialPosition& position
	)
	{
		return _implementation.AssociateNodePosition(id, position);
	}

	template <class Implementation>
	void DynamicNetwork<Implementation>::HandleIterationNumberException()
	{
		// simulation over: flush log buffer
		_stream_log << "NUMBER OF ITERATIONS EXCEEDED\n";
		_state_network.SetResult(NUMBER_ITERATIONS_ERROR);
		_stream_log.flush();
		_stream_log.close();
	}

	template <class Implementation>
	void DynamicNetwork<Implementation>::CheckPercentageAndLog(Time time)
	{
			if ( CurrentSimulationTime()  >= PercentageTimeTop() ){
				float perct = CurrentPercentage();
				_stream_log << perct << "% achieved\n";
				_stream_log.flush();
				AdaptPercentage();
			}
	}
	template <class Implementation>
	void DynamicNetwork<Implementation>::InitializePercentageQueue()
	{
		Time t_begin = _parameter_simulation_run.TBegin();
		Time t_end   = _parameter_simulation_run.TEnd();

		Time t_fract_small = (t_end - t_begin)/N_FRACT_PERCENTAGE_SMALL;
		Time t_fract_big   = (t_end - t_begin)/N_FRACT_PERCENTAGE_BIG;

		pair<float,Time> perct_time;

		for (int i = N_FRACT_PERCENTAGE_BIG; i >= 1; i-- )
		{
			perct_time.first  = i*100.0F/static_cast<float>(N_FRACT_PERCENTAGE_BIG);
			perct_time.second = t_begin + i*t_fract_big;
			_stack_percentage.push(perct_time);
		}

		for (int j = N_PERCENTAGE_SMALL; j >= 1; j--)
		{
			perct_time.first  = j*100.0F/static_cast<float>(N_FRACT_PERCENTAGE_SMALL);
			perct_time.second = t_begin + j*t_fract_small;
			_stack_percentage.push(perct_time);
		}

		perct_time.first  = 0.0F;
		perct_time.second = t_begin;
		_stack_percentage.push(perct_time);

	}

	template <class Implementation>
	float DynamicNetwork<Implementation>::CurrentPercentage() const
	{
		return _stack_percentage.top().first;
	}

	template <class Implementation>
	void DynamicNetwork<Implementation>::AdaptPercentage()
	{
		return _stack_percentage.pop();
	}

	template <class Implementation>
	Time DynamicNetwork<Implementation>::PercentageTimeTop() const
	{
		return _stack_percentage.top().second;
	}
	
	template <class Implementation>
	NodeIterator<DynamicNode<typename Implementation::WeightType_> >DynamicNetwork<Implementation>::begin()
	{ 
		return  NodeIterator<DynamicNode<typename Implementation::WeightType_> >(&(*_implementation.begin()));
	}

	template <class WeightValue>
	bool DynamicNetwork<WeightValue>::GetPosition
	(
		NodeId id,
		SpatialPosition* p_pos
	)
	{
		return _implementation.GetPosition(id,p_pos);
	}

} // end of DynamicLib

#endif // include guard
