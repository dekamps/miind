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

#include <MPILib/include/SimulationRunParameter.hpp>

namespace MPILib {

SimulationRunParameter::SimulationRunParameter(
		const report::handler::AbstractReportHandler& handler, Number max_iter,
		Time t_begin, Time t_end, Time t_report, Time t_step,
		const std::string& name_log, Time t_state_report) :

		_p_handler(&handler), //
		_max_iter(max_iter), //
		_t_begin(t_begin), //
		_t_end(t_end), //
		_t_report(t_report), //
		_t_step(t_step), //
		_name_log(name_log), //
		_t_state_report((t_state_report == 0) ? t_end : t_state_report) {
}

SimulationRunParameter::SimulationRunParameter(
		const SimulationRunParameter& parameter) :
		_p_handler(parameter._p_handler), //
		_max_iter(parameter._max_iter), //
		_t_begin(parameter._t_begin), //
		_t_end(parameter._t_end), //
		_t_report(parameter._t_report), //
		_t_step(parameter._t_step), //
		_name_log(parameter._name_log), //
		_t_state_report(parameter._t_state_report) {
}


SimulationRunParameter& SimulationRunParameter::operator=(
		const SimulationRunParameter& parameter) {
	if (&parameter == this)
		return *this;

	_p_handler = parameter._p_handler;

	_max_iter = parameter._max_iter;
	_t_begin = parameter._t_begin;
	_t_end = parameter._t_end;
	_t_report = parameter._t_report;
	_t_step = parameter._t_step;
	_name_log = parameter._name_log;
	_t_state_report = parameter._t_state_report;

	return *this;
}

} // end namespace MPILib
