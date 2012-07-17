// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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
#include <MPILib/include/SimulationRunParameter.hpp>
#include <MPILib/include/utilities/FileNameGenerator.hpp>
namespace MPILib {

SimulationRunParameter::SimulationRunParameter(
		const report::handler::AbstractReportHandler& handler, Number max_iter,
		Time t_begin, Time t_end, Time t_report, Time t_step,
		const std::string& name_log, Time t_state_report) :

		_pHandler(&handler), //
		_maxIter(max_iter), //
		_tBegin(t_begin), //
		_tEnd(t_end), //
		_tReport(t_report), //
		_tStep(t_step), //
		_logFileName(name_log), //
		_tStateReport((t_state_report == 0) ? t_end : t_state_report) {
}

SimulationRunParameter::SimulationRunParameter(
		const SimulationRunParameter& parameter) :
		_pHandler(parameter._pHandler), //
		_maxIter(parameter._maxIter), //
		_tBegin(parameter._tBegin), //
		_tEnd(parameter._tEnd), //
		_tReport(parameter._tReport), //
		_tStep(parameter._tStep), //
		_logFileName(parameter._logFileName), //
		_tStateReport(parameter._tStateReport) {
}

SimulationRunParameter& SimulationRunParameter::operator=(
		const SimulationRunParameter& parameter) {
	if (&parameter == this)
		return *this;

	_pHandler = parameter._pHandler;

	_maxIter = parameter._maxIter;
	_tBegin = parameter._tBegin;
	_tEnd = parameter._tEnd;
	_tReport = parameter._tReport;
	_tStep = parameter._tStep;
	_logFileName = parameter._logFileName;
	_tStateReport = parameter._tStateReport;

	return *this;
}

Time SimulationRunParameter::getTBegin() const {
	return _tBegin;
}

Time SimulationRunParameter::getTEnd() const {
	return _tEnd;
}

Time SimulationRunParameter::getTReport() const {
	return _tReport;
}

Time SimulationRunParameter::getTStep() const {
	return _tStep;
}

Time SimulationRunParameter::getTState() const {
	return _tStateReport;
}

std::string SimulationRunParameter::getLogName() const {
	utilities::FileNameGenerator fg (_logFileName);
	return fg.getFileName();
}

const report::handler::AbstractReportHandler& SimulationRunParameter::getHandler() const {
	return *_pHandler;
}

Number SimulationRunParameter::getMaximumNumberIterations() const {
	return _maxIter;
}

} // end namespace MPILib
