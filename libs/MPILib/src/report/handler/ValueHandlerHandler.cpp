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
#ifdef WIN32
#pragma warning(disable: 4267 4996)
#endif

#include <sstream>
#include <boost/bind.hpp>
#include <TGraph.h>
#include <TDirectory.h>
#include <MPILib/include/report/handler/ValueHandlerHandler.hpp>
#include <MPILib/include/BasicTypes.hpp>

namespace MPILib {
namespace report {
namespace handler {

ValueHandlerHandler::ValueHandlerHandler() {
}

ValueHandlerHandler::Event processValues(const ReportValue& value, NodeId id) {
	ValueHandlerHandler::Event ret_event;
	std::ostringstream ost;
	ost << value._name_quantity << "-" << id;
	ret_event._str = ost.str();
	ret_event._time = static_cast<float>(value._time);
	ret_event._value = static_cast<float>(value._value);

	return ret_event;
}

void ValueHandlerHandler::distributeEvent(const Event& ev) {
	auto iter = std::find(_vec_names.begin(), _vec_names.end(), ev._str);

	if (iter != _vec_names.end()) {
		auto dif = iter - _vec_names.begin();
		auto it_time = _vec_time.begin() + dif;
		it_time->push_back(ev._time);
		auto it_q = _vec_quantity.begin() + dif;
		it_q->push_back(ev._value);
	} else {
		_vec_names.push_back(ev._str);
		_vec_time.push_back(std::vector<float>(0));
		_vec_quantity.push_back(std::vector<float>(0));

		_vec_time.back().push_back(ev._time);
		_vec_quantity.back().push_back(ev._value);
	}
}

void ValueHandlerHandler::addReport(const Report& report) {
	// strip all ReportValues and store them in Events
	std::vector<Event> vec_event(report._values.size());
	std::transform(report._values.begin(), report._values.end(),
			vec_event.begin(), boost::bind(processValues, _1, report._id));

	for_each(vec_event.begin(), vec_event.end(),
			boost::bind(&ValueHandlerHandler::distributeEvent, this, _1));

}

void ValueHandlerHandler::write() {
	TDirectory* p_dir = gDirectory;
	TDirectory* p_dir_quant = p_dir->mkdir("quantities");
	p_dir_quant->cd();

	for (Index i = 0; i < _vec_names.size(); i++) {
		TGraph* p_graph = new TGraph(_vec_time[i].size(), &(_vec_time[i][0]),
				&(_vec_quantity[i][0]));
		p_graph->SetName(_vec_names[i].c_str());
		p_graph->Write();
	}
	_is_written = true;

	p_dir->cd();
}

bool ValueHandlerHandler::isWritten() const {return _is_written;}


void ValueHandlerHandler::reset() {
	_is_written = false;
	_vec_names.clear();
	_vec_time.clear();
	_vec_quantity.clear();
}

} // end namespace of handler
} // end namespace of report
} // end namespace of MPILib
