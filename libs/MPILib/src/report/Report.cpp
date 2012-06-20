/*
 * Report.cpp
 *
 *  Created on: 19.06.2012
 *      Author: david
 */

#include <MPILib/include/report/Report.hpp>

namespace MPILib {
namespace report {

Report::Report(Time time, Rate rate, NodeId id, std::string log_message) :
		_time(time), _rate(rate), _id(id), _log_message(log_message) {
}

Report::Report(Time time, Rate rate, NodeId id, algorithm::AlgorithmGrid grid,
		std::string log_message, ReportType type,
		std::vector<ReportValue> vec_values) :
		_time(time), _rate(rate), _id(id), _grid(grid), _log_message(
				log_message), _type(type), _values(vec_values) {
}

void Report::addValue(const ReportValue& value) {
	_values.push_back(value);
}



} //end namespace report
} //end namespace MPILib

