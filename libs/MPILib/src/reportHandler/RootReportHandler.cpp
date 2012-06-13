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
// All these warnings relate to ROOT source code
#pragma warning(disable: 4267 4305 4800 4996)
#endif 

#include <algorithm>
#include <sstream>
#include <assert.h>
#include <MPILib/include/reportHandler/RootReportHandler.hpp>
#include <MPILib/include/utilities/Exception.hpp>

#include <MPILib/include/BasicTypes.hpp>
#include <MPILib/include/reportHandler/GraphKey.hpp>

#include <TApplication.h>
#include <TFile.h>
#include <TGraph.h>
#include <TNtuple.h>

namespace MPILib {



// need one global application object
TApplication APPLICATION("application", 0, 0);


TFile* RootReportHandler::_pFile = nullptr;
TNtuple* RootReportHandler::_pTuple = nullptr;

std::vector<NodeId> RootReportHandler::_nodes(0);

ValueHandlerHandler RootReportHandler::_valueHandler;

RootReportHandler::RootReportHandler(const std::string& file_name, bool writeState) :
		AbstractReportHandler(file_name), //
		_isStateWriteMandatory(writeState) {
}

RootReportHandler::RootReportHandler(const RootReportHandler& rhs) :
		AbstractReportHandler(rhs.getStreamFileName()), //
		_isStateWriteMandatory(rhs._isStateWriteMandatory) {
	if (rhs._spCurrentRateGraph)
		throw utilities::Exception(STR_HANDLER_STALE);
}

RootReportHandler::~RootReportHandler()
// 11-07-2007: test on p_tuple by Volker Baier
{
	if (_pFile) {
		if (_pTuple)
			_pTuple->Write();
		_pFile->Close();
		delete _pFile;
		_pFile = nullptr;
	}

}

void RootReportHandler::writeReport(const Report& report) {

	if (_nrReports == 0) {

		_spCurrentRateGraph = std::unique_ptr < TGraph > (new TGraph);

		std::ostringstream stream;
		stream << "rate_" << report._id;
		_spCurrentRateGraph->SetName(stream.str().c_str());

	}

	_spCurrentRateGraph->SetPoint(_nrReports++, report._time, report._rate);

	_spCurrentStateGraph.reset();
	_spCurrentStateGraph = convertAlgorithmGridToGraph(report);

	if (report._type == STATE && isConnectedToAlgorithm()
			&& (isStateWriteMandatory()))
		_spCurrentStateGraph->Write();

	// always log ReportValue elements
	_valueHandler.AddReport(report);
}

RootReportHandler* RootReportHandler::clone() const {

	return new RootReportHandler(*this);
}

std::unique_ptr<TGraph> RootReportHandler::convertAlgorithmGridToGraph(
		const Report& report) const {

	std::vector<double> vector_of_grid_values = report._grid.ToStateVector();

	// if the Report does not contain a filled AlgorithmGrid, no Graph can be made
	if (vector_of_grid_values.size() == 0)
		return 0;

	std::vector<double> vector_of_state_interpretation =
			report._grid.ToInterpretationVector();

	std::unique_ptr<TGraph> p_state_graph { new TGraph };

	GraphKey key(report._id, report._time);
	p_state_graph->SetName(key.Name().c_str());

	assert(
			vector_of_grid_values.size() == vector_of_state_interpretation.size());

	for (unsigned int i = 0; i<vector_of_grid_values.size(); i++){
			p_state_graph->SetPoint(i,
					vector_of_state_interpretation[i],
					vector_of_grid_values[i]);
	}


	return p_state_graph;
}

bool RootReportHandler::isConnectedToAlgorithm() const {
	return (_spCurrentRateGraph != 0);
}

bool RootReportHandler::isStateWriteMandatory() const {
	return _isStateWriteMandatory;
}


void RootReportHandler::initializeHandler(const NodeId& info) {
	// Purpose: this function will be called by MPINode upon configuration.
	// no canvas are generated as it would cause lot of problems with mpi
	if (!_pFile) {
		_pFile = new TFile(this->getStreamFileName().c_str(), "RECREATE");

		if (_pFile->IsZombie())
			throw utilities::Exception(STR_ROOT_FILE_OPENED_FAILED);

		_pTuple = new TNtuple("infotuple", "node info", "id:x:y:z:f");
		_valueHandler.Reset();

	}

	writeInfoTuple(info);

}

void RootReportHandler::writeInfoTuple(const NodeId& nodeId) {
	_pTuple->Fill(static_cast<Float_t>(nodeId));
	_nodes.push_back(nodeId);
}

void RootReportHandler::detachHandler(const NodeId& nodeId) {
	// Purpose: this function will be called upon DynamicNode destruction. 
	// This works under the assumption that no isolated DynamicNodes
	// exist which are associated with an open RootReporthandler. 
	// Author: Marc de Kamps
	// Date: 26-08-2005

	removeFromNodeList(nodeId);

	if (_spCurrentRateGraph) {
		_spCurrentRateGraph->Write();
		_spCurrentRateGraph.reset();
		if (!_valueHandler.IsWritten())
			_valueHandler.Write();

	}

	if (_nodes.empty())
		finalize();
}

void RootReportHandler::removeFromNodeList(NodeId id) {
	auto iter = std::find(_nodes.begin(), _nodes.end(), id);

	if (iter == _nodes.end())
		throw utilities::Exception(
				"Can't locate NodeId during detaching handler");

	_nodes.erase(iter);
}

void RootReportHandler::finalize() {
	_pTuple->Write();
	_pFile->Close();

	if (_pFile) {
		delete _pFile;
		_pFile = nullptr;
	}

	_pTuple = nullptr;
	_nodes.clear();
}

} //end namespace

