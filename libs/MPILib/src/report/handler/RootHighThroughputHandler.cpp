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
#include <TClass.h>
#include <TFile.h>
#include <TGraph.h>
#include <TVectorT.h>
#include <TVectorD.h>
#include <TTree.h>
#include <TNTuple.h>
#include <MPILib/include/utilities/Exception.hpp>
#include <MPILib/include/report/handler/RootHighThroughputHandler.hpp>
#include <iostream>
#include <MPILib/include/utilities/MPIProxy.hpp>
#include <cassert>

namespace MPILib {
namespace report {
namespace handler {

bool RootHighThroughputHandler::_isRecording = false;
bool RootHighThroughputHandler::_generateNodeGraphs = false;

std::map<int, double> RootHighThroughputHandler::_mData;

TTree* RootHighThroughputHandler::_pTree = nullptr;
TFile* RootHighThroughputHandler::_pFile = nullptr;
TVectorD* RootHighThroughputHandler::_pArray = nullptr;

RootHighThroughputHandler::RootHighThroughputHandler(
		const std::string& fileName,  bool generateGraphs) :
		AbstractReportHandler(fileName) {

	this->_generateNodeGraphs = generateGraphs;
}

RootHighThroughputHandler::RootHighThroughputHandler(
		const RootHighThroughputHandler& rhs) :
		AbstractReportHandler(rhs) {
}

void RootHighThroughputHandler::detachHandler(const NodeId&) {
	// write after the last array
	if (_pArray) {
		_pTree->Write();

		// if so desired, create node rate graphs
		// note that the file is closed by the first node so that the test for its existence is necessary
		if (_generateNodeGraphs && _pFile)
			generateNodeGraphs(_pFile->GetName());

	}

	// restore all static definitions
	delete _pArray;
	_pArray = nullptr;
	_isRecording = false;
	_generateNodeGraphs = false;
	_mData.clear();
	_pTree = nullptr;

	if (_pFile) {
		// clean up
		_pFile->Close();
		delete _pArray;
		delete _pFile;
		_pArray = nullptr;
		_pFile = nullptr;
	}
}

RootHighThroughputHandler* RootHighThroughputHandler::clone() const {
	return new RootHighThroughputHandler(*this);
}

void RootHighThroughputHandler::initializeHandler(const NodeId&) {
	if (!_pFile)
		_pFile = new TFile(this->getRootOutputFileName().c_str(), "RECREATE");
	if (_pFile->IsZombie())
		throw utilities::Exception("Couldn't open root file");

}

void RootHighThroughputHandler::writeReport(const Report& report) {

	if (!_isRecording) {
		utilities::MPIProxy mpiProxy;

		// Store the global node ids in the array
		TArrayI nodeIds(report._nrNodes);
		for (Index i = 0; i < report._nrNodes; i++) {
			nodeIds[i] = mpiProxy.getSize() * i + mpiProxy.getRank();
		}
		_pFile->WriteObject(&nodeIds, "GlobalNodeIds");

		//prepare the tree for the activations
		_pTree = new TTree("Activations", "Times slices");
		_pTree->Branch("slices", "TVectorT<double>", &_pArray, 32000, 0);
		_pArray = new TVectorD(report._nrNodes + 1);
		_isRecording = true;
	}

	if (_isRecording) {
		//store the node in a map one below the max number of nodes
		if(_mData.count(report._id)){
			throw utilities::Exception("The report for this node is already stored");
		}

		if (_mData.size() < (report._nrNodes - 1)) {
			_mData[report._id] = report._rate;
		}
		// write the map to the file
		else {
			_mData[report._id] = report._rate;

			(*_pArray)[0] = report._time;
			//They have to have the same size
			assert(_mData.size() == report._nrNodes);

			//store the data
			Index i = 1;
			for (auto& it : _mData) {
				(*_pArray)[i] = it.second;
				i++;
			}
			//clear the map
			_mData.clear();
			_pTree->Fill();
		}
	}
}
RootHighThroughputHandler::~RootHighThroughputHandler() {
}

void RootHighThroughputHandler::generateNodeGraphs(const char* fileName) {
	const std::string file_name(fileName);
	//close to flush buffers and reopen to append
	_pFile->Close();
	delete _pFile;
	//closing the file has invalidated (destroyed) the tree
	_pTree = nullptr;
	delete _pArray;
	_pArray = nullptr;

	_pFile = new TFile(file_name.c_str(), "UPDATE");

	Number number_of_nodes = -1;
	Number number_of_slices = -1;
	std::vector<double> vec_times;

	collectGraphInformation(vec_times, number_of_nodes, number_of_slices);

	storeRateGraphs(vec_times, number_of_nodes, number_of_slices);

	_pFile->Close(); // the file object does not exist anymore, don't delete the pointer
	_pFile = nullptr; // just set it to nullptr
	delete _pArray;
	_pArray = nullptr;

}

void RootHighThroughputHandler::storeRateGraphs(
		const std::vector<double>& vecTime, Number nrNodes, Number nrSlices) {

	auto nodeIds = std::unique_ptr<TArrayI>((TArrayI*)_pFile->Get("GlobalNodeIds"));
	if (!nodeIds) {
		throw utilities::Exception("No nodeIds found in the root file");
	}

	TBranch* p_branch = _pTree->GetBranch("slices");
	p_branch->SetAddress(&_pArray); //address of pointer!

	for (Index node = 0; node < nrNodes; node++) {
		Index id_node = node + 1;

		std::vector<double> vec_rate;
		for (Index slice = 0; slice < nrSlices; slice++) {
			p_branch->GetEvent(slice);
			vec_rate.push_back((*_pArray)[id_node]);
		}
		if (vec_rate.size() != vecTime.size())
			throw utilities::Exception("Inconsistency between times and rate");

		// Here we have all we need to create a Graph
		TGraph* p_graph = new TGraph(vec_rate.size(), &(vecTime[0]),
				&(vec_rate[0]));
		std::ostringstream stgr;
		stgr << "rate_" << (*nodeIds)[node];

		p_graph->SetName(stgr.str().c_str());
		p_graph->Write();
		delete p_graph;
	}
}

void RootHighThroughputHandler::collectGraphInformation(
		std::vector<double>& vecTime, Number& nrNodes, Number& nrSlices) {
	if (_pTree)
		throw utilities::Exception("There is a TTree that shouldn't be there");

	_pTree = (TTree*) _pFile->Get("Activations");
	if (!_pTree)
		throw utilities::Exception("No valid TTree");

	nrSlices = static_cast<Number>(_pTree->GetEntries());

	TBranch* p_branch = _pTree->GetBranch("slices");
	p_branch->SetAddress(&_pArray); //address of pointer!

	for (Index i = 0; i < nrSlices; i++) {
		p_branch->GetEvent(i);
		vecTime.push_back((*_pArray)[0]);
	}
	//ignore the time in the first entry
	nrNodes = _pArray->GetNoElements() - 1;
}

} // end namespace of handler
} // end namespace of report
} // end namespace of MPILib
