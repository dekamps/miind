// Copyright (c) 2005 - 2012 Marc de Kamps
//                      2012 David-Matthias Sichau
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

#include <algorithm>
#include <sstream>
#include <assert.h>
#include <MPILib/include/report/handler/CsvReportHandler.hpp>
#include <MPILib/include/utilities/Exception.hpp>
#include <MPILib/include/StringDefinitions.hpp>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

namespace MPILib {
namespace report {
namespace handler {

std::ofstream CsvReportHandler::_pFile;

std::vector<NodeId> CsvReportHandler::_nodes(0);

ValueHandlerHandler CsvReportHandler::_valueHandler;

CsvReportHandler::CsvReportHandler(const std::string& file_name, 
                                   bool writeState):
        AbstractReportHandler(file_name), _isStateWriteMandatory(writeState)
        {
            // _pfile.open(file_name << ".txt")
        }

CsvReportHandler::CsvReportHandler(const CsvReportHandler& rhs) :
        AbstractReportHandler(rhs.getFileName()),
        _isStateWriteMandatory(rhs._isStateWriteMandatory){
}

CsvReportHandler::~CsvReportHandler()
// 11-07-2007: test on p_tuple by Volker Baier
{
    if (_pFile.is_open()) {
        _pFile.close();
        // delete _pFile;
        // _pFile = nullptr;
    }
}

void CsvReportHandler::writeReport(const Report& report) {

    // if (_nrReports == 0) {
    //     std::ostringstream stream;
    //     stream << "rate_" << report._id;
    //     stream.str().c_str();
    // }

    // auto vectorOfGridValues = report._grid.toStateVector();
    // _nrReports++, report._time, report._rate;
    // _pFile << "report received" << endl;
    // _pFile << "report _type" << report._type << endl;
    // if (report._type == STATE && (isStateWriteMandatory())){
    //     _pFile << _nrReports++ << ',';
    //     _pFile << report._id << ',';
    //     _pFile << report._time << ',';
    //     _pFile << report._rate << ',' << endl;
    // }

    _pFile << _nrReports++ << ',';
    _pFile << report._id << ',';
    _pFile << report._time << ',';
    _pFile << report._rate << endl;
    // always log ReportValue elements
    _valueHandler.addReport(report);
}

CsvReportHandler* CsvReportHandler::clone() const {
    return new CsvReportHandler(*this);
}

void CsvReportHandler::initializeHandler(const NodeId& nodeId) {
    // Purpose: this function will be called by MPINode upon configuration.
    // no canvas are generated as it would cause lot of problems with mpi
    if (!_pFile.is_open()) {
        // std::ofstream outputfile;
        // outputfile.open(getRootOutputFileName().c_str());
        // _pFile = outputfile;
        _pFile.open(getRootOutputFileName().c_str());
        cout << "created _pfile" << endl;
        // if (_pFile->IsZombie())
        //     throw utilities::Exception(STR_ROOT_FILE_OPENED_FAILED);

        _valueHandler.reset();
    }
    // store the node
    // cout << "initialize node" << nodeId << endl;
    _nodes.push_back(nodeId);
}

void CsvReportHandler::detachHandler(const NodeId& nodeId) {
    // Purpose: this function will be called upon MPINode destruction.
    // This works under the assumption that no isolated DynamicNodes
    // exist which are associated with an open CsvReporthandler.
    // Author: Marc de Kamps
    // Date: 26-08-2005

    removeFromNodeList(nodeId);

    if (!_valueHandler.isWritten())
        _valueHandler.write();
    // cout << "removing node " << nodeId << endl;
    if (_nodes.empty())
        finalize();
}

void CsvReportHandler::removeFromNodeList(NodeId nodeId) {
    auto iter = std::find(_nodes.begin(), _nodes.end(), nodeId);

    if (iter == _nodes.end())
        throw utilities::Exception(
                "Can't locate NodeId during detaching handler");

    _nodes.erase(iter);
}

void CsvReportHandler::finalize() {
    // cout << "close simulation file" << endl;
    _pFile.close();
    // cout << "delete simulation file" << endl;
    // if (_pFile) {
    //     delete _pFile;
    //     _pFile = nullptr;
    // }
    // cout << "finalizing simulation" << endl;
    _nodes.clear();
}

bool CsvReportHandler::isStateWriteMandatory() const {
    return _isStateWriteMandatory;
}

} // end namespace of handler
} // end namespace of report
} // end namespace of MPILib
