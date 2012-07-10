/*
 * FileNameGenerator.cpp
 *
 *  Created on: 22.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/FileNameGenerator.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>

#include <sstream>

namespace MPILib {
namespace utilities {

FileNameGenerator::FileNameGenerator(const std::string& fileName, FileType fileType) {
	MPIProxy mpiProxy;

	std::stringstream tempFileName;

	if (fileType == LOGFILE) {
		tempFileName << fileName << "_" << mpiProxy.getRank() << ".log";
	}
	if (fileType == ROOTFILE) {
		tempFileName << fileName << "_" << mpiProxy.getRank() << ".root";
	}
	_fileName = tempFileName.str();

}

FileNameGenerator::~FileNameGenerator() {
}

std::string FileNameGenerator::getFileName() const {
	return _fileName;
}

} /* namespace utilities */
} /* namespace MPILib */
