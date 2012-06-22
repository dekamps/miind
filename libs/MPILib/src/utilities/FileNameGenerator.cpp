/*
 * FileNameGenerator.cpp
 *
 *  Created on: 22.06.2012
 *      Author: david
 */

#include <MPILib/include/utilities/FileNameGenerator.hpp>

#include <boost/mpi/communicator.hpp>
#include <sstream>
namespace mpi = boost::mpi;

namespace MPILib {
namespace utilities {

FileNameGenerator::FileNameGenerator(std::string& filename, FileType fileType) {
	mpi::communicator world;

	int processorId = world.rank();
	int totalProcessors = world.size();
	std::stringstream tempFileName;

	if (fileType == LOGFILE) {
		tempFileName << filename << "_" << processorId << ".log";
	}
	if (fileType == ROOTFILE) {
		tempFileName << filename << "_" << processorId << ".root";
	}
	_fileName = tempFileName.str();

}

FileNameGenerator::~FileNameGenerator() {
	// TODO Auto-generated destructor stub
}

std::string FileNameGenerator::getFileName() const {
	return _fileName;
}

} /* namespace utilities */
} /* namespace MPILib */
