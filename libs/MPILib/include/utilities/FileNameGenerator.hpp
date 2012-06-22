/*
 * FileNameGenerator.hpp
 *
 *  Created on: 22.06.2012
 *      Author: david
 */

#ifndef MPILIB_UTILITIES_FILENAMEGENERATOR_HPP_
#define MPILIB_UTILITIES_FILENAMEGENERATOR_HPP_

#include <string>

namespace MPILib {
namespace utilities {

enum FileType {
	LOGFILE, ROOTFILE
};

class FileNameGenerator {
public:

	explicit FileNameGenerator(std::string& filename, FileType fileType =
			LOGFILE);
	virtual ~FileNameGenerator();

	std::string getFileName() const;

private:

	std::string _fileName;
};

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_FILENAMEGENERATOR_HPP_ */
