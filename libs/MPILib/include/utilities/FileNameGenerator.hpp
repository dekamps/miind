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

	/**
	 * Constructor which will generate the filenames in this way:
	 * filename + "_" + procesorId + "fileIdentifier"(either .root or .log)
	 * depending on the given fileType
	 * @param fileName The Filename, do not append an ending
	 * @param fileType The Type of a file, the file extension depends on this param
	 */
	explicit FileNameGenerator(std::string& fileName, FileType fileType =
			LOGFILE);
	virtual ~FileNameGenerator();

	/**
	 * Gives the generated Filename back
	 * @return A file name
	 */
	std::string getFileName() const;

private:

	std::string _fileName;
};

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_FILENAMEGENERATOR_HPP_ */
