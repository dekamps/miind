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

#ifndef MPILIB_UTILITIES_PROGRESSBAR_HPP_
#define MPILIB_UTILITIES_PROGRESSBAR_HPP_

#include <iostream>


namespace MPILib {
namespace utilities {

class ProgressBar {
public:
	/**
	 * Constructor
	 * @param expected_count The expected count
	 * @param description The description of the progress Bar
	 * @param os The output stream
	 */
	explicit ProgressBar(unsigned long expected_count,
			const std::string & description = "", std::ostream& os = std::cout);

	/**
	 * displays the progressbar
	 * @param expected_count The expected total count
	 * @post count()=0
	 * @post expected_count()==expected_count
	 */
	void restart(unsigned long expected_count);

	/**
	 * Display appropriate progress tic if needed.
	 * @param increment
	 * @post count()== original count() + increment
	 */
	unsigned long operator+=(unsigned long increment);

	/**
	 * Prefix operator
	 */
	unsigned long operator++();

	/**
	 * Postfix operator
	 */
	unsigned long operator++(int);

private:
	unsigned long _count, _expectedCount, _nextTicCount;
	unsigned int _tic;

	/**
	 * Description of the progress Bar
	 */
	const std::string _description;
	/**
	 * The stream where the progress Bar is printed to.
	 */
	std::ostream& _outputStream;

	/**
	 * use of floating point ensures that both large and small counts
	 * work correctly.  static_cast<>() is also used several places
	 * to suppress spurious compiler warnings.
	 */
	void display_tic();
};

} /* namespace utilities */
} /* namespace MPILib */
#endif /* MPILIB_UTILITIES_PROGRESSBAR_HPP_ */
