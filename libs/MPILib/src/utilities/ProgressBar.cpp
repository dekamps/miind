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

#include <MPILib/include/utilities/ProgressBar.hpp>
#include <MPILib/include/utilities/MPIProxy.hpp>


namespace MPILib {
namespace utilities {

ProgressBar::ProgressBar(unsigned long expectedCount,
		const std::string & description, std::ostream& os) :
		_description(description), _outputStream(os) {
	MPIProxy mpiProxy;
	if (mpiProxy.getRank() == 0) {
		restart(expectedCount);
	}
}

void ProgressBar::restart(unsigned long expected_count) {
	_count = _nextTicCount = _tic = 0;
	_expectedCount = expected_count;
	_outputStream << _description << "\n"
			<< "0%   10   20   30   40   50   60   70   80   90   100%\n"
			<< "|----|----|----|----|----|----|----|----|----|----|"
			<< std::endl;

}

unsigned long ProgressBar::operator+=(unsigned long increment) {
	MPIProxy mpiProxy;
	if (mpiProxy.getRank() == 0) {
		if ((_count += increment) >= _nextTicCount) {
			display_tic();
		}
	}
	return _count;
}

unsigned long ProgressBar::operator++() {
	return operator+=(1);
}

unsigned long ProgressBar::operator++(int) {
	return operator+=(1);
}

void ProgressBar::display_tic() {
	unsigned int tics_needed =
			static_cast<unsigned int>((static_cast<double>(_count)
					/ _expectedCount) * 50.0);
	do {
		_outputStream << '*' << std::flush;
	} while (++_tic < tics_needed);
	_nextTicCount =
			static_cast<unsigned long>((_tic / 50.0) * _expectedCount);
	if (_count == _expectedCount) {
		if (_tic < 51)
			_outputStream << '*';
		_outputStream << std::endl;
	}
}

} /* namespace utilities */
} /* namespace MPILib */
