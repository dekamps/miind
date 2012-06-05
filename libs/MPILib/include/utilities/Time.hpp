/*
 *
 * Copyright (c) 2011 David-Matthias Sichau
 * Copyright (c) 2010 Marc Kirchner
 *
 * This file is part of libpipe.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef MPILIB_UTILITIES_TIME_HPP__
#define MPILIB_UTILITIES_TIME_HPP__

// for all OS
#include <ctime> // for timeval
#include <ostream>
#include <limits>
#ifndef _WIN32
//Linux and Mac stuff
#include <sys/time.h>
#endif



/** Comparison operator for \c struct \c timeval objects.
 */
bool operator==(const timeval& lhs, const timeval& rhs);

/** Comparison operator for \c struct \c timeval objects.
 */
bool operator<=(const timeval& lhs, const timeval& rhs);

/** Stream operator for timeval structs.
 */
std::ostream& operator<<(std::ostream& os, const timeval& tv);


#ifdef _WIN32
// Windows stuff
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
    int tz_minuteswest; /* minutes west of Greenwich */
    int tz_dsttime; /* type of dst correction */
};
/** gettimeofday for Windows
 * @param tv A timeval object.
 * @param tz A timezone object.
 * @return Error code. Zero upon success.
 */
int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif


namespace MPILib {
namespace utilities {

/** Platform independent max limit for type
 * @param val the variable which will get the max value
 */
template<typename T>
void set_max(T& val)
{
    val = std::numeric_limits<T>::max();
}
/** Platform independent min limit for type
 * @param val the variable which will get the min value
 */
template<typename T>
void set_min(T& val)
{
    val = std::numeric_limits<T>::min();
}
} // end namespace MPILib
} // end namespace libpipe
#endif /* MPILIB_UTILITIES_TIME_HPP__ */
