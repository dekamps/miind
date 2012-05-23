// Copyright (c) 2005 - 2009 Marc de Kamps, Dave Harrison
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
//
#include <cppunit/TestFixture.h>
#include <cppunit/Test.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestCaller.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

#include "Interpolation.h"
#include "NumtoolsLibException.h"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace NumtoolsLib;

class InterpolationTest: public CppUnit::TestFixture
{

    private:
        Interpolator* interpolator;
        const unsigned int SIZE;
        std::valarray<double> xa;
        std::valarray<double> ya;

    public:

        InterpolationTest() :
            SIZE(300), xa(SIZE), ya(SIZE)
        {
            double j;
            for (unsigned int i = 0; i < SIZE; ++i)
            {
                j = i / 10.0;
                xa[i] = j;
                ya[i] = j * cos(j * j);
            }
        }

        ~InterpolationTest()
        {
            if (interpolator)
                delete interpolator;
        }

        /*!
         * Sets up the test fixture
         *
         * Called before every test case method
         */
        void setUp()
        {
            // Nothing needed: each interpolator is created in its own
            // test method
        }

        /*!
         * Tears down the test fixture.
         *
         * Called after every test case method.
         */
        void tearDown()
        {
            // release objects under test here, as necessary
            delete interpolator;
            interpolator = 0;
        }

        /*!
         * Test Linear interpolation
         */
        void linear()
        {
            interpolator = new Interpolator(::INTERP_LINEAR, xa, ya);

            double result = interpolator->InterpValue(15.55);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE( "Linear Interpolation failed", -0.24932, result, 0.0001);
        }

        /*!
         * Test Linear interpolation at end of data
         *
         * Should throw NumtoolsException
         */
        void linear_end()
        {
            interpolator = new Interpolator(::INTERP_LINEAR, xa, ya);

            interpolator->InterpValue(30.0);
        }

        /*!
         * Test Linear interpolation at start of data
         *
         * Should throw NumtoolsException
         */
        void linear_start()
        {
            interpolator = new Interpolator(::INTERP_LINEAR, xa, ya);

            interpolator->InterpValue(0.0);
        }

        /*!
         * Test CSpline interpolation
         */
        void cspline()
        {
            interpolator = new Interpolator(::INTERP_CSPLINE, xa, ya);

            double result = interpolator->InterpValue(15.55);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE( "CSpline Interpolation failed", -0.62031, result, 0.0001);
        }

        /*!
         * Test CSpline interpolation at end of data
         *
         * Should throw NumtoolsException
         */
        void cspline_end()
        {
            interpolator = new Interpolator(::INTERP_CSPLINE, xa, ya);

            interpolator->InterpValue(30.0);
        }

        /*!
         * Test CSpline interpolation at start of data
         *
         * Should throw NumtoolsException
         */
        void cspline_start()
        {
            interpolator = new Interpolator(::INTERP_CSPLINE, xa, ya);

            interpolator->InterpValue(0.0);
        }

        /*!
         * Test Akima interpolation
         */
        void akima()
        {
            interpolator = new Interpolator(::INTERP_AKIMA, xa, ya);

            double result = interpolator->InterpValue(15.55);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE( "Akima Interpolation failed", -0.09333, result, 0.0001);

        }

        /*!
         * Test Akima interpolation at end of data
         *
         * Should throw NumtoolsException
         */
        void akima_end()
        {
            interpolator = new Interpolator(::INTERP_AKIMA, xa, ya);

            interpolator->InterpValue(30.0);
        }

        /*!
         * Test Akima interpolation at start of data
         *
         * Should throw NumtoolsException
         */
        void akima_start()
        {
            interpolator = new Interpolator(::INTERP_AKIMA, xa, ya);

            // Akima needs 5 consecutive points, so this will not
            // throw.
            CPPUNIT_ASSERT_NO_THROW(interpolator->InterpValue(0.0));

            // But this will:
            interpolator->InterpValue(-1.0);
        }

        CPPUNIT_TEST_SUITE( InterpolationTest );

            CPPUNIT_TEST( linear );
            CPPUNIT_TEST_EXCEPTION( linear_end, NumtoolsException );
            CPPUNIT_TEST_EXCEPTION( linear_end, NumtoolsException );

            CPPUNIT_TEST( cspline );
            CPPUNIT_TEST_EXCEPTION( cspline_end, NumtoolsException );
            CPPUNIT_TEST_EXCEPTION( cspline_end, NumtoolsException );

            CPPUNIT_TEST( akima );
            CPPUNIT_TEST_EXCEPTION( akima_end, NumtoolsException );
            CPPUNIT_TEST_EXCEPTION( akima_start, NumtoolsException );

        CPPUNIT_TEST_SUITE_END();
};

int main()
{
    CppUnit::TextUi::TestRunner runner;
    runner.addTest(InterpolationTest::suite());
    runner.run();
    return 0;
}

