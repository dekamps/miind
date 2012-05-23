// Copyright (c) 2005 - 2011 Marc de Kamps
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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <PopulistLib.h>

using namespace PopulistLib;


BOOST_AUTO_TEST_CASE(OneTimeBatchHasElapsed){

	ProbabilityQueue queue;

	StampedProbability prob;
	Time t_11 = 0.5e-4;
	Time t_12 = 0.6e-4;
	Time t_first_batch = 1e-4; // default time step
	double prob_11 = 0.1;
	double prob_12 = 0.2;

	prob._time = t_11;
	prob._prob = prob_11;
	queue.push(prob);

	prob._time = t_12;
	prob._prob = prob_12;
	queue.push(prob);
	double retrieved = queue.CollectAndRemove(t_first_batch);

	BOOST_CHECK(retrieved == 0);
	BOOST_CHECK(queue.TotalProbability() == prob_11 + prob_12);
}


BOOST_AUTO_TEST_CASE(Queue)
{
	ProbabilityQueue queue;

	StampedProbability prob;
	Time t_11 = 0.5e-4;
	Time t_12 = 0.6e-4;
	Time t_first_batch = 1e-4; // default time step
	double prob_11 = 0.1;
	double prob_12 = 0.2;
	BOOST_REQUIRE(! queue.HasProbability(t_first_batch) );

	prob._time = t_11;
	prob._prob = prob_11;
	queue.push(prob);

	BOOST_REQUIRE(! queue.HasProbability(t_first_batch) );

	prob._time = t_12;
	prob._prob = prob_12;
	queue.push(prob);

	BOOST_REQUIRE( ! queue.HasProbability(t_first_batch) );
	BOOST_CHECK_CLOSE( queue.TotalProbability(),prob_11 + prob_12, 1e-9);

	Time t_21 = 1.1e-4;
	double prob_21 = 0.4;
	prob._time = t_21;
	prob._prob = prob_21;
	queue.push(prob);

	BOOST_REQUIRE( queue.HasProbability(t_first_batch) );
	BOOST_CHECK_CLOSE( queue.TotalProbability(), prob_11 + prob_12 + prob_21, 1e-9);

	double retrieved = queue.CollectAndRemove(t_first_batch);
	BOOST_CHECK_CLOSE( retrieved, prob_11 + prob_12, 1e-9);
	BOOST_CHECK_CLOSE( queue.TotalProbability(), prob_21,1e-9);
}

BOOST_AUTO_TEST_CASE(TriggerException){
	ProbabilityQueue queue;

	StampedProbability prob;
	Time t_11 = 0.5e-4;
	Time t_12 = 0.6e-4;
	Time t_13 = 0.7e-4;

	double prob_11 = 0.1;
	double prob_12 = 0.2;
	double prob_13 = 0.3;

	prob._time = t_11;
	prob._prob = prob_11;
	queue.push(prob);

	prob._time = t_12;
	prob._prob = prob_12;
	queue.push(prob);

	prob._time = t_13;
	prob._prob = prob_13;
	try {
		queue.push(prob);
	}
	catch(PopulistException& excep)
	{
	}
	catch(...){
		BOOST_CHECK(false);
	}

}
BOOST_AUTO_TEST_CASE(RetrieveMultipleDepositis){
	ProbabilityQueue queue;

	StampedProbability prob;
	Time t_11 = 0.5e-4;
	Time t_12 = 0.6e-4;
	Time t_first_batch = 1e-4; // default time step
	double prob_11 = 0.1;
	double prob_12 = 0.2;

	prob._time = t_11;
	prob._prob = prob_11;
	queue.push(prob);

	prob._time = t_12;
	prob._prob = prob_12;
	queue.push(prob);

	Time t_21 = 1.5e-4;
	double prob_21 = 0.01;
	prob._time = t_21;
	prob._prob = prob_21;
	queue.push(prob);

	Time t_22 = 1.6e-4;
	double prob_22 = 0.023;
	prob._time = t_22;
	prob._prob = prob_22;
	queue.push(prob);

	Time t_31 = 2.1e-4;
	double prob_31 = 0.000123;
	prob._time = t_31;
	prob._prob = prob_31;
	queue.push(prob);

	double retrieved = queue.CollectAndRemove(2*t_first_batch);
	
	BOOST_CHECK_CLOSE(retrieved, prob_11 + prob_12 + prob_21 + prob_22, 1e-9);
	BOOST_CHECK_CLOSE(queue.TotalProbability(),prob_31,1e-9);
}

BOOST_AUTO_TEST_CASE(LargerThanBatch){
	// Add events that have time stamps much larger than individual batches
	ProbabilityQueue queue;

	StampedProbability prob;
	Time t_11 = 10e-4;
	Time t_12 = 20e-4;

	double prob_11 = 0.1;
	double prob_12 = 0.2;

	prob._time = t_11;
	prob._prob = prob_11;
	queue.push(prob);

	prob._time = t_12;
	prob._prob = prob_12;
	queue.push(prob);

}

BOOST_AUTO_TEST_CASE(QueueScaleTest){
	const double scale = 0.6;
	ProbabilityQueue queue;

	StampedProbability prob;
	Time t_11 = 0.5e-4;
	Time t_12 = 0.6e-4;
	Time t_first_batch = 1e-4; // default time step
	double prob_11 = 0.1;
	double prob_12 = 0.2;
	BOOST_REQUIRE(! queue.HasProbability(t_first_batch) );

	prob._time = t_11;
	prob._prob = prob_11;
	queue.push(prob);

	BOOST_REQUIRE(! queue.HasProbability(t_first_batch) );

	prob._time = t_12;
	prob._prob = prob_12;
	queue.push(prob);

	BOOST_REQUIRE( ! queue.HasProbability(t_first_batch) );
	BOOST_CHECK_CLOSE( queue.TotalProbability(),prob_11 + prob_12, 1e-9);

	Time t_21 = 1.1e-4;
	double prob_21 = 0.4;
	prob._time = t_21;
	prob._prob = prob_21;
	queue.push(prob);

	BOOST_REQUIRE( queue.HasProbability(t_first_batch) );

	BOOST_CHECK_CLOSE(queue.TotalProbability(), prob_11 + prob_12 + prob_21, 1e-9);
	queue.Scale(scale);
	BOOST_CHECK_CLOSE(queue.TotalProbability(), scale*(prob_11 + prob_12 +prob_21), 1e-9);
	double retrieved = queue.CollectAndRemove(1e-4);
	BOOST_CHECK_CLOSE(retrieved,scale*(prob_11+prob_12),1e-9);
}
