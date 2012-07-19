// Copyright (c) 2005 - 2010 Marc de Kamps
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
#include "Sequence.h"
#include "SequenceIteratorIterator.h"
#include "UtilLibException.h"

using namespace UtilLib;
#include <sstream>
#include <boost/foreach.hpp>
#include "boost/shared_ptr.hpp"
#include "SequenceIteratorIterator.h"

using namespace boost;
using namespace UtilLib;
using namespace std;

SequenceIteratorIterator::SequenceIteratorIterator(bool is_unique):
_is_unique(is_unique),
_has_not_started(true),
_ind(0),
_size(0),
_vec_sequences(0),
_vec_placeholder(0)
{
}

void SequenceIteratorIterator::ConfigureLoop()
{
	_has_not_started = false;
	BOOST_FOREACH(sequence_with_iterator& p,_vec_sequences){
		p.second = p.first->begin();
	}
	_vec_placeholder.resize(_vec_sequences.size());
	this->size();
	CopyToPlaceHolder();
}

bool SequenceIteratorIterator::AddLoop(const Sequence& seq)
{
	if (_has_not_started){
		try {
			sequence_with_iterator seqwi;
			seqwi.first = boost::shared_ptr<Sequence>(seq.Clone());
			_vec_sequences.push_back(seqwi);
		}
		catch(...){
			return false;
		}
	}else 
		return false;

	return true;
}

vector<double>& SequenceIteratorIterator::operator*()
{
	if (_has_not_started)
		ConfigureLoop();
	return _vec_placeholder;
}


Number SequenceIteratorIterator::size() const {

	Number n = 1;
	BOOST_FOREACH(sequence_with_iterator seqwi,_vec_sequences)
	{
		n *= seqwi.first->size();
	}

	_size = n;
	return _size;
}

SequenceIteratorIterator& SequenceIteratorIterator::operator ++()
{
	ShiftIndices();
	CopyToPlaceHolder();
	return *this;
}

SequenceIteratorIterator SequenceIteratorIterator::operator++(int)
{
	SequenceIteratorIterator ret = *this;
	ShiftIndices();
	CopyToPlaceHolder();
	return ret;
}	

void SequenceIteratorIterator::CopyToPlaceHolder()
{
	if (_ind >= _size)
		return;

	if (_has_not_started)
		ConfigureLoop();

	vector<double>::iterator it_double= _vec_placeholder.begin();
	BOOST_FOREACH(sequence_with_iterator& seqwi, _vec_sequences){
		*it_double++ = *seqwi.second;
	}
}

vector<string> SequenceIteratorIterator::NamesList() 
{
	vector<string> vec_ret;

	if (_has_not_started)
		ConfigureLoop();

	vector<string>::iterator it_string;

	BOOST_FOREACH(sequence_with_iterator& seqwi, _vec_sequences){
		*it_string++ = seqwi.first->Name();
	}

	return vec_ret;
}

void SequenceIteratorIterator::ShiftIndices(){
	++_ind;

	// iterator is now equal to 'end', do nothing
	if (_ind >= _size)
		return;

	// loop back over all iterators that are at the end until you reach the first one that is not at the end
	vector<sequence_with_iterator>::iterator it_sequence = _vec_sequences.begin();

	while(it_sequence->second  == (it_sequence->first->end())-1)
		it_sequence++;

	// now it is at the first iterator that is not at the end, hurray. increase that one
	it_sequence->second++;
	// reset all the earlier iterators to their beginning


	for
	(
		vector<sequence_with_iterator>::iterator it = _vec_sequences.begin();
		it != it_sequence;
		it++
	)
		it->second = it->first->begin();
}

SequenceIteratorIterator& SequenceIteratorIterator::operator+(Index i)
{
	if (_has_not_started)
		ConfigureLoop();

	Index new_ind = _ind + i;
	if (new_ind < _size){
		for (Index ind = 0; ind < i; ind++)
			this->operator++();
	}
	else
		_ind = _size;

	return *this;
}

string SequenceIteratorIterator::CurrentName() 
{
	if (_has_not_started)
		ConfigureLoop();

	ostringstream str_ret;

	BOOST_FOREACH(sequence_with_iterator& seqwi,_vec_sequences)
	{
		str_ret << "_" << seqwi.first->Name() << "_" << *seqwi.second ;
	}
	str_ret << '\0';

	return str_ret.str();
}

bool UtilLib::operator!=(const SequenceIteratorIterator& it1, const SequenceIteratorIterator& it2)
{
	return (it1._ind != it2._ind);
}

SequenceIteratorIterator UtilLib::operator+(const SequenceIteratorIterator& it, Index i)
{
	SequenceIteratorIterator iter_ret = it;
	return iter_ret + i;
}
