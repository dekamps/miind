// Copyright (c) 2005 - 2008 Marc de Kamps
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

#include <sstream>
#include "../UtilLib/UtilLib.h"
#include "SquashingFunctionFactory.h"
#include "UnknownSquashingException.h"
#include "LocalDefinitions.h"
#include "Sigmoid.h"
#include "NoSquashingFunction.h"

using namespace std;
using namespace UtilLib;
using namespace NetLib;

                 
SquashingFunctionFactory::SquashingFunctionFactory()
{
		// Register known squashing functions

		NoSquashingFunction no_squash;

		// Force static member into existence before relying on its existence

		this->Register
			(
				no_squash.Tag(),
				no_squash
			);

		Sigmoid sigmoid;
		this->Register
			(
				sigmoid.Tag(),
				sigmoid
			);
}

auto_ptr<AbstractSquashingFunction> SquashingFunctionFactory::FromStream(istream& s) const
{
	string tag;
	s >> tag;

	const AbstractSquashingFunction* p_function;
	if ( (p_function = this->find(tag)) )
	{
//		for (int i_length = static_cast<int>(tag.length()) - 1; i_length >= 0; i_length--)
//
		// The front tag has been removed from the istream, to determine the
		// the type, but we need the entire object descriptio to create
		// the SuqshingFunction, therefore we create 
		// a stringstream to constuct the object from

		string string_current;
		ostringstream  stream_squashing_object;
		stream_squashing_object << tag << " ";
	
		ConcreteStreamable streamable;
		while ( string_current != streamable.ToEndTag(tag) )
		{
			s >> string_current;
			stream_squashing_object << string_current << " ";
		}

		istringstream stream_squash(stream_squashing_object.str());

//		s.putback(tag[i_length]);
		AbstractSquashingFunction* p_return = p_function->Clone();
		p_return->FromStream(stream_squash);
		return auto_ptr<AbstractSquashingFunction>(p_return);
	}
	else 
		throw UnknownSquashingException(STR_SQUASH_EXCEPTION);

}

bool SquashingFunctionFactory::Register(const string& tag, const AbstractSquashingFunction& function)
{
	pair<const string, const AbstractSquashingFunction*> value_pair(tag,function.Clone());

	typedef map<const string, const AbstractSquashingFunction*>::iterator Iterator;

	pair<Iterator,bool> result_insert = _map_squashing_function.insert(value_pair);

/*	if (result_insert.second)
		return true;
	else 
		throw UnknownSquashingException(STR_DOUBLE_REGISTER);
*/
	return true;
}

const AbstractSquashingFunction* SquashingFunctionFactory::find(const string& tag) const
{
	map<const string,  const AbstractSquashingFunction*>::const_iterator iter;
	
	iter = _map_squashing_function.find(tag);
	
	if (iter != _map_squashing_function.end() )
		return iter->second;
	else 
		return 0;
}
