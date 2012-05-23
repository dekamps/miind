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

#ifndef _CODE_LIBS_UTIL_STREAMABLE_INCLUDE_GUARD
#define _CODE_LIBS_UTIL_STREAMABLE_INCLUDE_GUARD 

#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>
#include "AttributeList.h"
#include "BasicDefinitions.h"
#include "LocalDefinitions.h"
#include "Named.h"

using std::istream_iterator;
using std::istream;
using std::ostream;
using std::ostringstream;
using std::string;
using std::vector;

//! \page miind_streaming The streaming model for MIIND objects.
//! \section stream_intro Introduction
//! Many MIIND objects must be written to and read from disk (serialization). There are examples of polymorphic frameworks (e.g. DynamicLib::DynamicNetwork), where the framework
//! needs to serialise objects whilst being unaware of the exact subclass of objects. In such cases the Streamable paradigm will be used. Objects that need to be serialised derive
//! from Streamable, which offers a Tag, a FromStream and d ToStream method. Tag is a string which provides an XML Tag for the object, and which must be overloaded by subclasses
//! of Streamable. The FrmStream and ToStream methods are not meant to be used by clients. Clients can use an operator<< function to write objects to a stream. When none is provided,
//! clients are not supposed to handle the objects serialization. Clients can use a constructor which accepts an istream& from an input stream. This construction moves the declaration-construction
//! away from the client: when a client is supposed to be able to construct an object from a stream, they can do so directly.
//! In general no operator>> is provided, except for simple struct-like objects.
namespace UtilLib
{
	//! Base class for object that need to be written to and from a stream.

	//! A pseudo XML format is usually used to keep the object human readable.
	class Streamable : public Named
	{
	public:

		Streamable();

		Streamable(const Streamable&);

		virtual ~Streamable() = 0;

		virtual bool   ToStream   (ostream&) const = 0;
		virtual bool   FromStream (istream&)       = 0;

		//! XML tag to be overloaded by the object to be serialized
		virtual string Tag      () const { return STR_UNDEFINED; }

		string ToEndTag (const string&) const;
		string DeTag	(const string&) const;

		//! The second string will be  a tag around the first string
		template <class Value>
		string WrapTag(const Value&, const string&) const;

		//! Untag the string, the value of the tag will not be considered
		string UnWrapTag(const string&) const;

		string InsertNameInTag(const string&, const string&) const;

		//! Add an attribute to an existing XML tag so that in the tag 'bla="ble"' appears, where
		//! bla is the attribute and ble is its value
		string AddAttributeToTag
		(	
			const string&, //!< original tag, which already may contain other attributes, which are separated by at least a white space
			const string&, //!< name of the attribute
			const string&  //!< value of the attribute, which MUST be in quotes! (So "ble", not ble)
		) const;

		//! Splits a tag and determines whether the format of the tag is valid. Returns true if so, and false if not.
		//! The tag's name is insert in the string that the first argument is pointing to. A list of attribute-value pairs is insert
		//! in the AttributeList that the second argument is pointing to.

		//! A tag can be decode into its tag name and a list of attribute-value pairs so that it can be more easily manipulated,
		//! for example to insert a new pair into an existing tag. If a node tag contains no valid attribute pairs, but is a tag, the
		//! function will still be true. It must be tested within the list whether the expected number of attributes are present.
		bool DecodeTag
		(
			string*,		//! insert a pointer to the string holding the complete tag, and after running this string will contain only the tag name
			AttributeList*  //! a list of attribute-value pairs
		);

		//! after manipulation the tag and attribute list can be  recompiled into a full tag string
		string CompileTag(const string&, const AttributeList&) const;

		bool StripNameFromTag(string*, const string&) const;

		template <class Value>
			bool StreamToVector
			(	
				istream&,
				Number,
				vector<Value>*
			);

		template <class Value>
			bool StreamToArray
			(
				istream&,
				Number,
				Value*
			);
				
	}; // end of Streamable

	template <class Value>
	bool Streamable::StreamToVector
	(
		istream&        s,
		Number         number_of_elements,
		vector<Value>* p_vector
	)
	{
		p_vector->clear();
		p_vector->reserve(number_of_elements);

		copy
		(
			istream_iterator<Value>(s),
			istream_iterator<Value>(),
			back_inserter(*p_vector)
		);

		// fails because end tag is reached,
		// clear state to restore end tag

		s.clear();

		string str;
		s >> str;
		if (
				str == ToEndTag(Tag()) &&
				p_vector->size() == number_of_elements &&
				s.good()
		)
			return true;
		else
			return false;
	}

	template <class Value>
	bool Streamable::StreamToArray
	(
		istream&      s,
		Number        number_of_elements,
		Value*        p_begin
	)
	{
		for (
				int index = 0; 
				index < static_cast<int>(number_of_elements); 
				index++ 
			)
				s >> *(p_begin + index);


		if (
				s.good()
			)
			return true;
		else 
			return false;

	}

	template <class Value>
	string Streamable::WrapTag(const Value& val, const string& tag_name) const
	{
		ostringstream str;
		str << "<" << tag_name << ">" << val << "</" << tag_name << ">";
		return str.str();
	}
		

} // end of UtilLib

#endif // include guard
