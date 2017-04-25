/*
 * TranslationObject.hpp
 *
 *  Created on: Apr 20, 2017
 *      Author: scsmdk
 */

#ifndef TRANSLATIONOBJECT_HPP_
#define TRANSLATIONOBJECT_HPP_

#include <vector>
#include <iostream>
#include "Coordinates.hpp"
#include "Mesh.hpp"
#include "pugixml.hpp"

namespace TwoDLib {

	struct Translation {
		double _v;
		double _w;

		Translation():_v(0),_w(0){}

		Translation(double v, double w):_v(v),_w(w){}
	};

	class TranslationObject {
	public:


		//! A translation object can either be instantiated by a double
		TranslationObject(double, double);

		//! Or by a jump file
		TranslationObject(std::istream&);

		//! Must be called before  the translation list can be used. Returns false if the Mesh is incompatible with the existing information,
		//! e.g. as read from a jump file.
		bool GenerateTranslationList(const Mesh&);

		//! There is always an efficacy associated with a TransitionMatrix, even if the individual jumps per cell may be different.
		//! This efficacy is written into the matrix file and used by the framework to pick the right transition matrix
		Translation Efficacy() const { return _efficacy; }

		//! Provides access to the list of translations
		const std::vector< std::vector<Translation> > TranslationList () const {return _translation_list; }

	private:

		pugi::xml_document _doc;

		Translation  EfficacyFromJumpFile(std::istream&);      // this function has as a side effect that the XML document will be loaded
		std::vector< std::vector<Translation> > ListFromDoc(); // uses the fact that the doc has been read
		const double _v;
		const double _w;

		const Translation _efficacy;

		std::vector< std::vector<Translation> > _translation_list;
	};
}

#endif /* TRANSLATIONOBJECT_HPP_ */
