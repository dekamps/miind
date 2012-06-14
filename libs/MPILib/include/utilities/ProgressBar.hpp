/*
 * ProgressBar.hpp
 *
 *  Created on: 14.06.2012
 *      Author: david
 */

#ifndef PROGRESSBAR_HPP_
#define PROGRESSBAR_HPP_

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
	unsigned long _count, _expected_count, _next_tic_count;
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
#endif /* PROGRESSBAR_HPP_ */
