#include <iostream>
#include <fstream>
#include <SparseImplementationLib/SparseImplementationTest.h>
#include <boost/shared_ptr.hpp>

using std::ofstream;

using SparseImplementationLib::SparseImplementationTest;

int main()
{
	boost::shared_ptr<ostream> p(new ofstream("test.log"));


    SparseImplementationTest test(p);
     
	if (! test.Execute() )
		cout << "SparseImplementation test failed" << endl;

     return 0;
}

