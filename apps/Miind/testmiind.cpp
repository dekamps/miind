#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <PopulistLib/PopulistLib.h>
#include <MiindLib/MiindTest.h>

using MiindLib::MiindTest;

int main(){

	boost::shared_ptr<ostream> p(new ofstream("test.log"));

	MiindTest test(p);

	try 
	{
		if (test.Execute())
			cout << "MiindLib test suite succeeded" << endl;
		else
			cout << "MiindLib test suite failed"    << endl;
	}

	catch (PopulistLib::PopulistException& exception)
	{
		cout << exception.Description() << endl;
	}
	catch (NetLib::NetLibException& exception)
	{
		cout << exception.Description() << endl;
	}
	catch (UtilLib::GeneralException& exception)
	{
		cout << exception.Description() << endl;
	}
	catch (...)
	{
		cout << "Unknown error occured" << endl;
	}

	return 0;
}
