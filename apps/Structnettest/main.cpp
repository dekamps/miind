#include <iostream>

using std::cout;
using std::endl;

#include <StructnetLib/StructnetLib.h>
#include <StructnetLib/StructnetLibTest.h>

using std::ifstream;
using std::cout;
using std::endl;
using StructnetLib::StructNetLibTest;
using StructnetLib::StructnetLibException;


int main()
{
	boost::shared_ptr<ostream> p(new ofstream("test.log"));

	StructNetLibTest test(p);

	try 
	{
		if (test.Execute())
			cout << "StructnetLib test suite succeeded" << endl;
		else
			cout << "StructnetLib test suite failed"    << endl;
	}

	catch (StructnetLib::StructnetLibException& exception)
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

	cout << "Finished" << endl;
	return 0;
}

