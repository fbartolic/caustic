#include <iostream>
#include "Data.h"
#include "../../DNest4/code/DNest4.h"
#include "ESPL.h"

using namespace std;

int main(int argc, char** argv)
{
	Data::get_instance().load("../../../microlensing_data/OGLE/2017/blg-0831/phot.dat");
     // Initialize distribution objects for use in from_prior and perturb methods

    DNest4::start<ESPL>(argc, argv);
   
	return 0;
}

