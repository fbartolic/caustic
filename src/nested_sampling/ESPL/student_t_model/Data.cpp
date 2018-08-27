#include "Data.h"
#include <fstream>
#include <vector>

using namespace std;

// The static instance
Data Data::instance;

Data::Data()
{

}

void Data::load(const char* filename)
{
	// Vectors to hold the data
	std::vector<double> _t;
	std::vector<double> _F;
	std::vector<double> _sigF;

	// Open the file
	fstream fin(filename, ios::in);

	// Temporary variables
	double temp1, temp2, temp3, temp4, temp5;

    // Lambda functions for converting magnitudes to fluxes
    auto mag_to_flux = [] (double mag) { return pow(10., -mag/2.5); };
    auto magerr_to_fluxerr = [mag_to_flux] (double magerr, double mag) { return magerr*mag_to_flux(mag); };

	// Read until end of file
	while(fin>>temp1 && fin>>temp2 && fin>>temp3 && fin>>temp4 && fin>>temp5)
	{
        // Save HJD - 2450000
		_t.push_back(temp1 - 2450000.);
        // Convert I magnitude to flux
        double flux = mag_to_flux(temp2);
		_F.push_back(flux);
        // Convert I magnitude uncertainty to flux uncertainty  
        double flux_err = magerr_to_fluxerr(temp3, temp2);
		_sigF.push_back(flux_err);
	}

	// Close the file
	fin.close();

	// Copy the data to the valarrays
    // (these are basically the same as vector<double>)
	t = valarray<double>(&_t[0], _t.size());
	F = valarray<double>(&_F[0], _F.size());
	sigF = valarray<double>(&_sigF[0], _sigF.size());

    // Normalize data to unit interval
    double F_min = F.min();
    double F_max = F.max();

    for (std::size_t i; i < F.size(); ++i) {
        F[i] = (F[i] - F_min)/(F_max - F_min);
        sigF[i] = sigF[i]/(F_max - F_min);
    }
}

