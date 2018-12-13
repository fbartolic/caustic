#ifndef DNest4_Data
#define DNest4_Data

#include <valarray>

/*
* An object of this class is a dataset
*/
class Data
{
	private:
		// The data points
		std::valarray<double> t;
		std::valarray<double> F;
		std::valarray<double> sigF;

	public:
		// Constructor
		Data();

		// Load data from a file
		void load(const char* filename);

		// Access to the data points
		const std::valarray<double>& get_t() const
		{ return t; }
		const std::valarray<double>& get_F() const
		{ return F; }
        const std::valarray<double>& get_sigF() const
		{ return sigF; }


	private:
		// Static "global" instance
		static Data instance;

	public:
		// Getter for the global instance
		static Data& get_instance()
		{ return instance; }
};

#endif

