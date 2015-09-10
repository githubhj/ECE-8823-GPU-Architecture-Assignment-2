#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include "pgma_io.hpp"
#include <vector>
#include <string>
#include <algorithm>

class PGM
{
public:
	PGM() : N(0), ptr(NULL) {}
	PGM(const PGM &rhs) : N(0), ptr(NULL)
	{
		copy(rhs);
	}
	~PGM() {
		if (ptr != NULL) {
			delete [] ptr;
		}
	}
	PGM& operator=(const PGM &rhs)
	{
		if (this == &rhs)
			return *this;
		return copy(rhs);
	}
	PGM& copy(const PGM &rhs)
	{
		if (ptr != NULL)
		{
			delete [] ptr;
		}
		N = rhs.N;
		size_t imageSize = N * N * sizeof *(rhs.ptr);
		ptr = new int[imageSize];
		memcpy(ptr, rhs.ptr, imageSize);
		return *this;
	}
	bool operator==(const PGM &rhs) const
	{
		if (N == rhs.N) {
			for(int i = 0; i < N * N; i++)
			{
				if (ptr[i] != rhs.ptr[i])
				{
					return false;
				}
			}
		} else {
			return false;
		}
		return true;
	}
	int N;
	int *ptr;
};

PGM getImage(std::string fileName)
{
	PGM image;
	int x, y, maxVal;
	pgma_read(fileName, x, y, maxVal, &(image.ptr));
	assert(x == y);
	image.N = x;
	return image;
}

int main(int argc, char** argv)
{
	assert(argc > 2);
	std::vector<std::string> args;
	std::copy(argv+1, argv + argc, std::back_inserter(args));
	
	std::string fileName = args[1];
	PGM image = getImage(fileName);
	PGM hostOutput = image;
	
	//construct kxk filter
	std::cout << "Constructing kernel:\n";
	int k = atoi(args[0].c_str());
	assert(k % 2 == 1);
	int *kernel = new int[k*k*sizeof(int)];
	int totalVal = 0;
	for (int row = 0; row < k; row++)
	{
		for (int col = 0; col < k; col++)
		{
			int colVal = (col < (k/2+1)) ? col+1 : k-col;
			int rowVal = (row < (k/2+1)) ? row+1 : k-row;
			kernel[row * k + col] = colVal + rowVal;
			totalVal += colVal + rowVal;
			std::cout << kernel[row*k + col] << " ";
		}
		std::cout << "\n";
	}

	PGM gpuOutput = image;
/////////////////////////////////////////////////////////////////
// Insert CUDA launch code here
/////////////////////////////////////////////////////////////////

	std::cout << "Running GPU kernel\n\n";
	//use k, kernel, image.N and image.ptr as your inputs
	//copy output to gpuOutput.ptr, data is already allocated
	//make sure the dimensions of the image are the same

/////////////////////////////////////////////////////////////////

	//CPU convolution
	std::cout << "Running host kernel\n\n";
	int kRadius = k/2;
	for (int row = 0; row < image.N; row++)
	{
		for (int col = 0; col < image.N; col++)
		{
			//sample from neighbor pixels 
			int index = row * image.N + col;
			int value = 0;
			for (int kRow = -kRadius; kRow <= kRadius; kRow++)
			{
				//image bounds check
				if (row-kRadius < 0 || row+kRadius >= image.N)
					continue;
				for (int kCol = -kRadius; kCol <= kRadius; kCol++)
				{
					//image bounds check
					if (col-kRadius < 0 || col+kRadius >= image.N)
						continue;
					value += kernel[(kRadius + kRow)*k + kRadius + kCol] * image.ptr[index + kRow*image.N + kCol];
				}
			}
			hostOutput.ptr[index] = value / totalVal;
		}
	}

	std::cout << "Comparing results:\n";
	bool passed = hostOutput == gpuOutput;
	std::string resultString = (passed) ? "Passed\n" : "Failed\n";
	std::cout << resultString;

	std::cout << "Writing image outputs: output_host.pgm output_gpu.pgm\n";
	std::string outputFileName = "output_host.pgm";
	pgma_write(outputFileName, image.N, image.N, hostOutput.ptr);
	outputFileName = "output_gpu.pgm";
	pgma_write(outputFileName, image.N, image.N, gpuOutput.ptr);

	delete(kernel);
}
