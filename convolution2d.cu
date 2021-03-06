//============================================================================
// Name        : convolution2d.cu
// Author      : Harshit Jain
// Class	   : ECE 8823
// GTID		   : 903024992
// Assignment  : Assignment 1
// Copyright   : Public
// Description : 2D Convolution in CUDA
//============================================================================

/************************************************/
// Kernel Size	|	Min Trans.	|	Max Trans.	//
//		3		|	 293764		|	 293764		//
//		5		|	 327184		|	 327184		//
//		7		|	 362404		|	 362404		//
//		9		|	 399424		|	 399424		//
//		11		|	 438244		|	 438244		//
//		13		|	 478864		|	 478864		//
//		15		|	 521284		|	 521284		//
//		17		|	 565504		|	 565504		//
//		19		|	 611524		|	 611524		//
//		21		|	 659344		|	 659344		//
//		23		|	 708964		|	 708964		//
//		25		|	 760384		|    760384		//
//		27		|	 813604		|	 813604		//
//		29		|	 868624		|	 868624		//
//		31		|	 925444		|    925444	 	//
/************************************************/


#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include "pgma_io.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_profiler_api.h>
#include <stdio.h>

#define TILE_WIDTH 32
#define KERNEL_SIZE 7

//extern __shared__ int s[];

#define checkCudaError(status) { \
	if(status != cudaSuccess) { \
		std::cout << "CUDA Error " << __FILE__ << ", " << __LINE__ \
			<< ": " << cudaGetErrorString(status) << "\n"; \
		exit(-1); \
	} \
}

__constant__ int gpuKernel[31*31];


__global__ void convolutionGPU(int* inputImage, int* outputImage, int imageWidth, int kernelSize, int totalVal) {

	//ADD CODE HERE
	//Shared memory of size TILE_WIDTH plus apron width on top and bottom
	extern __shared__ int sharedImageData[];
	
	//get kernel radius
	int kRadius = kernelSize/2;
	
	//get particular thread data location in input image
	int threadDataLoc = threadIdx.x + blockIdx.x*blockDim.x + threadIdx.y*imageWidth + (blockIdx.y*blockDim.y)*imageWidth;
	
	//get thread x,y coordinates
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	
	//Boundary x=0 y=0 copies upper apron
	if(threadIdx.x==0 && threadIdx.y==0){
		if(blockIdx.y==0){
			for(int row=0; row<kRadius ; row++){
				for(int col=0; col<TILE_WIDTH+kernelSize-1; col++){
					sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] = 0;
				}
			}
		}
		
		else{
			for(int row=0; row<kRadius ; row++){
				for(int col=0; col<TILE_WIDTH+kernelSize-1; col++){
					if(blockIdx.x==0 && col < kRadius){
						sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] =0;
					}
					else if(blockIdx.x==gridDim.x-1 && col>TILE_WIDTH+kRadius-1){
						sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] =0;
					}
					else{
						sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] = inputImage[threadDataLoc-kRadius*imageWidth - kRadius + col + row*imageWidth];
					}
					
				}
			}
		}
		
	}
	
	//Boundary threadID x=0, y=blockDim-1 copies lower apron
	else if(threadIdx.x==0 && threadIdx.y==blockDim.y-1){
		int starting_index = (TILE_WIDTH + kRadius)*(TILE_WIDTH+kernelSize-1);
		if(blockIdx.y == gridDim.y-1){
			for(int row=0; row<kRadius ; row++){
				for(int col=0; col<TILE_WIDTH+kernelSize-1; col++){
					sharedImageData[starting_index + row*(TILE_WIDTH+kernelSize-1)+col] = 0;
				}
			}
		}
		else{
			for(int row=0; row<kRadius ; row++){
				for(int col=0; col<TILE_WIDTH+kernelSize-1; col++){
					if(blockIdx.x==0 && col < kRadius){
						sharedImageData[starting_index + row*(TILE_WIDTH+kernelSize-1)+col] =0;
					}
					else if(blockIdx.x==gridDim.x-1 && col>TILE_WIDTH+kRadius-1){
						sharedImageData[starting_index + row*(TILE_WIDTH+kernelSize-1)+col] =0;
					}
					else{
						sharedImageData[starting_index + row*(TILE_WIDTH+kernelSize-1)+col] = inputImage[threadDataLoc + imageWidth -kRadius + col + row*imageWidth];
					}					
				}
			}
		}
		
	}
	
	//Side apron and image data by thread ID x=0
	if(threadIdx.x==0){
		int row = threadIdx.y + kRadius;
		for(int col =0 ; col <TILE_WIDTH +kernelSize -1; col++){
			if(col < kRadius && blockIdx.x==0){
				sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] = 0;
			}
			else if(col > (TILE_WIDTH + kRadius -1) && blockIdx.x==gridDim.x-1){
				sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] = 0;
			}
			else{
				sharedImageData[row*(TILE_WIDTH+kernelSize-1)+col] = inputImage[threadDataLoc+col-kRadius];
			}
		}
	}
	
	__syncthreads();

	
	int value = 0;
	for (int kRow = -kRadius; kRow <= kRadius; kRow++)
		for (int kCol = -kRadius; kCol <= kRadius; kCol++){
			value += sharedImageData[(threadIdx.x+kRadius) + kCol+ (threadIdx.y + kRadius + kRow)*(TILE_WIDTH+kernelSize-1)] * gpuKernel[(kRadius + kRow)*kernelSize + kRadius + kCol];
		}
	outputImage[threadDataLoc] = value/totalVal;
}


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
	std::cout << "Image Width : " << image.N << std::endl;
/////////////////////////////////////////////////////////////////	 
// Insert CUDA launch code here
/////////////////////////////////////////////////////////////////
	int device;
	int * gpuInputImage, * gpuOutputImage;
	
	checkCudaError(cudaSetDevice(5));

	checkCudaError(cudaGetDevice(&device));
	cudaDeviceProp prop;
	checkCudaError(cudaGetDeviceProperties(&prop, device));
	std::cout << "Device " << device << ": " << prop.name << "\n";
	std::cout << "GPU/SM Cores: " << prop.multiProcessorCount << "\n";
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
	std::cout << "Shared Memory per Block: " << (prop.sharedMemPerBlock>>10) << "\n";
	
	
	checkCudaError(cudaMalloc(&gpuInputImage, image.N * image.N * sizeof(int)));
	std::cout << "Woks" << std::endl;
    checkCudaError(cudaMalloc(&gpuOutputImage, image.N * image.N * sizeof(int)));
    
    checkCudaError(cudaMemcpy(gpuInputImage, image.ptr, image.N * image.N * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(gpuOutputImage, gpuOutput.ptr, image.N * image.N * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(gpuKernel, kernel, k * k * sizeof(int)));
    
    const int GRID_SIZE = ceil((float)image.N/TILE_WIDTH);
    std::cout << GRID_SIZE << std::endl;
	const int CTA_SIZE = TILE_WIDTH;
    
    std::cout << "Image size: " << image.N << "X" << image.N << std::endl << "Threads per block: " << CTA_SIZE << "X" << CTA_SIZE << std::endl << "Blocks: " << GRID_SIZE << "X" << GRID_SIZE << std::endl;
    
    dim3 dimBlock(CTA_SIZE,CTA_SIZE,1);
    dim3 dimGrid(GRID_SIZE,GRID_SIZE);
	
	std::cout << "Running GPU kernel\n\n";
	//use k, kernel, image.N and image.ptr as your inputs
	//copy output to gpuOutput.ptr, data is already allocated
	//make sure the dimensions of the image are the same
	
	int shared_memory = sizeof(int)*(TILE_WIDTH+k-1)*(TILE_WIDTH+k-1);
	printf("Shared Memory: %d\n",shared_memory>>10);
	
	cudaProfilerStart();
	convolutionGPU<<<dimGrid, dimBlock, shared_memory>>>(gpuInputImage, gpuOutputImage, image.N, k, totalVal);
	cudaProfilerStop();
	
	checkCudaError(cudaDeviceSynchronize());
	cudaMemcpy(gpuOutput.ptr, gpuOutputImage, image.N * image.N * sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n",image.ptr[0]);
	
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
				if (row+kRow < 0 || row+kRow >= image.N)
					continue;
				for (int kCol = -kRadius; kCol <= kRadius; kCol++)
				{
					//image bounds check
					if (col+kCol < 0 || col+kCol >= image.N)
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
