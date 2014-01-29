/**	@file

	@brief GameOfLife Cuda kernel and wrapper call

	@author def.gsus-
	@version 2014/01/28 started
*/
#include <inttypes.h>
#include <iostream>

#include "cudautil.h"

/** same type as in Gol class */
typedef uint8_t Byte;


/** kernel for counting neighbours of each cell (non-wrapping) */
__global__ void kernel_getneighbours(Byte * map, Byte * nmap, int w, int h)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x<w && y<h)
	{
		int n = 0;

		if (x>0)
			n += map[y*w+x-1];
		if (x<w-1)
			n += map[y*w+x+1];

		if (y>0)
		{
			n += map[(y-1)*w+x];
			if (x>0)
				n += map[(y-1)*w+x-1];
			if (x<w-1)
				n += map[(y-1)*w+x+1];
		}
		if (y<h-1)
		{
			n += map[(y+1)*w+x];
			if (x>0)
				n += map[(y+1)*w+x-1];
			if (x<w-1)
				n += map[(y+1)*w+x+1];
		}

		nmap[y*w+x] = n;
	}
}


/** kernel for counting neighbours of each cell (edge-wrapping) */
__global__ void kernel_getneighbours_wrap(Byte * map, Byte * nmap, int w, int h)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x<w && y<h)
	{
		int n = 0;

	#if (1) // the branch version is 50% faster than the modulo version !!!
		const int
			y0 = (y>0)? y-1 : h-1,
			y1 = (y<h-1)? y+1 : 0,
			x0 = (x>0)? x-1 : w-1,
			x1 = (x<w-1)? x+1 : 0;
	#else
		const int
			y0 = (y+h-1) % h,
			y1 = (y  +1) % h,
			x0 = (x+w-1) % w,
			x1 = (x  +1) % w;
	#endif

		n += map[y0*w+x0];
		n += map[y0*w+x ];
		n += map[y0*w+x1];

		n += map[y *w+x0];
		n += map[y *w+x1];

		n += map[y1*w+x0];
		n += map[y1*w+x ];
		n += map[y1*w+x1];

		nmap[y*w+x] = n;
	}
}



/** kernel for setting the new state according to rule */
__global__ void kernel_setnewstate(Byte * map, Byte * nmap, Byte * rule, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i<N)
		map[i] = rule[ map[i]*9 + nmap[i] ];
}



void cudaGol(Byte * map, Byte * rule, uint16_t w, uint16_t h,
			uint16_t numSteps, uint16_t maxThreads, bool wrap)
{
	// setup device memory

	Byte
		* dev_map, 		// cell map
		* dev_n, 		// neighbour count
		* dev_rule;		// rule

	CHECK_CUDA( cudaMalloc((void**)&dev_map,  w * h) );
	CHECK_CUDA( cudaMalloc((void**)&dev_n,    w * h) );
	CHECK_CUDA( cudaMalloc((void**)&dev_rule, 9 * 2) );

	// transfer to / init device data
	CHECK_CUDA( cudaMemcpy((void*)dev_map,  (void*)map, w * h, cudaMemcpyHostToDevice) );
	CHECK_CUDA( cudaMemcpy((void*)dev_rule, (void*)rule, 9 * 2, cudaMemcpyHostToDevice) );
	CHECK_CUDA( cudaMemset((void*)dev_n, 0, w * h) );

	// setup blocksize and number threads

	int numThreadsU = maxThreads;
	int numBlocksU = (w * h + numThreadsU - 1) / numThreadsU;

	// for neighbour counting use 2-dimensional blocksize
	dim3 numThreadsN(sqrt(maxThreads), sqrt(maxThreads));
	dim3 numBlocksN((w+numThreadsN.x-1)/numThreadsN.x, (h+numThreadsN.y-1)/numThreadsN.y);

	std::cout << "running kernel " << numSteps << " times on " << w << "x" << h << " map.\n"
			  << "numBlocksN = " << numBlocksN.x << "x" << numBlocksN.y << ", numThreadsN = " << numThreadsN.x << "x" << numThreadsN.y << "\n"
			  << "numBlocksU = " << numBlocksU << ", numThreadsU = " << numThreadsU << "\n\n";

	// execute

	if (wrap)
	{
		for (uint16_t it = 0; it < numSteps; ++it)
		{
			kernel_getneighbours_wrap<<<numBlocksN, numThreadsN>>>(dev_map, dev_n, w, h);
			kernel_setnewstate<<<numBlocksU, numThreadsU>>>(dev_map, dev_n, dev_rule, w*h);
		}
	}
	else
	{
		for (uint16_t it = 0; it < numSteps; ++it)
		{
			kernel_getneighbours<<<numBlocksN, numThreadsN>>>(dev_map, dev_n, w, h);
			kernel_setnewstate<<<numBlocksU, numThreadsU>>>(dev_map, dev_n, dev_rule, w*h);
		}
	}
	// make sure, kernel calls worked out properly
	CHECK_CUDA( cudaGetLastError() );

	// transfer map back
	CHECK_CUDA( cudaMemcpy((void*)map, (void*)dev_map, w * h, cudaMemcpyDeviceToHost) );

	// free device memory
	CHECK_CUDA( cudaFree(dev_rule) );
	CHECK_CUDA( cudaFree(dev_n) );
	CHECK_CUDA( cudaFree(dev_map) );
}
