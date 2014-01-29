/**	@file

	@brief Cuda utility functions

	@author def.gsus-
	@version 2014/01/28 started

	<p>This header can be included by host and device code.
	It basically helps with error-checking.</p>
*/

#ifndef CUDAUTIL_H_INCLUDED
#define CUDAUTIL_H_INCLUDED

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>


#ifndef CHECK_CUDA
	/** Macro for checking for cuda errors.
		This ungracefully terminates program on execution.
		Define CHECK_CUDA before including this header to change behaviour */
	#define CHECK_CUDA( command__ ) \
	{ \
		cudaError_t err = command__; \
		if (err != cudaSuccess) \
		{ \
			std::cerr << "Cuda Error: " << cudaGetErrorString(err) \
					  << "\nfor command '" #command__ "'\n"; \
			exit(-err); \
		} \
	}
#endif




#endif // CUDAUTIL_H_INCLUDED
