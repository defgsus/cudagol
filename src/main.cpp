#include <iostream>

#include "cudautil.h"
#include "cudagol.h"

/** get system time in seconds */
double sysTime()
{
	timespec cls;
	clock_gettime(CLOCK_MONOTONIC, &cls);
	// second + nanoseconds
	return cls.tv_sec + 0.000000001 * cls.tv_nsec;
}

/** class to messure passed time */
class Messure
{
	public:

	Messure() { start(); }

	/** restart counter */
	void start() { last_time = sysTime(); }

	/** return elapsed time */
	double elapsed() { return sysTime() - last_time; }

	private:

	double last_time;
};



/** function to print device properties */
void printDeviceProps(int dev)
{
	cudaDeviceProp p;
	CHECK_CUDA( cudaGetDeviceProperties(&p, dev) );

	#define CDP_PRINT( entry__ ) \
		std::cout << std::setw(32) << #entry__ << " : " << p. entry__ << "\n";

	CDP_PRINT( name )
	CDP_PRINT( major )
	CDP_PRINT( minor )
	CDP_PRINT( asyncEngineCount )
	CDP_PRINT( canMapHostMemory )
	CDP_PRINT( clockRate )
	CDP_PRINT( computeMode )
	CDP_PRINT( concurrentKernels )
	CDP_PRINT( deviceOverlap )
	CDP_PRINT( ECCEnabled )
	CDP_PRINT( integrated )
	CDP_PRINT( kernelExecTimeoutEnabled )
	CDP_PRINT( l2CacheSize )
	CDP_PRINT( maxGridSize[0] )
	CDP_PRINT( maxGridSize[1] )
	CDP_PRINT( maxGridSize[2] )
    CDP_PRINT( maxTexture1D )
	CDP_PRINT( maxTexture1DLayered[0] )
	CDP_PRINT( maxTexture1DLayered[1] )
	CDP_PRINT( maxTexture2D[0] )
	CDP_PRINT( maxTexture2D[1] )
	CDP_PRINT( maxTexture2DLayered[0] )
	CDP_PRINT( maxTexture2DLayered[1] )
	CDP_PRINT( maxTexture2DLayered[2] )
	CDP_PRINT( maxTexture3D[0] )
	CDP_PRINT( maxTexture3D[1] )
	CDP_PRINT( maxTexture3D[2] )
	CDP_PRINT( maxThreadsDim[0] )
	CDP_PRINT( maxThreadsDim[1] )
	CDP_PRINT( maxThreadsDim[2] )
	CDP_PRINT( maxThreadsPerBlock )
	CDP_PRINT( maxThreadsPerMultiProcessor )
	CDP_PRINT( memoryBusWidth )
	CDP_PRINT( memoryClockRate )
	CDP_PRINT( memPitch )
	CDP_PRINT( multiProcessorCount )
	CDP_PRINT( pciBusID )
	CDP_PRINT( pciDeviceID )
	CDP_PRINT( pciDomainID )
	CDP_PRINT( regsPerBlock )
	CDP_PRINT( sharedMemPerBlock )
	CDP_PRINT( surfaceAlignment )
	CDP_PRINT( tccDriver )
	CDP_PRINT( textureAlignment )
	CDP_PRINT( totalConstMem )
	CDP_PRINT( totalGlobalMem )
	CDP_PRINT( unifiedAddressing )
	CDP_PRINT( warpSize )

/*  well, those seem to be missing in my current cuda headers?

    CDP_PRINT( maxSurface1D )
    CDP_PRINT( maxSurface1DLayered[0] )
    CDP_PRINT( maxSurface1DLayered[1] )
    CDP_PRINT( maxSurface2D[0] )
    CDP_PRINT( maxSurface2D[1] )
    CDP_PRINT( maxSurface2DLayered[0] )
    CDP_PRINT( maxSurface2DLayered[1] )
    CDP_PRINT( maxSurface2DLayered[2] )
    CDP_PRINT( maxSurface3D[0] )
    CDP_PRINT( maxSurface3D[1] )
    CDP_PRINT( maxSurface3D[2] )
    CDP_PRINT( maxSurfaceCubemap )
    CDP_PRINT( maxSurfaceCubemapLayered[0] )
    CDP_PRINT( maxSurfaceCubemapLayered[1] )
    CDP_PRINT( texturePitchAlignment )
    CDP_PRINT( maxTextureCubemap )
    CDP_PRINT( maxTextureCubemapLayered[0] )
    CDP_PRINT( maxTextureCubemapLayered[1] )
    CDP_PRINT( maxTexture2DLinear[0] )
    CDP_PRINT( maxTexture2DLinear[1] )
    CDP_PRINT( maxTexture2DLinear[2] )
    CDP_PRINT( maxTexture2DGather[0] )
    CDP_PRINT( maxTexture2DGather[1] )
    CDP_PRINT( maxTexture1DLinear )
*/
	#undef CDP_PRINT
}



// macro for printing elapsed time and iterations/sec
#define PRINT_TIME(t) \
    std::cout << "took " << t << " seconds, = " \
              << (int)((double)numIter / t) << " iterations per second.\n";


// test cuda gol with certain number of threads per block
void testCuda(Gol& gol, const Gol::Map& init_map, const Gol::Map& reference_map,
                        double cpu_time, size_t numIter, size_t numThreads)
{
    Messure m;

    std::cout << "\n-- " << numThreads << " cuda threads:\n";

    // re-init gol map
    gol.map = init_map;

    // run the cuda kernel
    m.start();
    gol.step_cuda(numIter, numThreads);
    double cuda_time = m.elapsed();

    // print time and ratio

    PRINT_TIME(cuda_time);
    std::cout << "\nCuda speed-up = " << cpu_time / cuda_time << "x\n";

    // compare resulting map

    float d = Gol::compare(gol.map, reference_map);
    if (d != 0)
    {
        std::cout << "UH-OH! Cuda calculated map differs from CPU map by " << d << "%\n";
        std::cout << "cpu:\n";
        gol.print(reference_map);
        std::cout << "cuda:\n";
        gol.print(gol.map);
    }
}


/** compare CPU versus GPU */
void compareSpeed(size_t width, size_t height, size_t numIter)
{

	std::cout << "\ncomparing cpu/gpu speed on " << width << "x" << height
              << " map with " << numIter << " iterations.\n\n";

	// create gol class
	Gol gol(width, height);

	// store initial state of map
	Gol::Map backup = gol.map;
	// used to compare cpu and gpu output
	Gol::Map reference;

	// cpu run

	std::cout << "------ CPU -----\n";

    Messure m;
	gol.step_cpu(numIter);
	double cpu_time = m.elapsed();
	PRINT_TIME(cpu_time);

	// keep result
	reference = gol.map;

	std::cout << "\n------ CUDA ------\n";

    // test different thread numbers
    testCuda(gol, backup, reference, cpu_time, numIter, 64);
    testCuda(gol, backup, reference, cpu_time, numIter, 128);
    testCuda(gol, backup, reference, cpu_time, numIter, 256);
    testCuda(gol, backup, reference, cpu_time, numIter, 512);
    testCuda(gol, backup, reference, cpu_time, numIter, 768);
    testCuda(gol, backup, reference, cpu_time, numIter, 1024);



    // print a bit of the map to show it's actually doing something
    std::cout << "\na piece of the final map:\n";
    gol.print(gol.map, 70, 8);

}

#undef PRINT_TIME

int main()
{
	std::cout << "Cuda GameOfLife\n";

	// Cuda devices are initialized automatically
	// You could do a cudaSetDevice() to choose among different devices, however

    // print some device properties
	printDeviceProps(0);

	// compare CPU vs GPU gameoflife implementation
	compareSpeed(1024, 1024, 200);

    return 0;
}
