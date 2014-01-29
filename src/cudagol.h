/**	@file

	@brief GameOfLife in Cuda

	@author def.gsus-
	@version 2014/01/28 started
*/
#ifndef CUDAGOL_H_INCLUDED
#define CUDAGOL_H_INCLUDED

#include <cstddef>
#include <inttypes.h>
#include <vector>

class Gol
{
	public:

	// -------------- types ----------------

	/** Basic cell state type. Needs to match the type used in cuda kernel! */
	typedef uint8_t Byte;
	/** Consecutive memory type */
	typedef std::vector<Byte> Map;

	// ------------ public member ----------

	/** Cellular rule. [0,8] is birth, [9,17] is survife */
	Map rule,
	/** The cell map. [h][w] */
		map;
	/** Dimension of the map */
	size_t w, h;

	/** TRUE for toroid world. */
	bool wrap_edges;

	/** Constructor for a particular size.
		Initializes rule with Conway's game of life,
		and randomizes map. */
	Gol(size_t w, size_t h);

	/** randomize map.
		@p density defines the average number of cells that are empty.
		If @p seed is 0, the current random seed is used, otherwise it
		will be set to this value. */
	void randomize(int density = 5, int seed = 0);

	/** run @p numIterations of the automaton on cpu. */
	void step_cpu(size_t numIterations);

	/** run @p numIterations of the automaton on gpu.
		@p maxThreads defines the number of threads per block.
		256 seems to be a good value for my card, but you need to test this probably. */
	void step_cuda(size_t numIterations, size_t maxThreads);

	/** print (upper-left part of) a map to console. */
	void print(const Map& map, size_t screen_width=70, size_t screen_height=30) const;

	/** Compare two maps and return difference in percent */
	static float compare(const Map& map1, const Map& map2);
};

#endif // CUDAGOL_H_INCLUDED
