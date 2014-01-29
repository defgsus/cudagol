#include "cudagol.h"

#include <cstdlib>
#include <iostream>

// forward of kernel wrapper
void cudaGol(uint8_t * map, uint8_t * rule,
			 uint16_t width, uint16_t height,
			 uint16_t numSteps, uint16_t maxThreads,
			 bool wrap_edges);


Gol::Gol(size_t w, size_t h)
	:	w(w), h(h), wrap_edges(true)
{
	// init rule

	rule.resize(9*2);
	for (size_t i=0; i<rule.size(); ++i)
		rule[i] = 0;

	rule[  3] = 1;
	rule[9+2] = 1;
	rule[9+3] = 1;

	// init map

	map.resize(w*h);

	randomize();
}

void Gol::randomize(int density, int seed)
{
	if (seed) srand(seed);

	for (size_t i=0; i<map.size(); ++i)
		map[i] = ((rand()%density) == 0)? 1 : 0;
}


void Gol::step_cpu(size_t numIterations)
{
	// neighbour buffer
	Map nmap;
	nmap.resize(w*h);

	for (size_t it=0; it<numIterations; ++it)
	{
		// count neighbours

		if (wrap_edges)
		{
			for (size_t y=0; y<h; ++y)
			for (size_t x=0; x<w; ++x)
			{
				int n = 0;

				const int
					y0 = (y>0)? y-1 : h-1,
					y1 = (y<h-1)? y+1 : 0,
					x0 = (x>0)? x-1 : w-1,
					x1 = (x<w-1)? x+1 : 0;

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
		else // no edge wrap
		{
			for (size_t y=0; y<h; ++y)
			for (size_t x=0; x<w; ++x)
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

		// update state

		for (size_t i=0; i<map.size(); ++i)
		{
			map[i] = rule[ map[i]*9 + nmap[i]];
		}

		//print(map);
	}
}



void Gol::step_cuda(size_t numIterations, size_t maxThreads)
{
	cudaGol(&map[0], &rule[0], w, h, numIterations, maxThreads, wrap_edges);
}





void Gol::print(const Map& map, size_t width, size_t height) const
{
	const size_t W = std::min(width, w);
	const size_t H = std::min(height, h);

	// top line

	for (int i=0; i<W; ++i)
		std::cout << "-";
	std::cout << "\n";

	// print map

	for (size_t j=0; j<H; ++j)
	{
		for (size_t i=0; i<W; ++i)
			if (map[j * w + i])
				std::cout << "*";
			else
				std::cout << ".";

		std::cout << "\n";
	}
}


float Gol::compare(const Map& map1, const Map& map2)
{
	// different size ?
	if (map1.size() != map2.size())
		return 100;

	if (map1.empty()) return 0;

	int d = 0;
	for (size_t i=0; i<map1.size(); ++i)
		d += abs(map1[i] - map2[i]);

	return (float)d / map1.size() * 100.f;
}
