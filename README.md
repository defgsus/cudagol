cudagol
=======

Game Of Life implemented in Cuda

This is a little project, basically to get my head around Cuda.

The main directory contains a Qt Project to compile.
You need to adjust some settings in the .pro file to your system!!
Qt will complain that it does not support builds above the source directory but will build it anyway.
(Who builds into their source dirs??)

There is also a Code::Blocks project to compile the source. 
It will probably not need special care, as long as nvcc is in the search path. 

speed
=====

This program will calculate a game of life map for a few iterations on the CPU and on the GPU to compare speeds. 

I'm far from beeing an expert in Cuda programming, so the way i layout memory (completely linear currently) and the way i designed the threadblocks might not be optimal.

Recent tests:

Speed-Up of single i7 core (with -O2 / 32bit) to Cuda implementation (on a cheap GTX 550 Ti) is about 30x !!!

Strangely: 
on single i7 core (with -O2 / 64bit) to Cuda (absolutely non-cheap Quadro K1000M) (Thinkpad W530) is only 4x. 


