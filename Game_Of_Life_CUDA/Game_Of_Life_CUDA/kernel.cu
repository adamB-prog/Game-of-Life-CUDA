
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gif.h"
#include <vector>
#include <cstdint>
#include <stdio.h>

#define X 100
#define Y 200
#define IT 1000
#define R 10
#define output "test.gif"


__device__ bool dev_field[X][Y];

__device__ uint8_t dev_image[X * Y * 4];

bool hst_field[X][Y];
uint8_t dev_image[X * Y * 4];









int main()
{
	int width = X;
	int height = Y;

	int delay = 100;

	


	auto filename = output;
	

	GifWriter g;

	GifBegin(&g, filename, width, height, delay);

	//GifWriteFrame(&g, black.data(), width, height, delay);
	//GifWriteFrame(&g, white.data(), width, height, delay);
	GifEnd(&g);

	return 0;
}