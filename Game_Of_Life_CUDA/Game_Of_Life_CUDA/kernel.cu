
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gif.h"
#include <vector>
#include <cstdint>
#include <stdio.h>

#define X 100
#define Y 200
#define T 100
#define IT 500
#define R 3
#define output "test.gif"


__device__ bool dev_field[X][Y];

__device__ int dev_neighbours[X][Y];

__device__ uint8_t dev_image[X * Y * 4];

__device__ int dev_result;

int result = 0;

bool hst_field[X][Y];
uint8_t hst_image[X * Y * 4];

__global__ void ResetNeighbourTable()
{
	if (blockIdx.x == 0 && blockIdx.y == 0)
	{
		dev_result = 0;
	}
	dev_neighbours[blockIdx.x][blockIdx.y] = 0;
	
}
__global__ void CalculateCellNeighbours()
{
	__shared__ bool shr_neighbours[3][3];

	__shared__ bool edge;
	
	__shared__ uint8_t minX, maxX, minY, maxY;

	minX = 0;
	maxX = 2;
	minY = 0;
	maxY = 2;


	//SETUP
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		//std::fill(dev_neighbours, dev_neighbours + X * Y, 0);
		edge = false;

		if (blockIdx.x == 0)
		{
			shr_neighbours[0][0] = 0;
			shr_neighbours[1][0] = 0;
			shr_neighbours[2][0] = 0;
			minX = 1;
			edge = true;
			
		}
		if (blockIdx.y == 0)
		{
			shr_neighbours[0][0] = 0;
			shr_neighbours[0][1] = 0;
			shr_neighbours[0][2] = 0;
			edge = true;
			minY = 1;
		}
		if (blockIdx.x == X)
		{
			shr_neighbours[2][0] = 0;
			shr_neighbours[2][1] = 0;
			shr_neighbours[2][2] = 0;
			edge = true;
			maxY = 1;
			
		}
		if (blockIdx.y == Y)
		{
			shr_neighbours[0][2] = 0;
			shr_neighbours[1][2] = 0;
			shr_neighbours[2][2] = 0;
			edge = true;
			
			maxX = 1;
		}



	}
	//LOADING
	__syncthreads();
	if (threadIdx.x >= minX && threadIdx.x <= maxX && threadIdx.y >= minY && threadIdx.y <= maxY)
	{
		shr_neighbours[threadIdx.x][threadIdx.y] = dev_field[blockIdx.x - 1 + threadIdx.x][blockIdx.y - 1 + threadIdx.y];
			
	}
	__syncthreads();

	//NO SELFREPORT
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		shr_neighbours[1][1] = 0;
	}
	
	
	
	__syncthreads();


	atomicAdd(&dev_neighbours[blockIdx.x][blockIdx.y], shr_neighbours[threadIdx.x][threadIdx.y]);

	__syncthreads();

	if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				printf("%i", dev_neighbours[i][j]);
			}
			printf("\n");
		}

		printf("---------------\n");
	}

	

}

__global__ void SetNewField()
{
	__shared__ bool shr_alive; 
	__shared__ int shr_neighbours;
	
	if (threadIdx.x == 0 &&  threadIdx.y == 0)
	{
		shr_alive = dev_field[blockIdx.x][blockIdx.y];
		shr_neighbours = dev_neighbours[blockIdx.x][blockIdx.y];
	}
	__syncthreads();
	if (shr_alive && (shr_neighbours < 1 || shr_neighbours > 3))
	{
		dev_field[blockIdx.x][blockIdx.y] = false;

	}

	else if (!shr_alive && shr_neighbours == 3)
	{
		dev_field[blockIdx.x][blockIdx.y] = true;
	}
	atomicAdd(&dev_result, dev_field[blockIdx.x][blockIdx.y]);
	__syncthreads();
}




int main()
{
	int width = X;
	int height = Y;

	int delay = T;
	/*
	hst_image[0] = 255;
	hst_image[1] = 255;
	hst_image[2] = 255;
	hst_image[3] = 255;
	hst_image[4] = 255;
	hst_image[5] = 255;
	hst_image[6] = 255;
	hst_image[7] = 255;
	hst_image[8] = 255;
	hst_image[9] = 255;
	hst_image[10] = 255;
	hst_image[11] = 255;
	hst_image[X * 4 + 0] = 255;
	hst_image[X * 4 + 1] = 255;
	hst_image[X * 4 + 2] = 255;
	hst_image[X * 4 + 3] = 255;
	hst_image[X * 4 + 4] = 255;
	hst_image[X * 4 + 5] = 255;
	hst_image[X * 4 + 6] = 255;
	hst_image[X * 4 + 7] = 255;
	hst_image[X * 4 + 8] = 255;
	hst_image[X * 4 + 9] = 255;
	hst_image[X * 4 + 10] = 255;
	hst_image[X * 4 + 11] = 255;
	hst_image[X * 4 * 2 + 0] = 255;
	hst_image[X * 4 * 2 + 1] = 255;
	hst_image[X * 4 * 2 + 2] = 255;
	hst_image[X * 4 * 2 + 3] = 255;
	hst_image[X * 4 * 2 + 4] = 255;
	hst_image[X * 4 * 2 + 5] = 255;
	hst_image[X * 4 * 2 + 6] = 255;
	hst_image[X * 4 * 2 + 7] = 255;
	hst_image[X * 4 * 2 + 8] = 255;
	hst_image[X * 4 * 2 + 9] = 255;
	hst_image[X * 4 * 2 + 10] = 255;
	hst_image[X * 4 * 2 + 11] = 255;
	
	*/
	auto filename = output;
	
	

	GifWriter g;

	GifBegin(&g, filename, width, height, delay);

	hst_field[0][0] = true;
	hst_field[1][0] = true;
	hst_field[0][1] = true;
	hst_field[2][2] = true;


	cudaMemcpyToSymbol(dev_field, hst_field, X * Y * sizeof(bool));

	
	for (size_t i = 0; i < 10000; i++)
	{
		ResetNeighbourTable << <dim3(X,Y), 1 >> > ();
		cudaDeviceSynchronize();
		CalculateCellNeighbours << <dim3(X,Y), dim3(3, 3) >> > ();

		cudaDeviceSynchronize();
		SetNewField << <dim3(X,Y), 1 >> > ();

		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol(&result, dev_result, sizeof(int));
		cudaMemcpyFromSymbol(hst_field, dev_field, X * Y * sizeof(bool));
		if (result > 4)
		{
			bool asd = true;
		}

		printf("%i, trues=%i\n", i, result);
	}


	

	

	GifWriteFrame(&g, hst_image, width, height, delay);
	//GifWriteFrame(&g, white.data(), width, height, delay);
	GifEnd(&g);

	return 0;
}