﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gif.h"
#include <vector>
#include <cstdint>
#include <stdio.h>


//Values
#define X 1000
#define Y 2000
#define T 5
#define IT 100
#define output "test.gif"

//Device variables
__device__ bool dev_field[X][Y];

__device__ bool dev_newField[X][Y];

__device__ int dev_neighbours[X][Y];

__device__ uint8_t dev_image[X * Y * 4];



//Host variables


bool hst_field[X][Y];
uint8_t hst_image[X * Y * 4];



/*
	Reset NeighbourTable with zeros
	And the living cell counter
*/
__global__ void ResetNeighbourTable()
{
	dev_neighbours[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y] = 0;
}

/*
	Neighbour calculation
*/


__global__ void CalculateCellNeighbours()
{
	
	__shared__ bool shr_neighbours[3][3];

	
	__shared__ uint8_t minX, maxX, minY, maxY;

	minX = 0;
	maxX = 2;
	minY = 0;
	maxY = 2;


	//SETUP
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{

		if (blockIdx.x == 0)
		{
			shr_neighbours[0][0] = 0;
			shr_neighbours[0][1] = 0;
			shr_neighbours[0][2] = 0;
			
			minX = 1;
		
		}
		if (blockIdx.y == 0)
		{
			shr_neighbours[0][0] = 0;
			shr_neighbours[1][0] = 0;
			shr_neighbours[2][0] = 0;
			
			minY = 1;
		}
		if (blockIdx.x == X)
		{
			shr_neighbours[0][2] = 0;
			shr_neighbours[1][2] = 0;
			shr_neighbours[2][2] = 0;

			maxX = 1;
			
		}
		if (blockIdx.y == Y)
		{
			
			shr_neighbours[2][0] = 0;
			shr_neighbours[2][1] = 0;
			shr_neighbours[2][2] = 0;
			maxY = 1;
		}



	}

	__syncthreads();
	//LOADING
	if (threadIdx.x >= minX && threadIdx.x <= maxX && threadIdx.y >= minY && threadIdx.y <= maxY)
	{
		shr_neighbours[threadIdx.x][threadIdx.y] = dev_field[blockIdx.x - 1 + threadIdx.x][blockIdx.y - 1 + threadIdx.y];
	}
	//__syncthreads();

	//NO SELFREPORT
	if (threadIdx.x == 1 && threadIdx.y == 1)
	{
		shr_neighbours[1][1] = 0;
	}
	
	
	

	//SUM NEIGHBOURS(bool true = int 1)
	atomicAdd(&dev_neighbours[blockIdx.x][blockIdx.y], shr_neighbours[threadIdx.x][threadIdx.y]);

	

	

}
/*
	After Calculating the Neighbours table, then using it by the rules.
*/
__global__ void SetNewField()
{
	__shared__ bool shr_alive[32][32];
	__shared__ int shr_neighbours[32][32];
	
	
	
	shr_alive[threadIdx.x][threadIdx.y] = dev_field[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y];
	shr_neighbours[threadIdx.x][threadIdx.y] = dev_neighbours[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y];

	

	//Dying condition
	if (shr_alive[threadIdx.x][threadIdx.y] && (shr_neighbours[threadIdx.x][threadIdx.y] < 2 || shr_neighbours[threadIdx.x][threadIdx.y] > 3))
	{
		dev_newField[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y] = false;
	}
	//Revive condition
	else if (!shr_alive[threadIdx.x][threadIdx.y] && shr_neighbours[threadIdx.x][threadIdx.y] == 3)
	{

		dev_newField[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y] = true;
	}
	//Otherwise just copy
	else
	{
		//dev_newField[blockIdx.x][blockIdx.y] = dev_field[blockIdx.x][blockIdx.y];
		dev_newField[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y] = shr_alive[threadIdx.x][threadIdx.y];
	}
	//Counting the living cells
	
	
}

/*
	Copy Method
*/
__global__ void CopyNewToOld()
{
	
	dev_field[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y] = dev_newField[32 * blockIdx.x + threadIdx.x][32 * blockIdx.y + threadIdx.y];

}
/*
	Copy Convert Method
*/
__global__ void MakeImage()
{
	dev_image[4 * Y * blockIdx.x + blockIdx.y * 4 + 0] = dev_newField[blockIdx.x][blockIdx.y] * 255;
	dev_image[4 * Y * blockIdx.x + blockIdx.y * 4 + 1] = dev_newField[blockIdx.x][blockIdx.y] * 255;
	dev_image[4 * Y * blockIdx.x + blockIdx.y * 4 + 2] = dev_newField[blockIdx.x][blockIdx.y] * 255;
	dev_image[4 * Y * blockIdx.x + blockIdx.y * 4 + 3] = dev_newField[blockIdx.x][blockIdx.y] * 255;
}


int main()
{
	int width = Y;
	int height = X;

	int delay = T;
	
	auto filename = output;
	
	

	GifWriter g;

	GifBegin(&g, filename, width, height, delay);

	hst_field[0][0] = true;
	hst_field[0][1] = true;
	hst_field[1][0] = true;

	//spin
	hst_field[5][5] = true;
	hst_field[5][6] = true;
	hst_field[5][7] = true;

	//go
	hst_field[10][10] = true;
	hst_field[11][11] = true;
	hst_field[11][12] = true;
	hst_field[12][10] = true;
	hst_field[12][11] = true;


	//starting field copy
	cudaMemcpyToSymbol(dev_field, hst_field, X * Y * sizeof(bool));
	
	
	for (size_t i = 0; i < IT; i++)
	{
		ResetNeighbourTable << <dim3(X / 32 + 1, Y / 32 + 1), dim3(32, 32) >> > ();

		CalculateCellNeighbours << <dim3(X,Y), dim3(3, 3) >> > ();

		SetNewField << <dim3(X / 32 + 1, Y / 32 + 1), dim3(32, 32) >> > ();

		MakeImage << <dim3(X, Y), 1 >> > ();
		
		CopyNewToOld << < dim3(X / 32 + 1,Y / 32 + 1), dim3(32,32) >> > ();
		

		cudaMemcpyFromSymbol(hst_image, dev_image, X * Y * 4 * sizeof(uint8_t));
		cudaMemcpyFromSymbol(hst_field, dev_field, X * Y * sizeof(bool));
		GifWriteFrame(&g, hst_image, width, height, delay);
		
	}


	

	

	GifEnd(&g);

	return 0;
}