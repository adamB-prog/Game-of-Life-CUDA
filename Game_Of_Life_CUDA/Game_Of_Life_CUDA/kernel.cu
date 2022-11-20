
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gif.h"
#include <vector>
#include <cstdint>
#include <stdio.h>

#define X 100
#define Y 200
#define T 100
#define IT 10
#define R 3
#define output "test.gif"


__device__ bool dev_field[X][Y];

__device__ bool dev_newField[X][Y];

__device__ int dev_neighbours[X][Y];

__device__ uint8_t dev_image[X * Y * 4];

__device__ int dev_result;

int result = 0;

bool hst_field[X][Y];
uint8_t hst_image[X * Y * 4];

__global__ void ResetNeighbourTable()
{
	if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0)
	{
		//printf("\nRESET\n");
	}
	if (blockIdx.x == 0 && blockIdx.y == 0)
	{
		dev_result = 0;
	}
	dev_neighbours[blockIdx.x][blockIdx.y] = 0;

	__syncthreads();
	
}
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
		if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0)
		{
			//printf("\nCALC\n");
		}

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

	/*if (blockIdx.x == 10 && blockIdx.y == 6 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				printf("%i", dev_neighbours[blockIdx.x - 1 + i][blockIdx.y - 1 + j]);
			}
			printf("\n");
		}

		printf("---------------\n");
	}*/

	

}

__global__ void SetNewField()
{
	__shared__ bool shr_alive; 
	__shared__ int shr_neighbours;
	
	
	
	shr_alive = dev_field[blockIdx.x][blockIdx.y];
	shr_neighbours = dev_neighbours[blockIdx.x][blockIdx.y];

	//printf("\n BlockId(%i,%i), alive=%i, neighbours=%i", blockIdx.x, blockIdx.y, shr_alive, shr_neighbours);

	
	
	if (shr_alive && (shr_neighbours < 2 || shr_neighbours > 3))
	{

		//printf("\nexe false %i %i", blockIdx.x, blockIdx.y);
		dev_newField[blockIdx.x][blockIdx.y] = false;
	}

	else if (!shr_alive && shr_neighbours == 3)
	{
		//printf("\nexe true %i %i", blockIdx.x, blockIdx.y);
		dev_newField[blockIdx.x][blockIdx.y] = true;
	}

	else
	{
		dev_newField[blockIdx.x][blockIdx.y] = dev_field[blockIdx.x][blockIdx.y];
	}
	atomicAdd(&dev_result, dev_field[blockIdx.x][blockIdx.y]);
	
}

__global__ void CopyNewToOld()
{
	dev_field[blockIdx.x][blockIdx.y] = dev_newField[blockIdx.x][blockIdx.y];
}

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


	//starting field copy
	cudaMemcpyToSymbol(dev_field, hst_field, X * Y * sizeof(bool));

	
	for (size_t i = 0; i < IT; i++)
	{
		ResetNeighbourTable << <dim3(X,Y), 1 >> > ();
		

		CalculateCellNeighbours << <dim3(X,Y), dim3(3, 3) >> > ();

		


		SetNewField << <dim3(X,Y), 1 >> > ();
		MakeImage << <dim3(X, Y), 1 >> > ();

		
		CopyNewToOld << < dim3(X, Y), 1 >> > ();
		
		cudaMemcpyFromSymbol(&result, dev_result, sizeof(int));
		cudaMemcpyFromSymbol(hst_field, dev_field, X * Y * sizeof(bool));

		cudaMemcpyFromSymbol(hst_image, dev_image, X * Y * 4 * sizeof(uint8_t));
		GifWriteFrame(&g, hst_image, width, height, delay);

		//printf("\nIteration: %i, trues=%i\n", i, result);
		if (result == 0)
		{
			break;
		}
		
	}


	

	

	//GifWriteFrame(&g, hst_image, width, height, delay);
	//GifWriteFrame(&g, white.data(), width, height, delay);
	GifEnd(&g);

	return 0;
}