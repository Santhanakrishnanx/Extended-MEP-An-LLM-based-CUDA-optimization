
/**
 * jacobi2D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#ifndef _WIN32
#include <sys/resource.h>
#include <sched.h>
#endif

#define POLYBENCH_TIME 1

#include "jacobi2D.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
const int N = JN;
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */


#define RUN_ON_CPU


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
			B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
		}
	}
}


void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
	{
    		for (int i = 1; i < _PB_N - 1; i++)
		{
			for (int j = 1; j < _PB_N - 1; j++)
			{
	  			B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
			}
		}
		
    		for (int i = 1; i < _PB_N-1; i++)
		{
			for (int j = 1; j < _PB_N-1; j++)
			{
	  			A[i][j] = B[i][j];
			}
		}
	}
}




__global__ void runJacobiCUDA_kernel1(int n, float* A, float* B)
{
    // Tile dimensions from blockDim
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;

    // Shared memory tile dimensions including 1-cell halo on each side
    // pitch = tile_width + 2
    __shared__ float sA[(32 + 2) * (32 + 2)]; // max blockDim.x/y assumed <= 32 for occupancy

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i_global = blockIdx.y * tile_height + ty;
    int j_global = blockIdx.x * tile_width + tx;

    int s_width = tile_width + 2;

    int s_i = ty + 1;
    int s_j = tx + 1;

    // Cooperative loading of shared memory tile with halo
    // Each thread loads its own element and selectively halo elements to minimize conditionals

    // Load center element
    if (i_global < n && j_global < n)
        sA[s_i * s_width + s_j] = A[i_global * n + j_global];
    else
        sA[s_i * s_width + s_j] = 0.0f;

    // Load halo elements by threads on tile edges

    // Left halo: threads with tx == 0 load left halo element for their row
    if (tx == 0)
    {
        int j_left = j_global - 1;
        if (i_global < n && j_left >= 0)
            sA[s_i * s_width + (s_j - 1)] = A[i_global * n + j_left];
        else
            sA[s_i * s_width + (s_j - 1)] = 0.0f;
    }

    // Right halo: threads with tx == tile_width - 1 load right halo element for their row
    if (tx == tile_width - 1)
    {
        int j_right = j_global + 1;
        if (i_global < n && j_right < n)
            sA[s_i * s_width + (s_j + 1)] = A[i_global * n + j_right];
        else
            sA[s_i * s_width + (s_j + 1)] = 0.0f;
    }

    // Top halo: threads with ty == 0 load top halo element for their column
    if (ty == 0)
    {
        int i_top = i_global - 1;
        if (i_top >= 0 && j_global < n)
            sA[(s_i - 1) * s_width + s_j] = A[i_top * n + j_global];
        else
            sA[(s_i - 1) * s_width + s_j] = 0.0f;
    }

    // Bottom halo: threads with ty == tile_height - 1 load bottom halo element for their column
    if (ty == tile_height - 1)
    {
        int i_bottom = i_global + 1;
        if (i_bottom < n && j_global < n)
            sA[(s_i + 1) * s_width + s_j] = A[i_bottom * n + j_global];
        else
            sA[(s_i + 1) * s_width + s_j] = 0.0f;
    }

    __syncthreads();

    // Compute output if inside valid range (excluding boundary)
    if (i_global >= 1 && i_global < (n - 1) && j_global >= 1 && j_global < (n - 1))
    {
        // Reuse shared memory neighbors in registers
        float center = sA[s_i * s_width + s_j];
        float left = sA[s_i * s_width + (s_j - 1)];
        float right = sA[s_i * s_width + (s_j + 1)];
        float top = sA[(s_i - 1) * s_width + s_j];
        float bottom = sA[(s_i + 1) * s_width + s_j];

        B[i_global * n + j_global] = 0.2f * (center + left + right + top + bottom);
    }
}






__global__ void runJacobiCUDA_kernel2(int n, float* A, float* B)
{
    constexpr int TILE_DIM_X = 32;
    constexpr int TILE_DIM_Y = 8;

    __shared__ float tile[TILE_DIM_Y][TILE_DIM_X];

    int tile_i = blockIdx.y * TILE_DIM_Y;
    int tile_j = blockIdx.x * TILE_DIM_X;

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    int global_i = tile_i + local_i;
    int global_j = tile_j + local_j;

    // Load tile from B to shared memory with boundary check
    // Coalesced access: threadIdx.x maps to contiguous memory in row-major order
    if (global_i >= 1 && global_i < n - 1 && global_j >= 1 && global_j < n - 1)
    {
        tile[local_i][local_j] = B[global_i * n + global_j];
    }
    else
    {
        tile[local_i][local_j] = 0.0f; // dummy, won't be used
    }

    __syncthreads();

    // Write from shared memory tile to A with boundary check
    if (global_i >= 1 && global_i < n - 1 && global_j >= 1 && global_j < n - 1)
    {
        A[global_i * n + global_j] = tile[local_i][local_j];
    }
}




void compareResults(int n, DATA_TYPE POLYBENCH_2D(a,N,N,n,n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(b,N,N,n,n), DATA_TYPE POLYBENCH_2D(b_outputFromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;   

	// Compare output from CPU and GPU
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
        }
	}
  
	for (i=0; i<n; i++) 
	{
       	for (j=0; j<n; j++) 
		{
        		if (percentDiff(b[i][j], b_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
        			fail++;
        		}
       	}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void runJacobi2DCUDA(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n))
{
	DATA_TYPE* Agpu;
	DATA_TYPE* Bgpu;

	cudaMalloc(&Agpu, n * n * sizeof(DATA_TYPE));
	cudaMalloc(&Bgpu, n * n * sizeof(DATA_TYPE));
	cudaMemcpy(Agpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(Bgpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

	// Kernel 1 block and grid configuration
	dim3 block1(32, 8);
	dim3 grid1((n + block1.x - 1) / block1.x, (n + block1.y - 1) / block1.y);

	// Kernel 2 block and grid configuration
	dim3 block2(32, 8);
	dim3 grid2((n + block2.x - 1) / block2.x, (n + block2.y - 1) / block2.y);
	
	/* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < tsteps; t++)
	{
		runJacobiCUDA_kernel1<<<grid1,block1>>>(n, Agpu, Bgpu);
		cudaDeviceSynchronize();
		runJacobiCUDA_kernel2<<<grid2,block2>>>(n, Agpu, Bgpu);
		cudaDeviceSynchronize();
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
  	polybench_print_instruments;
	
	cudaMemcpy(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);

	cudaFree(Agpu);
	cudaFree(Bgpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

	POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	runJacobi2DCUDA(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	  	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(n, POLYBENCH_ARRAY(a_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(a_outputFromGpu);
	POLYBENCH_FREE_ARRAY(b);
	POLYBENCH_FREE_ARRAY(b_outputFromGpu);

	return 0;
}

#include "../../common/polybench.c"

