/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#endif


#define POLYBENCH_TIME 1

#include "2mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0
#define TILE_SIZE 16

#define RUN_ON_CPU


void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), 
		DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj), 
		DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < nl; i++)
	{
		for (j = 0; j < nj; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			if (percentDiff(D[i][j], D_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}




__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, float alpha, float beta, float *tmp, float *A, float *B)
{
    // Tile indices
    int tile_i = blockIdx.y;
    int tile_j = blockIdx.x;

    // Thread indices within tile
    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // Global row and column indices
    int i = tile_i * TILE_SIZE + local_i;
    int j = tile_j * TILE_SIZE + local_j;

    // Declare shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles of dimension nk
    int numTiles = (nk + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)
    {
        // Compute k index for this tile and thread
        int k = t * TILE_SIZE + local_j; // for loading A: col index in A tile
        int k_b = t * TILE_SIZE + local_i; // for loading B: row index in B tile

        // Load A[i,k] into shared memory if in bounds
        if (i < ni && k < nk)
            As[local_i][local_j] = A[i * nk + k];
        else
            As[local_i][local_j] = 0.0f;

        // Load B[k,j] into shared memory if in bounds
        if (k_b < nk && j < nj)
            Bs[local_i][local_j] = B[k_b * nj + j];
        else
            Bs[local_i][local_j] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile
        // Unroll loop by 4 when possible
        int limit = TILE_SIZE - (TILE_SIZE % 4);
        int kk = 0;
        for (; kk < limit; kk += 4)
        {
            sum += alpha * (As[local_i][kk + 0] * Bs[kk + 0][local_j]
                          + As[local_i][kk + 1] * Bs[kk + 1][local_j]
                          + As[local_i][kk + 2] * Bs[kk + 2][local_j]
                          + As[local_i][kk + 3] * Bs[kk + 3][local_j]);
        }
        for (; kk < TILE_SIZE; kk++)
        {
            sum += alpha * As[local_i][kk] * Bs[kk][local_j];
        }

        __syncthreads();
    }

    if (i < ni && j < nj)
    {
        tmp[i * nj + j] = sum;
    }
}






__global__ void mm2_kernel2(int ni, int nj, int nk, int nl, float alpha, float beta, float *tmp, float *C, float *D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < nl)
    {
        float sum = 0.0f;
        int tileCount = (nj + TILE_SIZE - 1) / TILE_SIZE;

        for (int tile = 0; tile < tileCount; tile++)
        {
            int k_start = tile * TILE_SIZE;
            int k_end = k_start + TILE_SIZE;
            if (k_end > nj) k_end = nj;

            // Declare shared memory only if blockDim == TILE_SIZE
            // Otherwise, rely on registers only
            __shared__ float s_tmp[TILE_SIZE][TILE_SIZE];
            __shared__ float s_C[TILE_SIZE][TILE_SIZE];

            bool use_shared = (blockDim.x == TILE_SIZE) && (blockDim.y == TILE_SIZE);

            float tmp_val = 0.0f;
            float C_val = 0.0f;

            if (use_shared)
            {
                int local_i = threadIdx.y;
                int local_j = threadIdx.x;

                // Load tmp[i, k] into shared memory
                int k_idx = k_start + local_j;
                if (i < ni && k_idx < nj)
                    s_tmp[local_i][local_j] = tmp[i * nj + k_idx];
                else
                    s_tmp[local_i][local_j] = 0.0f;

                // Load C[k, j] into shared memory
                int k_idx2 = k_start + local_i;
                if (k_idx2 < nj && j < nl)
                    s_C[local_i][local_j] = C[k_idx2 * nl + j];
                else
                    s_C[local_i][local_j] = 0.0f;

                __syncthreads();

                int valid_k = k_end - k_start;
                for (int kk = 0; kk < valid_k; kk++)
                {
                    sum += s_tmp[threadIdx.y][kk] * s_C[kk][threadIdx.x];
                }
                __syncthreads();
            }
            else
            {
                for (int k = k_start; k < k_end; k++)
                {
                    tmp_val = tmp[i * nj + k];
                    C_val = C[k * nl + j];
                    sum += tmp_val * C_val;
                }
            }
        }
        D[i * nl + j] = D[i * nl + j] * beta + sum;
    }
}




void mm2_cpu(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
	int i, j, k;
	
	/* D := alpha*A*B*C + beta*D */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			tmp[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				tmp[i][j] += alpha * A[i][k] * B[k][j];
			}
		}
	}

	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			D[i][j] *= beta;
			for (k = 0; k < _PB_NJ; ++k)
			{
				D[i][j] += tmp[i][k] * C[k][j];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


void mm2Cuda(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), 
	DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), 
	DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu,NI,NL,ni,nl))
{
	DATA_TYPE *tmp_gpu;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;

	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NL * NJ);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NI * NL);
	
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NL * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
		
	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid1((size_t)ceil( ((float)NJ) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );
	dim3 grid2((size_t)ceil( ((float)NL) / ((float)block.x) ), (size_t)ceil( ((float)NI) / ((float)block.y)) );

	/* Start timer. */
  	polybench_start_instruments;

	mm2_kernel1<<<grid1,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, A_gpu, B_gpu);
	cudaThreadSynchronize();
	mm2_kernel2<<<grid2,block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu);
	cudaThreadSynchronize();

	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	cudaMemcpy(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);

	cudaFree(tmp_gpu);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
	POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
	POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu,DATA_TYPE,NI,NL,ni,nl);
	
	/* Initialize array(s). */
  	init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
	GPU_argv_init();

	mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), 
		POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nl, POLYBENCH_ARRAY(D_outputFromGpu));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(D_outputFromGpu);

  	return 0;
}

#include "../../common/polybench.c"
