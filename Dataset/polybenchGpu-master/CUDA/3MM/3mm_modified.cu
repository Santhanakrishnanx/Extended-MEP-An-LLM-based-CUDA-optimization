
/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3mm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0
#define TILE_SIZE 16

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU


void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), 
		DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm), DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
	int i, j;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}
  
	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}
  
	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nm; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
		}
	}
  
	for (i = 0; i < nm; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			if (percentDiff(G[i][j],G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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

	


__global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, float *A, float *B, float *E)
{
    // Configuration:
    // blockDim.x = 32 (warp size)
    // blockDim.y = 8
    // Each thread computes 4 output elements in j dimension (columns)
    // Total output columns per block = 32 * 4 = 128
    // Total output rows per block = 8

    const int BLOCK_COLS = blockDim.x * 4; // 128
    const int BLOCK_ROWS = blockDim.y;     // 8

    int base_i = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    int base_j = blockIdx.x * BLOCK_COLS + threadIdx.x * 4;

    if (base_i >= ni) return;

    // Partial sums in registers for 4 output elements per thread
    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;

    float *a_row = &A[base_i * nk];

    // Process k in chunks of 2 for vectorized loads of B
    int k = 0;

    // We will load B elements in a coalesced manner using float4 loads when possible
    // B is row-major: B[k * nj + j], so for fixed k, j varies contiguously
    // For each k, load 4 contiguous B elements at once (if within bounds)

    for (; k + 1 < nk; k += 2)
    {
        // Load two A elements from the same row (base_i)
        float a_val0 = a_row[k];
        float a_val1 = a_row[k + 1];

        // Pointers to B rows k and k+1
        float *b_row0 = &B[k * nj];
        float *b_row1 = &B[(k + 1) * nj];

        // Load 4 contiguous B elements for k-th row
        float4 b_vals0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 b_vals1 = make_float4(0.f, 0.f, 0.f, 0.f);

        // Check boundary for base_j + 3
        if (base_j + 3 < nj)
        {
            // Safe to load float4
            b_vals0 = *((float4 *)(b_row0 + base_j));
            b_vals1 = *((float4 *)(b_row1 + base_j));
        }
        else
        {
            // Boundary check per element
            float b0_0 = (base_j + 0 < nj) ? b_row0[base_j + 0] : 0.f;
            float b0_1 = (base_j + 1 < nj) ? b_row0[base_j + 1] : 0.f;
            float b0_2 = (base_j + 2 < nj) ? b_row0[base_j + 2] : 0.f;
            float b0_3 = (base_j + 3 < nj) ? b_row0[base_j + 3] : 0.f;
            b_vals0 = make_float4(b0_0, b0_1, b0_2, b0_3);

            float b1_0 = (base_j + 0 < nj) ? b_row1[base_j + 0] : 0.f;
            float b1_1 = (base_j + 1 < nj) ? b_row1[base_j + 1] : 0.f;
            float b1_2 = (base_j + 2 < nj) ? b_row1[base_j + 2] : 0.f;
            float b1_3 = (base_j + 3 < nj) ? b_row1[base_j + 3] : 0.f;
            b_vals1 = make_float4(b1_0, b1_1, b1_2, b1_3);
        }

        // Accumulate partial sums
        sum0 += a_val0 * b_vals0.x + a_val1 * b_vals1.x;
        sum1 += a_val0 * b_vals0.y + a_val1 * b_vals1.y;
        sum2 += a_val0 * b_vals0.z + a_val1 * b_vals1.z;
        sum3 += a_val0 * b_vals0.w + a_val1 * b_vals1.w;
    }

    // Handle last k if nk is odd
    if (k < nk)
    {
        float a_val = a_row[k];
        float *b_row = &B[k * nj];

        float4 b_vals = make_float4(0.f, 0.f, 0.f, 0.f);

        if (base_j + 3 < nj)
        {
            b_vals = *((float4 *)(b_row + base_j));
        }
        else
        {
            float b0 = (base_j + 0 < nj) ? b_row[base_j + 0] : 0.f;
            float b1 = (base_j + 1 < nj) ? b_row[base_j + 1] : 0.f;
            float b2 = (base_j + 2 < nj) ? b_row[base_j + 2] : 0.f;
            float b3 = (base_j + 3 < nj) ? b_row[base_j + 3] : 0.f;
            b_vals = make_float4(b0, b1, b2, b3);
        }

        sum0 += a_val * b_vals.x;
        sum1 += a_val * b_vals.y;
        sum2 += a_val * b_vals.z;
        sum3 += a_val * b_vals.w;
    }

    // Write results to E if within bounds
    if (base_j + 0 < nj)
        E[base_i * nj + base_j + 0] = sum0;
    if (base_j + 1 < nj)
        E[base_i * nj + base_j + 1] = sum1;
    if (base_j + 2 < nj)
        E[base_i * nj + base_j + 2] = sum2;
    if (base_j + 3 < nj)
        E[base_i * nj + base_j + 3] = sum3;
}



	


__global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, float *C, float *D, float *F)
{
    // Each thread computes a 2x2 block of output elements (register blocking)
    // blockDim.x and blockDim.y must be multiples of 2 for this to work properly
    int base_j = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // column index start for 2 outputs
    int base_i = (blockIdx.y * blockDim.y + threadIdx.y) * 2; // row index start for 2 outputs

    if (base_i >= nj || base_j >= nl)
        return;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    // Process k in pairs to reduce loop overhead and improve instruction-level parallelism
    int k = 0;
    int k_limit = nm - (nm % 2);

    for (; k < k_limit; k += 2)
    {
        // Load C elements for two rows and two k values (vectorized loads)
        // C indexing: C[i * nm + k]
        float2 c0 = make_float2(0.0f, 0.0f);
        float2 c1 = make_float2(0.0f, 0.0f);

        if (base_i < nj)
        {
            c0.x = C[base_i * nm + k];
            c0.y = C[base_i * nm + k + 1];
        }
        if ((base_i + 1) < nj)
        {
            c1.x = C[(base_i + 1) * nm + k];
            c1.y = C[(base_i + 1) * nm + k + 1];
        }

        // Load D elements for two columns and two k values (vectorized loads)
        // D indexing: D[k * nl + j]
        float2 d0 = make_float2(0.0f, 0.0f);
        float2 d1 = make_float2(0.0f, 0.0f);

        if (base_j < nl)
        {
            d0.x = D[k * nl + base_j];
            d1.x = D[(k + 1) * nl + base_j];
        }
        if ((base_j + 1) < nl)
        {
            d0.y = D[k * nl + base_j + 1];
            d1.y = D[(k + 1) * nl + base_j + 1];
        }

        // Accumulate partial results
        acc00 += c0.x * d0.x + c0.y * d1.x;
        acc01 += c0.x * d0.y + c0.y * d1.y;
        acc10 += c1.x * d0.x + c1.y * d1.x;
        acc11 += c1.x * d0.y + c1.y * d1.y;
    }

    // Handle leftover k if nm is odd
    if (k < nm)
    {
        float c0 = (base_i < nj) ? C[base_i * nm + k] : 0.0f;
        float c1 = (base_i + 1 < nj) ? C[(base_i + 1) * nm + k] : 0.0f;

        float d0 = (base_j < nl) ? D[k * nl + base_j] : 0.0f;
        float d1 = (base_j + 1 < nl) ? D[k * nl + base_j + 1] : 0.0f;

        acc00 += c0 * d0;
        acc01 += c0 * d1;
        acc10 += c1 * d0;
        acc11 += c1 * d1;
    }

    // Write results back to global memory with boundary checks
    if (base_i < nj && base_j < nl)
        F[base_i * nl + base_j] = acc00;
    if (base_i < nj && (base_j + 1) < nl)
        F[base_i * nl + base_j + 1] = acc01;
    if ((base_i + 1) < nj && base_j < nl)
        F[(base_i + 1) * nl + base_j] = acc10;
    if ((base_i + 1) < nj && (base_j + 1) < nl)
        F[(base_i + 1) * nl + base_j + 1] = acc11;
}



	


__global__ void mm3_kernel3(int ni, int nj, int nl, int nm_unused, int nm_unused2, float *E, float *F, float *G)
{
    // Each thread computes a 2x2 block of G elements (register blocking)
    // blockDim.x and blockDim.y must be multiples of 2 for this to work well
    int base_j = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int base_i = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    if (base_i >= ni || base_j >= nl)
        return;

    float g00 = 0.f, g01 = 0.f, g10 = 0.f, g11 = 0.f;

    int k = 0;

    // Process k dimension in pairs for better instruction-level parallelism and to reduce loop overhead
    for (; k + 1 < nj; k += 2)
    {
        // Load E elements for rows base_i and base_i+1 at k and k+1
        // Coalesced access: E is row-major, so threads with adjacent threadIdx.x access contiguous memory in j dimension
        float e0_k0 = (base_i < ni) ? E[base_i * nj + k] : 0.f;
        float e1_k0 = (base_i + 1 < ni) ? E[(base_i + 1) * nj + k] : 0.f;
        float e0_k1 = (base_i < ni) ? E[base_i * nj + k + 1] : 0.f;
        float e1_k1 = (base_i + 1 < ni) ? E[(base_i + 1) * nj + k + 1] : 0.f;

        // Load F elements for columns base_j and base_j+1 at k and k+1
        // Coalesced access: F is row-major, so threads with adjacent threadIdx.x access contiguous memory in column dimension
        float f_k0_0 = (base_j < nl) ? F[k * nl + base_j] : 0.f;
        float f_k0_1 = (base_j + 1 < nl) ? F[k * nl + base_j + 1] : 0.f;
        float f_k1_0 = (base_j < nl) ? F[(k + 1) * nl + base_j] : 0.f;
        float f_k1_1 = (base_j + 1 < nl) ? F[(k + 1) * nl + base_j + 1] : 0.f;

        // Accumulate for k
        g00 += e0_k0 * f_k0_0;
        g01 += e0_k0 * f_k0_1;
        g10 += e1_k0 * f_k0_0;
        g11 += e1_k0 * f_k0_1;

        // Accumulate for k+1
        g00 += e0_k1 * f_k1_0;
        g01 += e0_k1 * f_k1_1;
        g10 += e1_k1 * f_k1_0;
        g11 += e1_k1 * f_k1_1;
    }

    // Handle remaining k if nj is odd
    for (; k < nj; k++)
    {
        float e0 = (base_i < ni) ? E[base_i * nj + k] : 0.f;
        float e1 = (base_i + 1 < ni) ? E[(base_i + 1) * nj + k] : 0.f;
        float f0 = (base_j < nl) ? F[k * nl + base_j] : 0.f;
        float f1 = (base_j + 1 < nl) ? F[k * nl + base_j + 1] : 0.f;

        g00 += e0 * f0;
        g01 += e0 * f1;
        g10 += e1 * f0;
        g11 += e1 * f1;
    }

    if (base_i < ni && base_j < nl)
        G[base_i * nl + base_j] = g00;
    if (base_i < ni && (base_j + 1) < nl)
        G[base_i * nl + base_j + 1] = g01;
    if ((base_i + 1) < ni && base_j < nl)
        G[(base_i + 1) * nl + base_j] = g10;
    if ((base_i + 1) < ni && (base_j + 1) < nl)
        G[(base_i + 1) * nl + base_j + 1] = g11;
}




/* Main computational kernel on CPU */
void mm3_cpu(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
	int i, j, k;

	/* E := A*B */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			E[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				E[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	/* F := C*D */
	for (i = 0; i < _PB_NJ; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			F[i][j] = 0;
			for (k = 0; k < _PB_NM; ++k)
			{
				F[i][j] += C[i][k] * D[k][j];
			}
		}
	}

	/* G := E*F */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			G[i][j] = 0;
			for (k = 0; k < _PB_NJ; ++k)
			{
				G[i][j] += E[i][k] * F[k][j];
			}
		}
	}
}


void mm3Cuda(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl),
		DATA_TYPE POLYBENCH_2D(G_outputFromGpu,NI,NL,ni,nl))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;
	
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM);
	cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL);
	cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ);
	cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL);
	cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice);
	cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice);
	cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice);	
	
	// Kernel launch configuration
	// For mm3_kernel1:
	// blockDim.x = 32, blockDim.y = 8
	// gridDim.x = ceil(NJ / (32*4)) = ceil(NJ / 128)
	// gridDim.y = ceil(NI / 8)
	dim3 block1(32, 8);
	dim3 grid1((nj + (32*4) - 1) / (32*4), (ni + 8 - 1) / 8);

	// For mm3_kernel2 and mm3_kernel3:
	// Each thread computes 2x2 block, so effective dimension is half
	// blockDim.x and blockDim.y should be multiples of 2, use 16x16
	// gridDim.x = ceil(NL / (blockDim.x * 2))
	// gridDim.y = ceil(NJ / (blockDim.y * 2)) for kernel2
	// gridDim.y = ceil(NI / (blockDim.y * 2)) for kernel3
	dim3 block2(16, 16);
	dim3 grid2((nl + (block2.x * 2) - 1) / (block2.x * 2), (nj + (block2.y * 2) - 1) / (block2.y * 2));
	dim3 grid3((nl + (block2.x * 2) - 1) / (block2.x * 2), (ni + (block2.y * 2) - 1) / (block2.y * 2));

	/* Start timer. */
  	polybench_start_instruments;

	mm3_kernel1<<<grid1,block1>>>(ni, nj, nk, nl, nm, A_gpu, B_gpu, E_gpu);
	cudaDeviceSynchronize();
	mm3_kernel2<<<grid2,block2>>>(ni, nj, nk, nl, nm, C_gpu, D_gpu, F_gpu);
	cudaDeviceSynchronize();
	mm3_kernel3<<<grid3,block2>>>(ni, nj, nl, nm, nm, E_gpu, F_gpu, G_gpu);
	cudaDeviceSynchronize();

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
	cudaMemcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	cudaFree(D_gpu);
	cudaFree(E_gpu);
	cudaFree(F_gpu);
	cudaFree(G_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
	if ((i * nl + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;
	int nm = NM;

	/* Variable declaration/allocation. */
	POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
	POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
	POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
	POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
	POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
	POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
	POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
	POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

	init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

	GPU_argv_init();

	mm3Cuda(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), 
		POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(E), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(C), 
			POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(G));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

	#else //print output to stderr so no dead code elimination

		print_array(ni, nl, POLYBENCH_ARRAY(G_outputFromGpu));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(E);
	POLYBENCH_FREE_ARRAY(F);
	POLYBENCH_FREE_ARRAY(G);
	POLYBENCH_FREE_ARRAY(G_outputFromGpu);

	return 0;
}

#include "../../common/polybench.c"

