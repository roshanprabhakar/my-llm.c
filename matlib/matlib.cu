#include "matlib/mat.cuh"
#include "llmc/cuda_common.h"

#if 0
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
    const float* inp, const float* weight, const float* bias,
    int C, int OC) {

	// blockIdx.y
	// 			The index of the output tile along the cols dimension
	// blockIdx.x
	// 			The index of the output tile along the rows dimension

  // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
  // inp is (B,T,C), weight is (OC, C), bias is (OC)
  // each thread handles 8x8 elements; each block 128 by 128 elements.
  int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

  // buffers to cache chunks of the input matrices
  __shared__ float lhs_s[128][32];
  __shared__ float rhs_s[128][32];

  // adjust our pointers for the current block
  inp += 128 * blockIdx.x * C;
  weight += 128 * blockIdx.y * C;
  out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

  float vals[8][8] = {};
  if(bias != NULL) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j += 4) {
				st_vec(&vals[i][j], ld_vec(bias + oc + j));
      }
    }
  }

  int si_start = 4*(16 * threadIdx.y + threadIdx.x);
  for (int so = 0; so < C; so += 32) {
    __syncthreads();
    int xmod8 = threadIdx.x % 8;
    int xby8 = threadIdx.x / 8;
    int xo = 4 * xmod8;
    for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
      st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
      st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
    }
    __syncthreads();

    for (int si = si_start; si < si_start + 32; si += 4) {
      float4 rhs[8];
      for (int u = 0; u < 8; ++u) {
        rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
      }

      for (int ii = 0; ii < 8; ++ii) {
        float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
        for (int ji = 0; ji < 8; ++ji) {
          vals[ii][ji] += lhs.x * rhs[ji].x;
          vals[ii][ji] += lhs.y * rhs[ji].y;
          vals[ii][ji] += lhs.z * rhs[ji].z;
          vals[ii][ji] += lhs.w * rhs[ji].w;
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; j += 4) {
      float4 result;
      result.x = vals[i][j + 0];
      result.y = vals[i][j + 1];
      result.z = vals[i][j + 2];
      result.w = vals[i][j + 3];
      st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
    }
  }
}
#endif

template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel(
		Matrix<OutPolicy> out,
		const Matrix<APolicy> A,
		const Matrix<BPolicy> B,
		const Matrix<BiasPolicy> bias) {
	
	// Grab a (128xC) chunk of A, and a (Cx128) chunk of B.
	Matrix<APolicy> A_sub = A.getSubMatrix(128 * blockIdx.y, 0, 128, A.cols());
	Matrix<BPolicy> B_sub = B.getSubMatrix(0, 128 * blockIdx.x, B.rows(), 128);
	Matrix<OutPolicy> out_sub = out.getSubMatrix(128 * blockIdx.x, 128 * blockIdx.y, 128, 128);

#if 0
	// Start tiled calculations.
	__shared__ nn_real lhs_s[128][32];
	__shared__ nn_real rhs_s[128][32];
#endif

	int oc = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

	// Load biases--this thread compute 64 output values, representing an 8x8 chunk of the output.
	// For any output row, the bias value is the same across columns.
	float vals[8][8] = {};
	if (bias.data() != NULL) {
		for (int i = 0; i < 8; ++i) {
			for (int j = 0; j < 8; ++j) {
				vals[i][j] = bias(0, oc + j);
			}
		}
	}

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			out_sub(8 * threadIdx.x + i, 8 * threadIdx.y + j) = result[i][j];
		}
	}

}
	

template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
void mul(Matrix<OutPolicy>& out, 
         const Matrix<APolicy>& A,
         const Matrix<BPolicy>& B,
         const Matrix<BiasPolicy>& bias) {

	int sqrt_block_size = 16;

	dim3 gridDim(CEIL_DIV(B.cols(), 8*sqrt_block_size), CEIL_DIV(A.rows(), 8*sqrt_block_size));
  dim3 blockDim(sqrt_block_size, sqrt_block_size);
	matmul_forward_kernel<<<gridDim, blockDim>>>(out, A, B, bias);

  cudaCheck(cudaGetLastError());
}

