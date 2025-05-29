#include "matrix_kernels.cuh"

template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel(
		Matrix<OutPolicy> out,
		const Matrix<APolicy> A,
		const Matrix<BPolicy> B,
		const Matrix<BiasPolicy> bias) {

	Matrix<APolicy> A_sub = A.getSubMatrix(128 * blockIdx.y, 0, 128, A.cols());
	Matrix<BPolicy> B_sub = B.getSubMatrix(0, 128 * blockIdx.x, B.rows(), 128);
	Matrix<OutPolicy> out_sub = out.getSubMatrix(128 * blockIdx.x, 128 * blockIdx.y, 128, 128);

#if 0
	__shared__ float lhs_s[128][32];
	__shared__ float rhs_s[128][32];
#endif

	int oc = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

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
			out_sub(8 * threadIdx.x + i, 8 * threadIdx.y + j) = vals[i][j];
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

// Explicit instantiation for the expected data ordering.
template void mul(Matrix<RowMajor>& out, 
									const Matrix<RowMajor>& A, 
									const Matrix<ColMajor>& B,
									const Matrix<RowMajor>& bias);
