#ifndef LLMC_MATRIX_H
#define LLMC_MATRIX_H

#include <cuda_runtime.h>
#include "llmc/cuda_common.h"

class RowMajor {
	public:
		static __host__ __device__ int index(int row, int col, int stride) { return row * stride + col; }
		static __host__ __device__ int stride(int rows, int cols) { return cols; }
};

class ColMajor {
	public:
		static __host__ __device__ int index(int row, int col, int stride) { return col * stride + row; }
		static __host__ __device__ int stride(int rows, int cols) { return rows; }
};

template <typename OrderPolicy>
class Matrix {
	protected:
		int rows_, cols_, size_, stride_;
		float	 *d_data_;

	public:
		__host__ __device__ Matrix(int rows, int cols, float *data)
			: rows_(rows), cols_(cols), size_(rows*cols), d_data_(data)  { 
				stride_ = OrderPolicy::stride(rows, cols);
		}

		__host__ __device__ Matrix(int rows, int cols, float *data, int stride)
			: rows_(rows), cols_(cols), size_(rows*cols), stride_(stride), d_data_(data)  { }

		__host__ __device__ int size(void) const { return rows_ * cols_; }
		__host__ __device__ int rows(void) const { return rows_; }
		__host__ __device__ int cols(void) const { return cols_; }
		__host__ __device__ int stride(void) const { return stride_; }

		__device__ float *data() const { return d_data_; }

		__device__ float &operator()(int row, int col) {
			return d_data_[OrderPolicy::index(row, col, this->stride())];
		}

		__device__ float operator()(int row, int col) const {
			return d_data_[OrderPolicy::index(row, col, this->stride())];
		}

		__device__ Matrix<OrderPolicy> getSubMatrix(int row, int col, int numRows, int numCols) const {
			return Matrix<OrderPolicy>(
					numRows, 
					numCols, 
					this->d_data_ + OrderPolicy::index(row, col, this->stride()),
					this->stride()
				);
		}

		__host__ float *getHostCopy() {
			float *ptr = reinterpret_cast<float *>(malloc(this->size() * sizeof(float)));
			cudaMemcpy(ptr, this->d_data_, this->size() * sizeof(float), cudaMemcpyDeviceToHost);
			return ptr;
		}
};


#if 0
template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel(
		Matrix<OutPolicy> out,
		const Matrix<APolicy> A,
		const Matrix<BPolicy> B,
		const Matrix<BiasPolicy> bias) {

	Matrix<APolicy> A_sub = A.getSubMatrix(128 * blockIdx.y, 0, 128, A.cols());
	Matrix<BPolicy> B_sub = B.getSubMatrix(0, 128 * blockIdx.x, B.rows(), 128);
	Matrix<OutPolicy> out_sub = out.getSubMatrix(128 * blockIdx.x, 128 * blockIdx.y, 128, 128);

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
			
			if (8 * threadIdx.x + i < out.rows() && 8 * threadIdx.y + j < out.cols()) {
				out_sub(8 * threadIdx.x + i, 8 * threadIdx.y + j) = vals[i][j];
			}
		}
	}
}
#endif

template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
__global__ void matmul_forward_kernel(
		Matrix<OutPolicy> out,
		const Matrix<APolicy> A,
		const Matrix<BPolicy> B,
		const Matrix<BiasPolicy> bias) {

	for (int r = 0; r < 8; ++r) {
		for (int c = 0; c < 8; ++c) {
			int y = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
			int x = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

			if (y < out.rows() && x < out.cols()) {
				out(y, x) = bias(1, x);
			}
		}
	}
							
}




template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
void mul(Matrix<OutPolicy>& out,
         const Matrix<APolicy>& A,
         const Matrix<BPolicy>& B,
         const Matrix<BiasPolicy>& bias) {

	int sqrt_block_size = 16;

	if (out.rows() != A.rows() || out.cols() != B.cols() || out.cols() != bias.cols()) {
		printf("Error: incompatible matrix dimensions.\n");
		return;
	}

	dim3 gridDim(CEIL_DIV(B.cols(), 8*sqrt_block_size), CEIL_DIV(A.rows(), 8*sqrt_block_size));
	dim3 blockDim(sqrt_block_size, sqrt_block_size);
	matmul_forward_kernel<<<gridDim, blockDim>>>(out, A, B, bias);

	cudaCheck(cudaGetLastError());
}


#endif // LLMC_MATRIX_H
