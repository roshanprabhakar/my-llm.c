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

#endif // LLMC_MATRIX_H