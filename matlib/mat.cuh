#ifndef LLMC_MAT_LIB
#define LLMC_MAT_LIB

#include <cuda_runtime.h>

typedef float nn_real;

template <typename OrderPolicy>
class Matrix {
	protected:
		int rows_, cols_, size_, stride_;
		nn_real *d_data_;

	public:
		Matrix(int rows, int cols, nn_real *data)
			: rows_(rows), cols_(cols), size_(rows*cols), d_data_(data)  { 
				stride_ = OrderPolicy::stride(rows, cols);
		}

		Matrix(int rows, int cols, nn_real *data, int stride)
			: rows_(rows), cols_(cols), size_(rows*cols), stride_(stride), d_data_(data)  { }
		// Implicit destructor OK, we do not want to be manually deleting the gpu memory.

		__host__ __device__ int size(void) const { return rows_ * cols_; }
		__host__ __device__ int rows(void) const { return rows_; }
		__host__ __device__ int cols(void) const { return cols_; }
		__host__ __device__ int stride(void) const { return stride_; }

		__device__ nn_real *data() const { return d_data_; }

		__device__ nn_real &operator()(int row, int col) {
			return d_data_[OrderPolicy::index(row, col, this->rows(), this->cols(), this->stride())];
		}

		__device__ Matrix<OrderPolicy> getSubMatrix(int row, int col, int numRows, int numCols) {
			return Matrix<OrderPolicy>(
					numRows, 
					numCols, 
					this->d_data + OrderPolicy::index(row, col, this->stride()),
					this->stride()
				);
		}

		__host__ nn_real *getHostCopy() {
			nn_real *ptr = reinterpret_cast<nn_real *>(malloc(this->size() * sizeof(nn_real)));
			cudaMemcpy(ptr, this->d_data_, this->size() * sizeof(nn_real), cudaMemcpyDeviceToHost);
			return ptr;
		}
};

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

template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
void mul(Matrix<OutPolicy>& out, 
         const Matrix<APolicy>& A,
         const Matrix<BPolicy>& B,
         const Matrix<BiasPolicy>& bias);

#endif // LLMC_MAT_LIB
