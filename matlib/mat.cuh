#ifndef LLMC_MAT_LIB
#define LLMC_MAT_LIB

#include <cuda_runtime.h>

typedef float nn_real;

class Matrix {
	protected:
		int rows_, cols_, size_;
		nn_real *d_data_;

	public:
		Matrix(int rows, int cols)
			: rows_(rows), cols_(cols), size_(rows*cols), d_data_(nullptr) { }
		Matrix(int row, int cols, nn_real *data) 
			: rows_(rows), cols_(cols), size_(rows*cols), d_data_(data) { }

		~Matrix() { if (d_data_) cudaFree(d_data_); }

		__device__ int size(void) const { return rows_ * cols_; }
		__device__ int rows(void) const { return rows_; }
		__device__ int cols(void) const { return cols_; }
		__device__ nn_real *data() const { return d_data_; }

		virtual __device__ nn_real &operator()(int row, int col) = 0;
};

class DenseMatrix : public Matrix {
	public:
		DenseMatrix(int rows, int cols): Matrix(rows, cols) {
			nn_real *data = nullptr;
			CUDA_CHECK(cudaMalloc((void **)&data, this->size() * sizeof(nn_real)));

			this->d_data_ = std::shared_ptr<nn_real>(data, [](nn_real *ptr) {
					if (ptr) { cudaFree(ptr); }
			});
		}
};

class RowMajorMatrix : public DenseMatrix {
	public:
		__device__ nn_real &operator()(int row, int col) {
			return this->d_data_[row * this->cols() + col];
		};
};

class ColMajorMatrix : public DenseMatrix {
	public:
		__device__ nn_real &operator()(int row, int col) {
			return this->d_data_[col * this->rows() + row];
		}
};

#endif // LLMC_MAT_LIB
