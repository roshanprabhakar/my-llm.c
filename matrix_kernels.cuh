#ifndef LLMC_MATRIX_KERNELS_H
#define LLMC_MATRIX_KERNELS_H

#include "matrix.cuh"

template <typename OutPolicy, typename APolicy, typename BPolicy, typename BiasPolicy>
void mul(Matrix<OutPolicy>& out, 
         const Matrix<APolicy>& A,
         const Matrix<BPolicy>& B,
         const Matrix<BiasPolicy>& bias);

#endif // LLMC_MATRIX_KERNELS_H