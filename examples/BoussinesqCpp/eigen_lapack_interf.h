#include <Eigen/Dense>

typedef int32_t int_lapack_t;

extern "C"
{
  /*
   * Compute for an N-by-N real nonsymmetric matrix A, the eigenvalues,
   * the real Schur form T, and, optionally, the matrix of Schur vectors Z
   */
  void dgees_(char *jobvs, char *sort, int_lapack_t *select, int_lapack_t *n, double *a,
              int_lapack_t *lda, int_lapack_t *sdim, double *wr, double *wi, double *vs,
              int_lapack_t *ldvs, double *work, int_lapack_t *lwork, int_lapack_t *bwork, int_lapack_t *info);

  void dtrsyl_(char *transa, char *transb, int_lapack_t *isgn, int_lapack_t *m, int_lapack_t *n, const double *a,
               int_lapack_t *lda, const double *b, int_lapack_t *ldb, double *c, int_lapack_t *ldc, double *scale, int_lapack_t *info);
}


/**
 * Schur decomposition of an Eigen::MatrixXd using Lapack
 * A = U T U^H
 * */
void schur_lapack(const Eigen::MatrixXd & A,
                  Eigen::MatrixXd &U,
                  Eigen::MatrixXd &T) {
  int n = A.rows();
  int lda = A.outerStride();
  int ldu = n;

  char jobvs = 'V';
  char sort = 'N';

  int lwork = 3 * n;
  int sdim = 0;
  int info = 0;

  Eigen::VectorXd wr(n);
  Eigen::VectorXd wi(n);
  Eigen::VectorXd work(lwork);

  // Lapack overwrites the input matrix, so make a copy
  T = A;

  dgees_(&jobvs, &sort, 0, &n, T.data(), &lda, &sdim,
         wr.data(), wi.data(), U.data(), &ldu, work.data(), &lwork, 0, &info);
}

/**
 * Triangular Sylvester equation defined with Eigen::MatrixXd
 * A X + X B = C
 * */
void triangular_sylvester_lapack(const Eigen::MatrixXd & A,
                                 const Eigen::MatrixXd & B,
                                 Eigen::MatrixXd &C) {
  int m = A.rows();
  int n = B.rows();
  int lda = A.outerStride();
  int ldb = B.outerStride();
  int ldc = C.outerStride();

  char transa = 'N';
  char transb = 'C';

  int isgn = 1;
  int info = 0;
  double scale = 1.0;

  dtrsyl_(&transa, &transb, &isgn, &m, &n, A.data(), &lda, B.data(), &ldb, C.data(), &ldc, &scale, &info);

  C *= scale;
}
