#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <random>
#include "eigen_lapack_interf.h"

class Boussinesq2D {
  public:

    // Grid and physical parameters
    int m_M{0}, m_N{0};
    double m_dt = 0.0;
    double m_dx = 0.0;
    double m_Pr{1.0};
    double m_Le{1.0};
    double m_A{5.0};
    double m_Ra{4.0e4};
    double m_delta{0.05};
    double m_tauT{0.1};
    double m_tauS{1.0};
    double m_beta{0.1};
    double m_eps{0.05};
    double m_t{0.0};
    int m_step{0};
    int m_K{4};

    // Grid containers
    Eigen::VectorXd m_xx;
    Eigen::VectorXd m_zz;

    // Operators
    Eigen::MatrixXd m_Dxx;
    Eigen::MatrixXd m_Fxx;
    Eigen::MatrixXd m_Dx;
    Eigen::MatrixXd m_Fx;
    Eigen::MatrixXd m_Dzz;
    Eigen::MatrixXd m_Fzz;
    Eigen::MatrixXd m_Dz;
    Eigen::MatrixXd m_Fz;
    Eigen::MatrixXd m_DzzT;
    Eigen::MatrixXd m_FzzT;
    Eigen::MatrixXd m_DzT;
    Eigen::MatrixXd m_FzT;
    Eigen::MatrixXd m_Hz;
    Eigen::MatrixXd m_HzT;
    Eigen::MatrixXd m_Scorr;

    // Advance matrices
    Eigen::MatrixXd m_AT;
    Eigen::MatrixXd m_BT;
    Eigen::MatrixXd m_FT;
    Eigen::MatrixXd m_AS;
    Eigen::MatrixXd m_BS;
    Eigen::MatrixXd m_FS;
    Eigen::MatrixXd m_Aw;
    Eigen::MatrixXd m_Bw;

    // Noise
    Eigen::MatrixXd m_cos_term;
    Eigen::MatrixXd m_sin_term;

    // Solution matrices
    Eigen::MatrixXd m_S;
    Eigen::MatrixXd m_T;
    Eigen::MatrixXd m_psi;
    Eigen::MatrixXd m_w;

  Boussinesq2D(int M, int N, double dt) : m_M{M},
    m_N{N},
    m_dt{dt} {
      // Initialize the computational grid
      m_xx = Eigen::VectorXd::LinSpaced(m_M + 1, 0.0, m_A);
      m_dx = m_xx[1] - m_xx[0];
      m_zz = Eigen::VectorXd::Zero(m_N + 1);
      for (int j = 0; j <= m_N; ++j) {
        m_zz(j) = z_j(j);
      }

      // Contruct operators and matrices
      make_x_operators();
      make_z_operators();
      make_boundary_layer_matrix();
      make_corrector_matrix();
      make_integration_matrices();
      init_Snoise();

      // Initialize the state
      init_state();
    };

  ~Boussinesq2D() {}

  double h(double z) {
      return exp(-(1.0 - z) / m_delta);
  }

  Eigen::VectorXd Ts(const Eigen::VectorXd& x) {
      return 0.5 * (x.unaryExpr([A = m_A](double xi) {
          return cos(2 * M_PI * (xi / A - 0.5)) + 1;
      }));
  }

  Eigen::VectorXd Ss(const Eigen::VectorXd& x, double a_beta) {
      return x.unaryExpr([A = m_A, a_beta](double xi) {
          return 3.5 * cos(2 * M_PI * (xi / A - 0.5)) - a_beta * sin(M_PI * (xi / A - 0.5));
      });
  }

  void init_Snoise() {
      m_cos_term = Eigen::MatrixXd::Zero(m_K, m_M + 1);
      m_sin_term = Eigen::MatrixXd::Zero(m_K, m_M + 1);
      for (int i = 0; i <= m_M; ++i) {
          for (int k = 1; k <= m_K; ++k) {
              m_cos_term(k - 1, i) = cos(2 * M_PI * k * m_xx[i] / m_A);
              m_sin_term(k - 1, i) = sin(2 * M_PI * k * m_xx[i] / m_A);
          }
      }
  }

  Eigen::MatrixXd Snoise(const Eigen::VectorXd& normalrange) {
      Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m_M + 1, m_N + 1);
      Eigen::VectorXd noise_surface = Eigen::VectorXd::Zero(m_M + 1);
      for (int k = 0; k < m_K; ++k) {
          noise_surface += normalrange[2 * k] * m_cos_term.row(k).transpose();
          noise_surface += normalrange[2 * k + 1] * m_sin_term.row(k).transpose();
      }
      for (int j = 0; j <= m_N; ++j) {
          result.col(j) = noise_surface * sqrt(m_eps / m_K) * h(m_zz[j]);
      }
      return result;
  }

  double z_j(int j) {
      double q = 3.0;
      return 0.5 + tanh(q * (double(j) / m_N - 0.5)) / (2 * tanh(q / 2));
  }

  double dz_j(int j) {
      return z_j(j + 1) - z_j(j);
  }

  void init_state() {
    m_w = Eigen::MatrixXd::Zero(m_M + 1, m_N + 1);
    m_S = Eigen::MatrixXd::Zero(m_M + 1, m_N + 1);
    m_T = Eigen::MatrixXd::Zero(m_M + 1, m_N + 1);
    m_psi = Eigen::MatrixXd::Zero(m_M + 1, m_N + 1);
    load_state("AlmostONState.bin");
  }

  void make_x_operators() {
      m_Dxx = Eigen::MatrixXd::Zero(m_M + 1, m_M + 1);
      m_Fxx = Eigen::MatrixXd::Zero(m_M + 1, m_M + 1);
      m_Dx = Eigen::MatrixXd::Zero(m_M + 1, m_M + 1);
      m_Fx = Eigen::MatrixXd::Zero(m_M + 1, m_M + 1);
      for (int i = 1; i < m_M; ++i) {
          // Diffusion operators: 2nd order, centered
          m_Dxx(i,i-1) = 1.0;
          m_Dxx(i,i) = -2.0;
          m_Dxx(i,i+1) = 1.0;
          m_Fxx(i, i - 1) = 1.0;
          m_Fxx(i, i) = -2.0;
          m_Fxx(i, i + 1) = 1.0;

          // Advection operators: 2nd order, centered
          m_Dx(i, i - 1) = -1.0;
          m_Dx(i, i + 1) = 1.0;
          m_Fx(i, i - 1) = -1.0;
          m_Fx(i, i + 1) = 1.0;
      }

      // Boundary conditions
      m_Dxx(0,0) = -2.0;
      m_Dxx(0,1) = 2.0;
      m_Dxx(m_M,m_M-1) = 2.0;
      m_Dxx(m_M,m_M) = -2.0;
      m_Fxx(0, 0) = m_Fxx(0, 1) = m_Fxx(m_M, m_M) = m_Fxx(m_M, m_M - 1) = 0;
      m_Dxx /= m_dx * m_dx;
      m_Fxx /= m_dx * m_dx;

      m_Fx(0,1) = 2.0;
      m_Fx(m_M,m_M-1) = -2.0;
      m_Dx /= (2 * m_dx);
      m_Fx /= (2 * m_dx);
  }

  void make_z_operators() {
      m_Dzz = Eigen::MatrixXd::Zero(m_N + 1, m_N + 1);
      m_Fzz = Eigen::MatrixXd::Zero(m_N + 1, m_N + 1);
      m_Dz = Eigen::MatrixXd::Zero(m_N + 1, m_N + 1);
      m_Fz = Eigen::MatrixXd::Zero(m_N + 1, m_N + 1);

      for (int j = 1; j < m_N; ++j) {
          double dz1 = dz_j(j - 1);
          double dz2 = dz_j(j);
          double dz_avg = 0.5 * (z_j(j+1) - z_j(j-1));

          // Diffusion operators: 2nd order, centered
          m_Dzz(j, j - 1) = 1.0 / (dz1 * dz_avg);
          m_Dzz(j, j) = -2.0 / (dz1 * dz2);
          m_Dzz(j, j + 1) = 1.0 / (dz2 * dz_avg);
          m_Fzz(j, j - 1) = 1.0 / (dz1 * dz_avg);
          m_Fzz(j, j) = -2.0 / (dz1 * dz2);
          m_Fzz(j, j + 1) = 1.0 / (dz2 * dz_avg);

          // Advection operators: 2nd order, centered
          m_Dz(j, j - 1) = -0.5 / dz_avg;
          m_Dz(j, j + 1) = 0.5 / dz_avg;
          m_Fz(j, j - 1) = -0.5 / dz_avg;
          m_Fz(j, j + 1) = 0.5 / dz_avg;
      }

      // Boundary conditions
      m_Dzz(0, 0) = - 2.0 / (dz_j(-1) * (z_j(1) - z_j(-1)))
                    - 2.0 / (dz_j(0) * (z_j(1) - z_j(-1)));
      m_Dzz(0, 1) = 2.0 / (dz_j(-1) * (z_j(1) - z_j(-1)))
                  + 2.0 / (dz_j(0) * (z_j(1) - z_j(-1)));
      m_Dzz(m_N, m_N - 1) = 2.0 / (dz_j(m_N - 1) * (z_j(m_N+1) - z_j(m_N - 1)))
                          + 2.0 / (dz_j(m_N) * (z_j(m_N+1) - z_j(m_N - 1)));
      m_Dzz(m_N, m_N) = - 2.0 / (dz_j(m_N - 1) * (z_j(m_N+1) - z_j(m_N - 1)))
                        - 2.0 / (dz_j(m_N) * (z_j(m_N+1) - z_j(m_N - 1)));
      m_Fzz(0, 0) = m_Fzz(0, 1) = m_Fzz(m_N, m_N) = m_Fzz(m_N, m_N - 1) = 0;

      m_Dz(0, 1) = m_Dz(m_N, m_N - 1) = 0;
      m_Fz(0, 1) = 1.0 / dz_j(-1);
      m_Fz(m_N, m_N - 1) = -1.0 / dz_j(m_N - 1);

      m_DzzT = m_Dzz.transpose();
      m_FzzT = m_Fzz.transpose();
      m_DzT = m_Dz.transpose();
      m_FzT = m_Fz.transpose();
  }

  void make_boundary_layer_matrix() {
      m_Hz = Eigen::MatrixXd::Zero(m_N + 1, m_N + 1);
      for (int j = 0; j <= m_N; ++j) {
          m_Hz(j, j) = h(z_j(j));
      }
      m_HzT = m_Hz.transpose();
  }

  void make_corrector_matrix() {
      m_Scorr = Eigen::MatrixXd::Identity(m_N + 1, m_N + 1);
      m_Scorr(0, 0) = 0.0;
      m_Scorr(m_N, m_N) = 0.0;
  }

  void make_integration_matrices() {
    m_AT = Eigen::MatrixXd::Identity(m_M + 1, m_M + 1) - m_dt * m_Dxx;
    m_BT = m_dt / m_tauT * m_HzT - m_dt * m_DzzT;
    m_AS = Eigen::MatrixXd::Identity(m_M + 1, m_M + 1) - m_dt / m_Le * m_Dxx;
    m_BS = - m_dt / m_Le * m_DzzT;
    m_Aw = Eigen::MatrixXd::Identity(m_M + 1, m_M + 1) - m_dt * m_Pr * m_Fxx;
    m_Bw = - m_dt * m_Pr * m_FzzT;

    Eigen::VectorXd h_z(m_N + 1);
    for (int j = 0; j <= m_N; ++j) {
        h_z[j] = h(m_zz[j]);
    }

    Eigen::VectorXd Tsurf = Ts(m_xx);
    Eigen::VectorXd Ssurf = Ss(m_xx, m_beta);
    m_FT = (Tsurf * h_z.transpose()) / m_tauT;
    m_FS = (Ssurf * h_z.transpose()) / m_tauS;
  }

  void write_state() {
    write_state("state", m_step, m_w, m_S, m_T, m_psi);
  }

  void write_state(const std::string a_prefix,
                   int a_step,
                   const Eigen::MatrixXd& a_w,
                   const Eigen::MatrixXd& a_S,
                   const Eigen::MatrixXd& a_T,
                   const Eigen::MatrixXd& a_psi) {
    // Setup filename
    std::string step_str = std::to_string(a_step);
    std::string filename = a_prefix + std::string(6 - step_str.length(), '0') + step_str + ".bin";

    std::cout << "Writing state to " << filename << std::endl;

    std::ofstream ofs;
    std::ostream* os = (std::ostream*)(&ofs);
    ofs.open(filename, std::ofstream::out | std::ofstream::binary);

    // Header
    (*os).write((char*)&m_M, sizeof(int));
    (*os).write((char*)&m_N, sizeof(int));
    (*os).write((char*)&m_t, sizeof(double));

    // Mesh
    (*os).write((char*)m_xx.data(), (sizeof(double) * m_xx.size()));
    (*os).write((char*)m_zz.data(), (sizeof(double) * m_zz.size()));

    // State
    (*os).write((char*)a_w.data(), sizeof(double) * (m_M + 1) * (m_N + 1));
    (*os).write((char*)a_S.data(), sizeof(double) * (m_M + 1) * (m_N + 1));
    (*os).write((char*)a_T.data(), sizeof(double) * (m_M + 1) * (m_N + 1));
    (*os).write((char*)a_psi.data(), sizeof(double) * (m_M + 1) * (m_N + 1));

    ofs.close();

    std::string xmlfilename = a_prefix + std::string(6 - step_str.length(), '0') + step_str + ".xmf";

    std::ofstream ofs_xml(xmlfilename);
    int offset = sizeof(int) + sizeof(int) + sizeof(double);
    if (ofs_xml.is_open()) {
      ofs_xml.precision(8);
      ofs_xml << "<?xml version=\"1.0\"?>\n";
      ofs_xml << "<Xdmf Version=\"3.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\">\n";
      ofs_xml << "  <Domain>\n";
      ofs_xml << "    <Grid Name=\"Grid\">\n";
      ofs_xml << "      <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"1 " << m_N+1 << " " << m_M+1 << "\"/>\n";
      ofs_xml << "      <Geometry GeometryType=\"VXVYVZ\">\n";
      ofs_xml << "        <DataItem Dimensions=\"" << m_M+1 << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Seek=\"" << offset << "\">\n";
      ofs_xml << "          ./" << filename << "\n";
      ofs_xml << "        </DataItem>\n";
      offset += sizeof(double) * (m_M + 1);
      ofs_xml << "        <DataItem Dimensions=\"" << m_N+1 << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Seek=\"" << offset << "\">\n";
      ofs_xml << "          ./" << filename << "\n";
      ofs_xml << "        </DataItem>\n";
      offset += sizeof(double) * (m_N + 1);
      ofs_xml << "        <DataItem Dimensions=\"1\" NumberType=\"Float\" Precision=\"8\">\n";
      ofs_xml << "          0.0" << "\n";
      ofs_xml << "        </DataItem>\n";
      ofs_xml << "      </Geometry>\n";
      ofs_xml << "      <Attribute Name=\"vorticity\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      ofs_xml << "        <DataItem Dimensions=\"1 " << m_N+1 << " " << m_M+1 << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Seek=\"" << offset << "\">\n";
      ofs_xml << "          ./" << filename << "\n";
      ofs_xml << "        </DataItem>\n";
      ofs_xml << "      </Attribute>\n";
      offset += sizeof(double) * (m_M + 1) * (m_N + 1);
      ofs_xml << "      <Attribute Name=\"salinity\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      ofs_xml << "        <DataItem Dimensions=\"1 " << m_N+1 << " " << m_M+1 << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Seek=\"" << offset << "\">\n";
      ofs_xml << "          ./" << filename << "\n";
      ofs_xml << "        </DataItem>\n";
      ofs_xml << "      </Attribute>\n";
      offset += sizeof(double) * (m_M + 1) * (m_N + 1);
      ofs_xml << "      <Attribute Name=\"temperature\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      ofs_xml << "        <DataItem Dimensions=\"1 " << m_N+1 << " " << m_M+1 << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Seek=\"" << offset << "\">\n";
      ofs_xml << "          ./" << filename << "\n";
      ofs_xml << "        </DataItem>\n";
      ofs_xml << "      </Attribute>\n";
      offset += sizeof(double) * (m_M + 1) * (m_N + 1);
      ofs_xml << "      <Attribute Name=\"streamfunction\" AttributeType=\"Scalar\" Center=\"Node\">\n";
      ofs_xml << "        <DataItem Dimensions=\"1 " << m_N+1 << " " << m_M+1 << "\" NumberType=\"Float\" Precision=\"8\" Format=\"Binary\" Seek=\"" << offset << "\">\n";
      ofs_xml << "          ./" << filename << "\n";
      ofs_xml << "        </DataItem>\n";
      ofs_xml << "      </Attribute>\n";
      ofs_xml << "    </Grid>\n";
      ofs_xml << "  </Domain>\n";
      ofs_xml << "</Xdmf>\n";
      ofs_xml.close();
    }

  }

  void load_state(std::string filename) {

    std::ifstream ifs;
    std::istream* is = (std::istream*)(&ifs);
    ifs.open(filename, std::ifstream::in | std::ifstream::binary);

    // Header
    int l_M{0};
    int l_N{0};
    double l_t{0.0};
    (*is).read((char*)&l_M, sizeof(int));
    (*is).read((char*)&l_N, sizeof(int));
    (*is).read((char*)&l_t, sizeof(double));
    if (l_M != m_M || l_N != m_N) {
      std::cout << "Error: incompatible mesh size\n";
      exit(0);
    }

    // Mesh
    Eigen::VectorXd l_xx = Eigen::VectorXd::Zero(m_M + 1);
    Eigen::VectorXd l_zz = Eigen::VectorXd::Zero(m_N + 1);
    (*is).read((char*)l_xx.data(), sizeof(double) * l_xx.size());
    (*is).read((char*)l_zz.data(), sizeof(double) * l_zz.size());

    // State
    (*is).read((char*)m_w.data(), sizeof(double) * (m_M + 1) * (m_N + 1));
    (*is).read((char*)m_S.data(), sizeof(double) * (m_M + 1) * (m_N + 1));
    (*is).read((char*)m_T.data(), sizeof(double) * (m_M + 1) * (m_N + 1));
    (*is).read((char*)m_psi.data(), sizeof(double) * (m_M + 1) * (m_N + 1));

    std::cout << "State loaded from " << filename << std::endl;
    ifs.close();
  }

  Eigen::MatrixXd sylvester_solve(const Eigen::MatrixXd& A,
                                  const Eigen::MatrixXd& B,
                                  const Eigen::MatrixXd& c) {
    // Solve the Sylvester eq. A x + x B = c using Bartels-Stewart algorithm

    Eigen::MatrixXd R_lp = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    Eigen::MatrixXd U_lp = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    schur_lapack(A, U_lp, R_lp);

    Eigen::MatrixXd S_lp = Eigen::MatrixXd::Zero(B.rows(), B.cols());
    Eigen::MatrixXd V_lp = Eigen::MatrixXd::Zero(B.rows(), B.cols());
    schur_lapack(B.conjugate().transpose(), V_lp, S_lp);

    Eigen::MatrixXd F = (U_lp.conjugate().transpose() * c) * V_lp;

    triangular_sylvester_lapack(R_lp, S_lp, F);
    Eigen::MatrixXd Y = F;

    return (U_lp * Y) * V_lp.conjugate().transpose();
  }

  void one_step(const Eigen::VectorXd &normal_noise) {

    std::cout << " Advance to t = " << m_t + m_dt << std::endl;

    Eigen::MatrixXd Fxpsi = m_Fx * m_psi;
    Eigen::MatrixXd psiFz = m_psi * m_FzT;

    // Scalar solves
    Eigen::MatrixXd DxT = m_Dx * m_T;
    Eigen::MatrixXd TDz = m_T * m_DzT;
    Eigen::MatrixXd DxS = m_Dx * m_S;
    Eigen::MatrixXd SDz = m_S * m_DzT;

    Eigen::MatrixXd QS = Fxpsi.cwiseProduct(SDz) - psiFz.cwiseProduct(DxS);
    Eigen::MatrixXd QT = Fxpsi.cwiseProduct(TDz) - psiFz.cwiseProduct(DxT);

    Eigen::MatrixXd CT = m_T + m_dt * (QT + m_FT);
    Eigen::MatrixXd CS = m_S + m_dt * (QS + m_FS) + sqrt(m_dt) * Snoise(normal_noise);

    Eigen::MatrixXd T_new = sylvester_solve(m_AT, m_BT, CT);
    Eigen::MatrixXd S_new = sylvester_solve(m_AS, m_BS, CS);

    double S_mean = S_new.mean();
    S_new = S_new * 1.0 / S_mean;

    // Momentum solves
    Eigen::MatrixXd src_w = ((m_Pr * m_Ra * m_Dx) * (T_new - S_new)) * m_Scorr;
    Eigen::MatrixXd adv_w = Fxpsi.cwiseProduct(m_w * m_FzT) - psiFz.cwiseProduct(m_Fx * m_w);
    Eigen::MatrixXd Cw = m_w + m_dt * (adv_w + src_w);

    Eigen::MatrixXd w_new = sylvester_solve(m_Aw, m_Bw, Cw);

    Eigen::MatrixXd psi_new = sylvester_solve(m_Fxx, m_FzzT, -w_new);

    m_psi = psi_new;
    m_T = T_new;
    m_S = S_new;
    m_w = w_new;

    m_t += m_dt;
    m_step += 1;
  }

  void one_step() {
    Eigen::VectorXd normal_zero = Eigen::VectorXd::Zero(m_K*2);
    one_step(normal_zero);
    if (m_step % 50 == 0) write_state();
  }

  void advance_trajectory(const double &t_end) {
    write_state();
    while (m_t < t_end) {
      Eigen::VectorXd normal_zero = Eigen::VectorXd::Zero(m_K*2);
      one_step(normal_zero);
      if (m_step % 100 == 0 || m_t >= t_end) write_state();
    }
  }
};
