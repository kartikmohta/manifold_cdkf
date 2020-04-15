#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

template <typename State, typename Input, typename ProcessNoiseVec>
class ManifoldCDKF
{
 public:
  using Scalar = typename State::Scalar;

  template <int N>
  using Vec = Eigen::Matrix<Scalar, N, 1>;

  template <int N, int M>
  using Mat = Eigen::Matrix<Scalar, N, M>;

  using StateCov = Mat<State::tangent_dim_, State::tangent_dim_>;

  using ProcessModelFunc = std::function<State(
      State const &state, Input const &u, ProcessNoiseVec const &w, Scalar dt)>;

  ManifoldCDKF(State const &state, StateCov const &state_covariance,
               ProcessModelFunc const &process_model)
      : state_{state},
        state_cov_{state_covariance},
        process_model_{process_model}
  {
    if(!process_model_)
    {
      throw std::invalid_argument("Invalid process_model");
    }
  }

  /// Get current state
  State getState() const { return state_; }
  /// Set current state
  void setState(State const &state) { state_ = state; }

  /// Get current state covariance
  StateCov getStateCovariance() const { return state_cov_; }
  /// Set current state covariance
  void setStateCovariance(StateCov const &cov) { state_cov_ = cov; }

  /**
   * Set the CDKF sigma point spread parameter. The sigma points are generated
   * as \f$ x \pm h \sqrt{P} \f$. The recommended value for gaussian noise is
   * \f$ \sqrt{3} \f$.
   *
   * @param[in] h Sigma point spread parameter
   */
  void setParameter(Scalar h) { h_ = h; }

  void setSigmaPointMeanParameters(Scalar threshold,
                                   unsigned int max_iterations)
  {
    sigma_points_mean_threshold_ = threshold;
    sigma_points_mean_max_iterations_ = max_iterations;
  }

  /**
   * Run the process update using the provided process model, input and input
   * covariance.
   *
   * @param[in] dt Time duration since last processUpdate
   * @param[in] u Input
   * @param[in] Q Input covariance
   * @param[in] debug Flag indicating whether to print internal debug messages
   *
   * @return True if the process update was successful else false
   */
  template <typename InputCov>
  bool processUpdate(Scalar const dt, Input const &u, InputCov const &Q,
                     bool debug = false);

  /**
   * Run the measurement update using the provided measurement model , current
   * measurement and measurement covariance.
   *
   * @param[in] measurement_func takes in the state and returns the expected
   * measurement with type MeasurementType
   * @param[in] z Current measurement
   * @param[in] R Measurement covariance
   * @param[in] debug Flag indicating whether to print internal debug messages
   *
   * @return True if the measurement update was successful else false
   */
  template <typename MeasurementFunc, typename MeasurementType,
            typename MeasCovType>
  bool measurementUpdate(MeasurementFunc const &measurement_func,
                         MeasurementType const &z, MeasCovType const &R,
                         bool debug = false);

  // TODO(Kartik): Add a MeasurementUpdateLinear function

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  /**
   * Generate weights for the CDKF equations
   *
   * @param[in] L Number of sigma points that would be used
   *
   * @return Array containing the 4 weights, [wm0, wm1, wc1, wc2]
   */
  std::array<Scalar, 4> generateWeights(unsigned int L) const
  {
    Scalar const h_sq = h_ * h_;
    Scalar const wm0 = (h_sq - L) / h_sq;
    Scalar const wm1 = 1 / (2 * h_sq);
    Scalar const wc1 = 1 / (4 * h_sq);
    Scalar const wc2 = (h_sq - 1) / (4 * h_sq * h_sq);
    return {wm0, wm1, wc1, wc2};
  }

  /**
   * Compute the mean of the sigma points
   *
   * @param[in] sigma_points Input sigma points
   * @param[in] wm0 Weight wm0 for CDKF
   * @param[in] wm1 Weight wm1 for CDKF
   *
   * @return Mean of the sigma points
   */
  template <typename T>
  T meanOfSigmaPoints(
      std::vector<T, Eigen::aligned_allocator<T>> const &sigma_points,
      Scalar wm0, Scalar wm1) const
  {
    T mean = sigma_points[0];
    unsigned int iterations = 0;
    Vec<T::tangent_dim_> dx;
    do
    {
      dx = wm0 * (sigma_points[0] - mean);
      for(unsigned int i = 1; i < sigma_points.size(); ++i)
      {
        dx += wm1 * (sigma_points[i] - mean);
      }
      mean = mean + dx;
      iterations += 1;
      // std::cout << "meanOfSigmaPoints: dx.norm(): " << dx.norm()
      //           << ", iterations: " << iterations << "\n";
    } while((dx.squaredNorm() >
             sigma_points_mean_threshold_ * sigma_points_mean_threshold_) &&
            (iterations < sigma_points_mean_max_iterations_));

    return mean;
  }

  /**
   * Calculate the matrix square root of a positive (semi-)definite matrix.
   *
   * @param[in] mat The positive (semi-)definite matrix
   *
   * @return The square root of the input matrix which satisfies \f$ret *
   * ret^{T} = mat\f$
   */
  template <typename Derived>
  static typename Derived::PlainObject matrixSquareRoot(
      Eigen::MatrixBase<Derived> const &mat)
  {
    // Try LLT first
    {
      Eigen::LLT<typename Derived::PlainObject> const cov_chol{mat};
      if(cov_chol.info() == Eigen::Success)
      {
        return cov_chol.matrixL();
      }
    }
    // If not successful, try LDLT
    {
      Eigen::LDLT<typename Derived::PlainObject> const cov_chol{mat};
      if(cov_chol.info() == Eigen::Success)
      {
        typename Derived::PlainObject const L = cov_chol.matrixL();
        auto const &P = cov_chol.transpositionsP();
        auto const D = cov_chol.vectorD().array();
        if((D >= 0).all())
        {
          auto const D2 = D.sqrt().matrix().asDiagonal();
          return P.transpose() * L * D2;
        }
      }
    }
    // If not successful, try eigen-decomposition with slightly inflated
    // eigenvalues
    {
      Eigen::SelfAdjointEigenSolver<typename Derived::PlainObject> const cov_es{
          mat};
      auto eigvals = cov_es.eigenvalues().array().eval();
      auto const lowest_eval = eigvals(0);
      if(lowest_eval < 0)
      {
        eigvals -= lowest_eval;
      }
      return cov_es.eigenvectors() * eigvals.sqrt().matrix().asDiagonal() *
             cov_es.eigenvectors().transpose();
    }
  }

  /// State
  State state_;

  /// State Covariance
  StateCov state_cov_;

  /// Process model
  ProcessModelFunc const process_model_;

  /// CDKF Parameter
  Scalar h_ = std::sqrt(Scalar(3));

  /// Threshold for change in state during sigma point mean computation
  Scalar sigma_points_mean_threshold_ = Scalar(1e-6);

  /// Maximum number of iterations for sigma point mean computation
  unsigned int sigma_points_mean_max_iterations_ = 5;
};

template <typename State, typename Input, typename ProcNoiseVec>
template <typename InputCov>
bool ManifoldCDKF<State, Input, ProcNoiseVec>::processUpdate(Scalar const dt,
                                                             Input const &u,
                                                             InputCov const &Q,
                                                             bool debug)
{
  if(dt < 0)
  {
    return false;
  }

  if(debug) // For debugging
  {
    std::cout << std::string(32, '=') << " Process Update "
              << std::string(32, '=') << "\n";
    std::cout << "State:\n" << state_ << std::endl;
    std::cout << "state_cov:\n" << state_cov_ << std::endl;
  }

  auto const proc_noise_count = static_cast<unsigned int>(Q.rows());
  auto const L = State::tangent_dim_ + proc_noise_count;

  // Generate sigma points
  auto const X = (h_ * matrixSquareRoot(state_cov_)).eval();
  auto const W = (h_ * matrixSquareRoot(Q)).eval();
  auto const weights = generateWeights(L);
  auto const wm0 = weights[0], wm1 = weights[1], wc1 = weights[2],
             wc2 = weights[3];
  // auto const [wm0, wm1, wc1, wc2] = generateWeights(L); // In C++17

  std::vector<State, Eigen::aligned_allocator<State>> Xa(2 * L + 1);

  // Apply process model
  Xa[0] = process_model_(state_, u, ProcNoiseVec::Zero(), dt);
  for(unsigned int k = 1; k <= State::tangent_dim_; ++k)
  {
    Xa[k] = process_model_(state_ + X.col(k - 1), u, ProcNoiseVec::Zero(), dt);
    Xa[L + k] =
        process_model_(state_ + -X.col(k - 1), u, ProcNoiseVec::Zero(), dt);
  }
  for(unsigned int k = 1; k <= proc_noise_count; ++k)
  {
    Xa[State::tangent_dim_ + k] = process_model_(state_, u, W.col(k - 1), dt);
    Xa[L + State::tangent_dim_ + k] =
        process_model_(state_, u, -W.col(k - 1), dt);
  }

  state_ = meanOfSigmaPoints(Xa, wm0, wm1);

  // Covariance
  state_cov_.setZero();
  for(unsigned int k = 1; k <= L; ++k)
  {
    auto const x1 = Xa[k] - Xa[0];
    auto const x2 = Xa[L + k] - Xa[0];
    auto const d1 = x1 - x2;
    auto const d2 = x1 + x2;
    state_cov_ += wc1 * d1 * d1.transpose() + wc2 * d2 * d2.transpose();
  }

  if(debug)
  {
    std::cout << std::string(80, '=') << "\n";
  }

  return true;
}

template <typename State, typename Input, typename ProcessNoiseVec>
template <typename MeasurementFunc, typename MeasurementType,
          typename MeasCovType>
bool ManifoldCDKF<State, Input, ProcessNoiseVec>::measurementUpdate(
    MeasurementFunc const &measurement_func, MeasurementType const &z,
    MeasCovType const &R, bool debug)
{
  if(debug)
  {
    std::cout << std::string(30, '=') << " Measurement Update "
              << std::string(30, '=') << "\n";
  }

  constexpr unsigned int L = State::tangent_dim_;

  // Generate sigma points
  auto const X = (h_ * matrixSquareRoot(state_cov_)).eval();
  auto const weights = generateWeights(L);
  auto const wm0 = weights[0], wm1 = weights[1], wc1 = weights[2],
             wc2 = weights[3];
  // auto const [wm0, wm1, wc1, wc2] = generateWeights(L); // In C++17

  if(debug)
  {
    std::cout << "Pa:\n" << state_cov_ << std::endl;
    std::cout << "X:\n" << X.template leftCols<L + 1>() << std::endl;
  }

  // Apply measurement model
  std::vector<MeasurementType, Eigen::aligned_allocator<MeasurementType>> Zaa(
      2 * L + 1);

  Zaa[0] = measurement_func(state_);
  for(unsigned int k = 1; k <= L; k++)
  {
    Zaa[k] = measurement_func(state_ + X.col(k - 1));
    Zaa[L + k] = measurement_func(state_ + -X.col(k - 1));
  }
  if(debug)
  {
    for(unsigned int k = 0; k < 2 * L + 1; ++k)
    {
      std::cout << "Z[" << k << "]:\n" << Zaa[k] << std::endl;
    }
  }

  auto const z_pred = meanOfSigmaPoints(Zaa, wm0, wm1);

  // Covariance
  auto Pzz = MeasCovType::Zero(R.rows(), R.cols()).eval();
  auto Pxz =
      Mat<State::tangent_dim_, MeasurementType::tangent_dim_>::Zero().eval();
  for(unsigned int k = 1; k <= L; k++)
  {
    auto const z1 = Zaa[k] - Zaa[0];
    auto const z2 = Zaa[L + k] - Zaa[0];
    auto const dz1 = z1 - z2;
    auto const dz2 = z1 + z2;
    Pzz += wc1 * dz1 * dz1.transpose() + wc2 * dz2 * dz2.transpose();
    Pxz += wm1 * X.col(k - 1) * dz1.transpose();
  }
  Pzz += R;

  // Innovation
  auto const inno = decltype(z_pred){z} - z_pred;
  if(debug)
  {
    std::cout << "z:\n" << z << "\n";
    std::cout << "z_pred:\n" << z_pred << "\n";
    std::cout << "Pxz:\n" << Pxz << "\n";
    std::cout << "Pzz:\n" << Pzz << "\n";
    // Kalman Gain;
    auto const K = Pxz * Pzz.inverse();
    std::cout << "K:\n" << K << "\n";
    std::cout << "inno: " << inno.transpose() << "\n";
  }

  auto dx = State::TangentVec::Zero().eval();

  auto const Pzz_llt = Pzz.llt();
  if(Pzz_llt.info() == Eigen::Success) // Pzz is positive-definite
  {
    dx = Pxz * Pzz_llt.solve(inno);
    // Covariance around the old mean
    state_cov_ -= Pxz * Pzz_llt.solve(Pxz.transpose());
  }
  else
  {
    // Pzz is not positive definite, try LDLT
    auto const Pzz_ldlt = Pzz.ldlt();
    if(Pzz_ldlt.info() == Eigen::Success) // Pzz is positive semi-definite
    {
      dx = Pxz * Pzz_ldlt.solve(inno);
      // Covariance around the old mean
      state_cov_ -= Pxz * Pzz_ldlt.solve(Pxz.transpose());
    }
  }

  // Make sure that diagonal elements of state_cov_ are non-negative
  for(unsigned int i = 0; i < State::tangent_dim_; ++i)
  {
    if(state_cov_(i, i) < 0)
    {
      for(unsigned int j = 0; j < State::tangent_dim_; ++j)
      {
        state_cov_(i, j) = 0;
        state_cov_(j, i) = 0;
      }
    }
  }

  if(debug)
  {
    std::cout << "dx: " << dx.transpose() << "\n";
    std::cout << "state_cov:\n" << state_cov_ << "\n";
  }

  // Compute the mean and move the covariance to be around the new mean
  auto const X_new = (h_ * matrixSquareRoot(state_cov_)).eval();
  if(debug)
  {
    std::cout << "X_new:\n" << X_new << "\n";
  }

  std::vector<State, Eigen::aligned_allocator<State>> Xa(2 * L + 1);
  Xa[0] = state_ + dx;
  for(unsigned int k = 1; k <= L; k++)
  {
    Xa[k] = state_ + (dx + X_new.col(k - 1));
    Xa[L + k] = state_ + (dx - X_new.col(k - 1));
  }

  // Get mean
  state_ = meanOfSigmaPoints(Xa, wm0, wm1);

  // Calculate covariance around the new mean
  state_cov_.setZero();
  for(unsigned int k = 1; k <= L; ++k)
  {
    auto const x1 = Xa[k] - Xa[0];
    auto const x2 = Xa[L + k] - Xa[0];
    auto const d1 = x1 - x2;
    auto const d2 = x1 + x2;
    state_cov_ += wc1 * d1 * d1.transpose() + wc2 * d2 * d2.transpose();
  }

  if(debug)
  {
    std::cout << std::string(80, '=') << "\n";
  }

  return true;
}
