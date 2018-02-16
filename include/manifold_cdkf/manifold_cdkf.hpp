#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <ros/time.h>

template <typename State, typename Input, typename ProcessNoiseVec>
class ManifoldCDKF
{
 public:
  using Scalar = typename State::Scalar_t;

  template <int N>
  using Vec = Eigen::Matrix<Scalar, N, 1>;

  template <int N, int M>
  using Mat = Eigen::Matrix<Scalar, N, M>;

  using StateCov = Mat<State::tangent_dim_, State::tangent_dim_>;

  using ProcessModel = std::function<State(
      State const &state, Input const &u, ProcessNoiseVec const &w, double dt)>;

  ManifoldCDKF(State const &state, StateCov const &state_covariance,
               ProcessModel const &process_model)
      : state_{state}, state_cov_{state_covariance}, process_model_{process_model}
  {
    if(!process_model_)
      throw std::invalid_argument("Invalid process_model");
  }

  State getState() const { return state_; }
  void setState(State const &state) { state_ = state; }

  StateCov getStateCovariance() const { return state_cov_; }
  void setStateCovariance(StateCov const &cov) { state_cov_ = cov; }

  ros::Time getStateTime() const { return prev_proc_update_time_; }
  void setStateTime(ros::Time const &time) { prev_proc_update_time_ = time; }

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
   * @param[in] time Timestamp for the current input
   * @param[in] u Input
   * @param[in] Q Input covariance
   * @param[in] debug Flag indicating whether to print internal debug messages
   *
   * @return True if the measurement update was successful else false
   */
  template <typename InputCov>
  std::enable_if_t<InputCov::RowsAtCompileTime != Eigen::Dynamic, bool>
  processUpdate(ros::Time const &time, Input const &u, InputCov const &Q,
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
  std::enable_if_t<MeasCovType::RowsAtCompileTime != Eigen::Dynamic, bool>
  measurementUpdate(MeasurementFunc const &measurement_func,
                    MeasurementType const &z, MeasCovType const &R,
                    bool debug = false);

  // TODO(Kartik): Add a MeasurementUpdateLinear function

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  using TangentVec = typename State::TangentVec;

  static constexpr unsigned int state_count_ = State::tangent_dim_;

  /**
   * Generate weights for the CDKF equations
   *
   * @param[in] L Number of sigma points that would be used
   *
   * @return Array containing the 4 weights, [wm0, wm1, wc1, wc2]
   */
  std::array<Scalar, 4> generateWeights(unsigned int L)
  {
    Scalar const h_sq = h_ * h_;
    Scalar const wm0 = (h_sq - L) / h_sq;
    Scalar const wm1 = 1 / (2 * h_sq);
    Scalar const wc1 = 1 / (4 * h_sq);
    Scalar const wc2 = (h_sq - 1) / (4 * h_sq * h_sq);
    return {wm0, wm1, wc1, wc2};
  }

  /**
   * Generate the sigma points given the noise covariance matrix
   *
   * @param[in] P Noise covariance matrix used for generating the sigma points
   */
  template <typename T>
  std::enable_if_t<T::RowsAtCompileTime != Eigen::Dynamic,
                   Mat<T::RowsAtCompileTime, 2 * T::RowsAtCompileTime + 1>>
  generateSigmaPoints(T const &P)
  {
    constexpr int L = T::RowsAtCompileTime;

    // Matrix square root
    Mat<L, L> const sqrtP = matrixSquareRoot(P);

    Mat<L, 2 * L + 1> Xaa;
    Xaa.col(0).setZero();
    Xaa.template block<L, L>(0, 1).noalias() = h_ * sqrtP;
    Xaa.template block<L, L>(0, L + 1).noalias() = -h_ * sqrtP;
    return Xaa;
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
  template <typename T, size_t N>
  T meanOfSigmaPoints(std::array<T, N> const &sigma_points, Scalar wm0,
                      Scalar wm1)
  {
    T mean = sigma_points[0];
    unsigned int iterations = 0;
    Vec<T::tangent_dim_> dx;
    do
    {
      dx = wm0 * (sigma_points[0] - mean);
      for(unsigned int i = 1; i < N; ++i)
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
  typename Derived::PlainObject matrixSquareRoot(
      Eigen::MatrixBase<Derived> const &mat)
  {
    // Try LLT first
    {
      Eigen::LLT<typename Derived::PlainObject> cov_chol{mat};
      if(cov_chol.info() == Eigen::Success)
        return cov_chol.matrixL();
    }
    // If not successful, try LDLT
    {
      Eigen::LDLT<typename Derived::PlainObject> cov_chol{mat};
      if(cov_chol.info() == Eigen::Success)
      {
        typename Derived::PlainObject const L = cov_chol.matrixL();
        auto const P = cov_chol.transpositionsP();
        auto const D2 = cov_chol.vectorD().array().sqrt().matrix().asDiagonal();
        return P.transpose() * L * D2;
      }
    }
    // Else give up
    return Derived::PlainObject::Zero(mat.rows(), mat.cols());
  }

  /// State
  State state_;

  /// State Covariance
  StateCov state_cov_;

  /// Process model
  ProcessModel const process_model_;

  /// State time (last process update time)
  ros::Time prev_proc_update_time_;

  /// Process update initialized indicator
  bool init_process_ = false;

  /// CDKF Parameter
  Scalar h_ = std::sqrt(Scalar(3));

  /// Threshold for change in state during sigma point mean computation
  Scalar sigma_points_mean_threshold_ = Scalar(1e-6);

  /// Maximum number of iterations for sigma point mean computation
  unsigned int sigma_points_mean_max_iterations_ = 5;
};

template <typename State, typename Input, typename ProcNoiseVec>
template <typename InputCov>
std::enable_if_t<InputCov::RowsAtCompileTime != Eigen::Dynamic, bool>
ManifoldCDKF<State, Input, ProcNoiseVec>::processUpdate(ros::Time const &time,
                                                        Input const &u,
                                                        InputCov const &Q,
                                                        bool debug)
{
  // Init Time
  if(!init_process_)
  {
    prev_proc_update_time_ = time;
    init_process_ = true;
    return false;
  }

  double dt = (time - prev_proc_update_time_).toSec();
  prev_proc_update_time_ = time;

  if(dt < 0)
    return false;

  constexpr unsigned int proc_noise_count = InputCov::RowsAtCompileTime;
  constexpr unsigned int L = state_count_ + proc_noise_count;

  if(debug) // For debugging
  {
    std::cout << "State:\n" << state_ << std::endl;
    std::cout << "state_cov:\n" << state_cov_ << std::endl;
  }

  // Generate sigma points
  auto const X = generateSigmaPoints(state_cov_);
  auto const W = generateSigmaPoints(Q);
  auto const[wm0, wm1, wc1, wc2] = generateWeights(L);

  std::array<State, 2 * L + 1> Xa;

  // Apply process model
  for(unsigned int k = 0; k <= 2 * state_count_; ++k)
  {
    Xa[k] = process_model_(state_ + X.col(k), u, W.col(0), dt);
  }
  for(unsigned int k = 1; k <= 2 * proc_noise_count; ++k)
  {
    Xa[2 * state_count_ + k] = process_model_(state_, u, W.col(k), dt);
  }

  state_ = meanOfSigmaPoints(Xa, wm0, wm1);

  // Covariance
  state_cov_.setZero();
  for(unsigned int k = 1; k <= L; ++k)
  {
    auto const x1 = Xa[k] - state_;
    auto const x2 = Xa[L + k] - state_;
    TangentVec const d1 = x1 - x2;
    TangentVec const d2 = x1 + x2;
    state_cov_ += wc1 * d1 * d1.transpose() + wc2 * d2 * d2.transpose();
  }
  return true;
}

template <typename State, typename Input, typename ProcessNoiseVec>
template <typename MeasurementFunc, typename MeasurementType,
          typename MeasCovType>
std::enable_if_t<MeasCovType::RowsAtCompileTime != Eigen::Dynamic, bool>
ManifoldCDKF<State, Input, ProcessNoiseVec>::measurementUpdate(
    MeasurementFunc const &measurement_func, MeasurementType const &z,
    MeasCovType const &R, bool debug)
{
  constexpr unsigned int L = state_count_;

  // Generate sigma points
  Mat<L, 2 *L + 1> const X = generateSigmaPoints(state_cov_);
  auto const[wm0, wm1, wc1, wc2] = generateWeights(L);

  if(debug)
  {
    std::cout << "Pa:\n" << state_cov_ << std::endl;
    std::cout << "X:\n" << X.template leftCols<L + 1>() << std::endl;
  }

  // Apply measurement model
  std::array<MeasurementType, 2 * L + 1> Zaa;
  for(unsigned int k = 0; k <= 2 * L; k++)
  {
    Zaa[k] = measurement_func(state_ + X.col(k));
    if(debug)
      std::cout << "Z[" << k << "]:\n" << Zaa[k] << std::endl;
  }

  auto const z_pred = meanOfSigmaPoints(Zaa, wm0, wm1);

  // Covariance
  MeasCovType Pzz = MeasCovType::Zero();
  Mat<state_count_, MeasurementType::tangent_dim_> Pxz =
      Mat<state_count_, MeasurementType::tangent_dim_>::Zero();
  for(unsigned int k = 1; k <= L; k++)
  {
    auto const z1 = Zaa[k] - z_pred;
    auto const z2 = Zaa[L + k] - z_pred;
    Vec<MeasurementType::tangent_dim_> const dz1 = z1 - z2;
    Vec<MeasurementType::tangent_dim_> const dz2 = z1 + z2;
    Pzz += wc1 * dz1 * dz1.transpose() + wc2 * dz2 * dz2.transpose();
    Pxz += wm1 * X.col(k) * dz1.transpose();
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
    Mat<state_count_, MeasurementType::tangent_dim_> const K =
        Pxz * Pzz.inverse();
    std::cout << "K:\n" << K << "\n";
    std::cout << "inno: " << inno.transpose() << "\n";
  }

  auto const Pzz_chol = Pzz.llt();
  if(Pzz_chol.info() != Eigen::Success)
    throw std::domain_error("Pzz is not positive definite!");

  TangentVec const dx = Pxz * Pzz_chol.solve(inno);

  // Covariance around the old mean
  state_cov_ -= Pxz * Pzz_chol.solve(Pxz.transpose());

  if(debug)
  {
    std::cout << "dx: " << dx.transpose() << "\n";
    std::cout << "state_cov:\n" << state_cov_ << "\n";
  }

  // Compute the mean and move the covariance to be around the new mean
  Mat<L, 2 * L + 1> const X_new = generateSigmaPoints(state_cov_);

  std::array<State, 2 * L + 1> Xa;
  for(unsigned int k = 0; k <= 2 * L; k++)
  {
    Xa[k] = state_ + (dx + X_new.col(k));
  }

  // Get mean
  state_ = meanOfSigmaPoints(Xa, wm0, wm1);

  // Calculate covariance around the new mean
  state_cov_.setZero();
  for(unsigned int k = 1; k <= L; ++k)
  {
    auto const x1 = Xa[k] - state_;
    auto const x2 = Xa[L + k] - state_;
    TangentVec const d1 = x1 - x2;
    TangentVec const d2 = x1 + x2;
    state_cov_ += wc1 * d1 * d1.transpose() + wc2 * d2 * d2.transpose();
  }

  return true;
}
