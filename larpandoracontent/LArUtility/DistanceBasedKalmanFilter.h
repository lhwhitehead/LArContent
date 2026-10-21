/**
 *  @file   larpandoracontent/LArUtility/DistanceBasedKalmanFilter.h
 *
 *  @brief  Kalman filter for predicting 3D trajectories.
 *
 *  $Log: $
 */
#ifndef LAR_DISTANCE_BASED_KALMAN_FILTER_H
#define LAR_DISTANCE_BASED_KALMAN_FILTER_H 1

#include "Eigen/Dense"

#include "Objects/CartesianVector.h"

namespace lar_content
{

/**
 *  @brief  DistanceBasedKalmanFilter class
 */
template <int DIM>
class DistanceBasedKalmanFilter
{
    static_assert(DIM > 0, "DistanceBasedKalmanFilter dimensionality must be positive");

public:
    using StateVector = Eigen::Matrix<double, 2 * DIM, 1>;
    using MeasurementVector = Eigen::Matrix<double, DIM, 1>;
    using StateMatrix = Eigen::Matrix<double, 2 * DIM, 2 * DIM>;
    using MeasurementMatrix = Eigen::Matrix<double, DIM, 2 * DIM>;
    using CovarianceMatrix = Eigen::Matrix<double, DIM, DIM>;
    using PositionVector = Eigen::Matrix<double, DIM, 1>;
    using DirectionVector = Eigen::Matrix<double, DIM, 1>;

    /**
     *  @brief  Default constructor
     *
     *  @param  position Initial position
     *  @param  direction Initial direction
     *  @param  positionError Uncertainty on initial position
     *  @param  directionError Uncertainty on initial direction
     */
    DistanceBasedKalmanFilter(const PositionVector &position, const DirectionVector &direction, const PositionVector &positionError, const DirectionVector &directionError);

    ~DistanceBasedKalmanFilter() = default;

    DistanceBasedKalmanFilter(const DistanceBasedKalmanFilter &) = default;
    DistanceBasedKalmanFilter &operator=(const DistanceBasedKalmanFilter &) = default;
    DistanceBasedKalmanFilter(DistanceBasedKalmanFilter &&) = default;
    DistanceBasedKalmanFilter &operator=(DistanceBasedKalmanFilter &&) = default;

    /**
     * @brief  Uses the state transition matrix to predict the next state
     */
    void Predict();

    /**
     * @brief  Uses the state transition matrix to predict the next state
     *
     * @param  ds the step size (distance, not time)
     */
    void Predict(double ds);

    /**
     * @brief  Uses the measurement matrix to update the state estimate
     *
     * @param  z The position to use for updating the state estimate
     */
    void Update(const MeasurementVector &measurement, const MeasurementVector &measurementError);

    /**
     * @brief  Get the current state vector
     *
     * @return The current state vector
     */
    const StateVector &GetState() const;

    /**
     * @brief  Get the current position vector
     *
     * @return The current postion vector
     */
    const PositionVector GetPosition() const;

    /**
     * @brief  Get the current position vector
     *
     * @return The current postion vector as a pandora::CartesianVector
     */
    const pandora::CartesianVector GetPositionAsCartesianVector() const;

    /**
     * @brief  Get the current predicted position vector
     *
     * @return The current predicted postion vector
     */
    const PositionVector GetPredictedPosition() const;

    /**
     * @brief  Get the current predicted position vector
     *
     * @return The current predicted postion vector as a pandora::CartesianVector
     */
    const pandora::CartesianVector GetPredictedPositionAsCartesianVector() const;

    /**
     * @brief  Get the current direction vector
     *
     * @return The current direction vector
     */
    const PositionVector GetDirection() const;

    /**
     * @brief  Get the current direction vector
     *
     * @return The current direction vector as a pandora::CartesianVector
     */
    const pandora::CartesianVector GetDirectionAsCartesianVector() const;

    /**
     * @brief  Get the current predicted direction vector
     *
     * @return The current predicted direction vector
     */
    const PositionVector GetPredictedDirection() const;

    /**
     * @brief  Get the current predicted direction vector
     *
     * @return The current predicted direction vector as a pandora::CartesianVector
     */
    const pandora::CartesianVector GetPredictedDirectionAsCartesianVector() const;

    /**
     * @brief  Get the temporary state vector
     *
     * @return The temporary state vector
     */
    const StateVector &GetTemporaryState() const;

    /**
     * @brief  Get the Mahalanobis distance between the predicted state and a measurement
     *
     * @param  z The measurement vector
     * @param  useTemp Whether to use the temporary state matrix
     *
     * @return The Mahalanobis distance
     */
    double GetMahalanobisDistance(const MeasurementVector &z, const bool useTemp = true) const;

private:
    double m_ds;            ///< Step along the direction
    int m_stateSize;        ///< Size of the state vector
    int m_measurementSize;  ///  Size of the measurement vector
    StateVector m_x;        ///< State vector (tracks position and direction)
    StateMatrix m_P;        ///< Covariance matrix
    StateMatrix m_F;        ///< State transition matrix
    MeasurementMatrix m_H;  ///< Measurement matrix
    CovarianceMatrix m_R;   ///< Measurement covariance matrix
    StateMatrix m_Q;        ///< Process covariance matrix
    StateVector m_xTemp;    ///< Temporary state vector
    StateMatrix m_PTemp;    ///< Temporary covariance matrix
};

//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
DistanceBasedKalmanFilter<DIM>::DistanceBasedKalmanFilter(const DistanceBasedKalmanFilter<DIM>::PositionVector &position, const DistanceBasedKalmanFilter<DIM>::DirectionVector &direction, const DistanceBasedKalmanFilter<DIM>::PositionVector &positionError, const DistanceBasedKalmanFilter<DIM>::DirectionVector &directionError):
    m_stateSize{2 * DIM},
    m_measurementSize{DIM}
{
    m_x = StateVector::Zero();
    m_x.head(DIM) = position;
    m_x.tail(DIM) = direction.normalized(); // Make sure it is a direction vector

    // Create a diagonal covariance matrix from the provided uncertainties
    m_P = StateMatrix::Zero();
    m_P.template topLeftCorner<DIM, DIM>() = positionError.array().square().matrix().asDiagonal();
    m_P.template bottomRightCorner<DIM, DIM>() = directionError.array().square().matrix().asDiagonal();

    m_H = MeasurementMatrix::Zero();
    m_H.leftCols(DIM) = Eigen::Matrix<double, DIM, DIM>::Identity();

    m_xTemp = m_x;
    m_PTemp = m_P;
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
void DistanceBasedKalmanFilter<DIM>::Predict(double ds)
{
    m_ds = ds;

    m_F = StateMatrix::Identity();
    m_F.topRightCorner(DIM, DIM) = Eigen::Matrix<double, DIM, DIM>::Identity() * m_ds;

    Eigen::Matrix<double, 2 * DIM, DIM> G = Eigen::Matrix<double, 2 * DIM, DIM>::Zero();
    G.topRows(DIM) = Eigen::Matrix<double, DIM, DIM>::Identity() * (0.5 * m_ds * m_ds);
    G.bottomRows(DIM) = Eigen::Matrix<double, DIM, DIM>::Identity() * m_ds;

    m_Q = G * G.transpose(); // * m_processVariance;
    m_xTemp = m_F * m_x;
    m_PTemp = m_F * m_P * m_F.transpose() + m_Q;
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
void DistanceBasedKalmanFilter<DIM>::Update(const DistanceBasedKalmanFilter<DIM>::MeasurementVector &measurement,
                                            const DistanceBasedKalmanFilter<DIM>::MeasurementVector &measurementError)
{
    m_R = CovarianceMatrix::Zero();
    m_R.diagonal() = measurementError.array().square().matrix();

    m_x = m_xTemp;
    m_P = m_PTemp;
    MeasurementVector y = measurement - m_H * m_x;
    auto Ht = m_H.transpose();
    auto S = m_H * m_P * Ht + m_R;
    auto K = m_P * Ht * S.inverse();
    m_x += K * y;
    m_P = (StateMatrix::Identity() - K * m_H) * m_P;

    m_x.tail(DIM) = m_x.tail(DIM).normalized();
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename DistanceBasedKalmanFilter<DIM>::StateVector &DistanceBasedKalmanFilter<DIM>::GetState() const
{
    return m_x;
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename DistanceBasedKalmanFilter<DIM>::PositionVector DistanceBasedKalmanFilter<DIM>::GetPosition() const
{
    return m_x.head(DIM);
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename pandora::CartesianVector DistanceBasedKalmanFilter<DIM>::GetPositionAsCartesianVector() const
{
    auto const pos{m_x.head(DIM)};
    return pandora::CartesianVector(pos.x(), pos.y(), pos.z());
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename DistanceBasedKalmanFilter<DIM>::PositionVector DistanceBasedKalmanFilter<DIM>::GetPredictedPosition() const
{
    return m_xTemp.head(DIM);
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename pandora::CartesianVector DistanceBasedKalmanFilter<DIM>::GetPredictedPositionAsCartesianVector() const
{
    auto const pos{m_xTemp.head(DIM)};
    return pandora::CartesianVector(pos.x(), pos.y(), pos.z());
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename DistanceBasedKalmanFilter<DIM>::PositionVector DistanceBasedKalmanFilter<DIM>::GetDirection() const
{
    return m_x.tail(DIM).normalized();
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename pandora::CartesianVector DistanceBasedKalmanFilter<DIM>::GetDirectionAsCartesianVector() const
{
    auto const dir{m_x.tail(DIM).normalized()};
    return pandora::CartesianVector(dir.x(), dir.y(), dir.z());
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename DistanceBasedKalmanFilter<DIM>::PositionVector DistanceBasedKalmanFilter<DIM>::GetPredictedDirection() const
{
    return m_xTemp.tail(DIM).normalized();
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename pandora::CartesianVector DistanceBasedKalmanFilter<DIM>::GetPredictedDirectionAsCartesianVector() const
{
    auto const dir{m_xTemp.tail(DIM).normalized()};
    return pandora::CartesianVector(dir.x(), dir.y(), dir.z());
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
const typename DistanceBasedKalmanFilter<DIM>::StateVector &DistanceBasedKalmanFilter<DIM>::GetTemporaryState() const
{
    return m_xTemp;
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <int DIM>
double DistanceBasedKalmanFilter<DIM>::GetMahalanobisDistance(const DistanceBasedKalmanFilter<DIM>::MeasurementVector &z, const bool useTemp) const
{
    const StateVector &x = useTemp ? m_xTemp : m_x;
    const StateMatrix &P = useTemp ? m_PTemp : m_P;
    MeasurementVector y = z - m_H * x;
    auto Ht = m_H.transpose();
    auto S = m_H * P * Ht + m_R;
    return std::sqrt(y.transpose() * S.inverse() * y);
}

using DistanceBasedKalmanFilter2D = DistanceBasedKalmanFilter<2>;
using DistanceBasedKalmanFilter3D = DistanceBasedKalmanFilter<3>;

} // namespace lar_content

#endif // #ifndef LAR_DISTANCE_BASED_KALMAN_FILTER_H
