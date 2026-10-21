/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/KalmanThreeDTrackHitCreationAlgorithm.h
 *
 *  @brief  Header file for the three dimensional hit creation algorithm class.
 *
 *  $Log: $
 */
#ifndef LAR_KALMAN_THREE_D_TRACK_HIT_CREATION_ALGORITHM_H
#define LAR_KALMAN_THREE_D_TRACK_HIT_CREATION_ALGORITHM_H 1

#include "Pandora/Algorithm.h"
#include "Pandora/AlgorithmTool.h"

#include "larpandoracontent/LArUtility/DistanceBasedKalmanFilter.h"

#include <vector>

namespace lar_content
{

class HitCreationBaseTool;
class ThreeDSlidingFitResult;

//------------------------------------------------------------------------------------------------------------------------------------------

/**
 *  @brief  KalmanThreeDTrackHitCreationAlgorithm::Algorithm class
 */
class KalmanThreeDTrackHitCreationAlgorithm : public ThreeDHitCreationAlgorithm
{
public:

    using ProtoHit = ThreeDHitCreationAlgorithm::ProtoHit;
    using TrajectorySample = ThreeDHitCreationAlgorithm::TrajectorySample;
    using ViewVectorMap = std::unordered_map<pandora::HitType, pandora::CartesianVector>;

    /**
     *  @brief  Default constructor
     */
    KalmanThreeDTrackHitCreationAlgorithm();

private:
    pandora::StatusCode Run();

    /**
     *  @brief  Calculate a 3D position from two or three views
     *
     *  @param  viewVectorMap the map of view (hit type) to 2D position
     *
     *  @return the calculated 3D position
     */
    pandora::CartesianVector Calculate3DPosition(ViewVectorMap &viewVectorMap) const;

    /**
     *  @brief  Get the position and direction to seed the Kalman Filter
     *
     *  @param  seedPosition to be filled with the seed position
     *  @param  seedDirection to be filled with the seed direction
     *  @param  backwardMode whether or not to run the Kalman Filter from the end of the track
     *
     *  @return the initialised Kalman Filter
     */
    void GetKalmanSeedPositionAndDirection(pandora::CartesianVector &seedPosition, pandora::CartesianVector &seedDirection, const bool backwardMode);

    /**
     *  @brief  Create and initialise a Kalman Filter
     *
     *  @param  backwardMode whether or not to run the Kalman Filter from the end of the track
     *
     *  @return the initialised Kalman Filter
     */
    DistanceBasedKalmanFilter3D InitialiseKalmanFilter(const bool backwardMode);

    /**
     *  @brief  Select potential 3D hits by using a Kalman filter on proto hits
     *
     *  @param  preToolProtoHitVector the set of 2D hits that have already been accepted by previous filters
     *  @param  protoHitVector the vector of protohits accepted by the Kalman Filters
     *  @param  potentialHitVector the vector of protohits that we want to add to the track
     *  @param  kalmanFilter the Kalman Filter object
     *
     *  @return whether we added any hits or not
     */
    bool KalmanFilterTreatment(const std::set<const pandora::CaloHit*> &preToolProtoHitVector, ProtoHitVector &protoHitVector, const ProtoHitVector &potentialProtoHits, DistanceBasedKalmanFilter3D &kalmanFilter) const;

    /**
     *  @brief  Calculate the pfo end points in 3D
     *
     *  @param  pPfo pointer to the pfo in question
     */
    void GetPfoEndPoints(const pandora::ParticleFlowObject *const pPfo);

    /**
     *  @brief  Calculate the pfo end points in a 2D view
     *
     *  @param  pPfo pointer to the pfo in question
     *  @param  vertices a map object to take the calculated 2D vertex
     *  @param  ends a map object to take the calculated 2D end point
     *  @param  view the 2D view in question
     */
    void GetPfoEndPoints2D(const pandora::ParticleFlowObject *const pPfo, ViewVectorMap &vertices, ViewVectorMap &ends, const pandora::HitType view);

    /**
     *  @brief  Calculate a step size based on the track length and number of 2D hits
     *
     *  @param  nHits the number of 2D CaloHits
     */
    void CalculateDynamicStepSize(const unsigned int nHits);

    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    float m_tolerance;                        ///< Distance tolerance for adding a hit to a track
    bool m_hasNeutrinoVertex;                 ///< Did we find a neutrino vertex
    pandora::CartesianVector m_nuVtxPosition; ///< Neutrino vertex position
    float m_kalmanStepSize;                   ///< Size of the Kalman filter steps in cm
    unsigned int m_stepsWithoutMatch;         ///< Number of Kalman steps without finding a hit to add before giving up
    bool m_useDynamicStepSize;                ///< Estimate the step size from the track length and number of hits
    pandora::CartesianVector m_pfoVertex;     ///< Vertex position of the current pfo
    pandora::CartesianVector m_pfoEnd;        ///< End position of the current pfo
};

} // namespace lar_content

#endif // #ifndef LAR_KALMAN_THREE_D_TRACK_HIT_CREATION_ALGORITHM_H
