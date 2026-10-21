/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/KalmanThreeDTrackHitCreationAlgorithm.cc
 *
 *  @brief  Implementation of the three dimensional hit creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/KalmanThreeDTrackHitCreationAlgorithm.h"
#include "larpandoracontent/LArUtility/DistanceBasedKalmanFilter.h"
#include "larpandoracontent/LArUtility/KalmanFilter.h"

#include <algorithm>
#include <cmath>

using namespace pandora;

namespace lar_content
{

KalmanThreeDTrackHitCreationAlgorithm::KalmanThreeDTrackHitCreationAlgorithm() : ThreeDHitCreationAlgorithm(),
    m_tolerance(1.f),
    m_hasNeutrinoVertex(false),
    m_nuVtxPosition(0.f, 0.f, 0.f),
    m_kalmanStepSize(0.05f),
    m_stepsWithoutMatch(25),
    m_useDynamicStepSize(true),
    m_pfoVertex(0.f, 0.f, 0.f),
    m_pfoEnd(0.f, 0.f, 0.f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode KalmanThreeDTrackHitCreationAlgorithm::Run()
{
    std::cout << "Using detailed geometry? " << m_useDetailedGeometry << std::endl;

    const PfoList *pPfoList(nullptr);
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, m_inputPfoListName, pPfoList));

    if (!pPfoList || pPfoList->empty())
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "KalmanThreeDTrackHitCreationAlgorithm: unable to find pfo list " << m_inputPfoListName << std::endl;

        return STATUS_CODE_SUCCESS;
    }

    // Get the neutrino vertex to see if we need to be going backwards. This works for primaries... but could
    // fail for later tiers
    const VertexList *pVertexList(nullptr);
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, "NeutrinoVertices3D", pVertexList));

    if (!pVertexList || pVertexList->empty())
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "KalmanThreeDTrackHitCreationAlgorithm: unable to find pfo list NeutrinoVertices3D" << std::endl;
    }
    else
    {
        m_nuVtxPosition = pVertexList->front()->GetPosition();
        m_hasNeutrinoVertex = true;
        std::cout << "Got neutrino vertex at position " << m_nuVtxPosition << std::endl;
    }

    CaloHitList allNewThreeDTrackHits;

    PfoVector pfoVector(pPfoList->begin(), pPfoList->end());
    std::sort(pfoVector.begin(), pfoVector.end(), LArPfoHelper::SortByNHits);

    for (const ParticleFlowObject *const pPfo : pfoVector)
    {
        if (!LArPfoHelper::IsTrack(pPfo))
            continue;

        this->GetPfoEndPoints(pPfo);

        const unsigned int nPfoHits{LArPfoHelper::GetNumberOfTwoDHits(pPfo)};
        std::cout << " ====== Processing Pfo " << pPfo << " with " << nPfoHits << " 2D hits ======" << std::endl;

        ProtoHitVector allGoodProtoHits;

        bool keepIterating{true};
        unsigned int i{0};
        float fractionUsed{0.f};
        while (keepIterating)
        {
            ++i;

            bool addedHits{false};
            for (HitCreationBaseTool *const pHitCreationTool : m_algorithmToolVector)
            {
                CaloHitVector remainingTwoDHits;
                this->SeparateTwoDHits(pPfo, allGoodProtoHits, remainingTwoDHits);
//                std::cout << "Running tool with pointer " << pHitCreationTool->GetType() << ", " << allGoodProtoHits.size() << " 3D hits and " << remainingTwoDHits.size() << " 2D hits remaining" << std::endl;

                if (remainingTwoDHits.empty())
                {
                    keepIterating = false;
                    break;
                }

                std::set<const CaloHit*> preToolCaloHitSet;
                for (const ProtoHit &hit : allGoodProtoHits)
                    preToolCaloHitSet.insert(hit.GetParentCaloHit2D());

                ProtoHitVector protoHitVector;
                pHitCreationTool->Run(this, pPfo, remainingTwoDHits, protoHitVector);
//                std::cout << "Tool " << pHitCreationTool << " created " << protoHitVector.size() << " potential hits" << std::endl;

                if (protoHitVector.empty())
                    continue;

                if (m_useDynamicStepSize)
                {
                    unsigned int nHitsAvailable(allGoodProtoHits.size());
                    for (auto const &hit : protoHitVector)
                        if (!preToolCaloHitSet.count(hit.GetParentCaloHit2D()))
                            ++nHitsAvailable;
                    this->CalculateDynamicStepSize(nHitsAvailable);
//                    std::cout << "Calculating step size with " << nHitsAvailable << " available hits... " << m_kalmanStepSize << std::endl;
                }
                // Since the vertex region can be messy, run the filter in reverse first
                DistanceBasedKalmanFilter3D kalmanFilterBackward(this->InitialiseKalmanFilter(true));
                const bool addedHitsBackward{this->KalmanFilterTreatment(preToolCaloHitSet, allGoodProtoHits, protoHitVector, kalmanFilterBackward)};
//                std::cout << "Backward pass added " << allGoodProtoHits.size() - preToolCaloHitSet.size() << " hits" << std::endl;    

                std::set<const CaloHit*> postBackwardCaloHitSet;
                for (const ProtoHit &hit : allGoodProtoHits)
                    postBackwardCaloHitSet.insert(hit.GetParentCaloHit2D());
    
                DistanceBasedKalmanFilter3D kalmanFilterForward(this->InitialiseKalmanFilter(false));
                const bool addedHitsForward{this->KalmanFilterTreatment(postBackwardCaloHitSet, allGoodProtoHits, protoHitVector, kalmanFilterForward)};
//                std::cout << "Backward pass added " << allGoodProtoHits.size() - postBackwardCaloHitSet.size() << " hits" << std::endl;    

                if (!addedHits)
                    addedHits = addedHitsForward || addedHitsBackward;
            }

            fractionUsed = allGoodProtoHits.size() / static_cast<float>(nPfoHits);
//            std::cout << "Used " << 100.f * fractionUsed << "% of hits in iteration " << i << std::endl;
            if (!addedHits)
                keepIterating = false;

            if (!keepIterating)
                break;

        }

        std::cout << "Final: Percentage of hits made in 3D = " << 100.f * fractionUsed << std::endl;
        if (allGoodProtoHits.empty())
            continue;

        if ((m_iterateTrackHits && LArPfoHelper::IsTrack(pPfo)) || (m_iterateShowerHits && LArPfoHelper::IsShower(pPfo)))
            this->IterativeTreatment(allGoodProtoHits);

        CaloHitList newThreeDTrackHits;
        this->CreateThreeDHits(allGoodProtoHits, newThreeDTrackHits);
        this->AddThreeDHitsToPfo(pPfo, newThreeDTrackHits);

        allNewThreeDTrackHits.insert(allNewThreeDTrackHits.end(), newThreeDTrackHits.begin(), newThreeDTrackHits.end());
    }

    if (!allNewThreeDTrackHits.empty())
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList(*this, allNewThreeDTrackHits, m_outputCaloHitListName));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

CartesianVector KalmanThreeDTrackHitCreationAlgorithm::Calculate3DPosition(ViewVectorMap &viewVectorMap) const
{
    // Look for consistency between different combinations of vertex directions in 3D
    CartesianVector vtxPosUV{0.f, 0.f, 0.f}, vtxPosUW{0.f, 0.f, 0.f}, vtxPosVW{0.f, 0.f, 0.f};
    const bool haveU(viewVectorMap.count(TPC_VIEW_U));
    const bool haveV(viewVectorMap.count(TPC_VIEW_V));
    const bool haveW(viewVectorMap.count(TPC_VIEW_W));

    if (haveU && haveV)
    {
        const float u{viewVectorMap.at(TPC_VIEW_U).GetZ()};
        const float v{viewVectorMap.at(TPC_VIEW_V).GetZ()};
        const float x{((viewVectorMap.at(TPC_VIEW_U) + viewVectorMap.at(TPC_VIEW_V)) * 0.5).GetX()};
        const float y(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->UVtoY(u, v));
        const float z(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->UVtoZ(u, v));
        vtxPosUV.SetValues(x, y, z);
    }
    if (haveU && haveW)  
    {
        const float u{viewVectorMap.at(TPC_VIEW_U).GetZ()};
        const float w{viewVectorMap.at(TPC_VIEW_W).GetZ()};
        const float x{((viewVectorMap.at(TPC_VIEW_U) + viewVectorMap.at(TPC_VIEW_W)) * 0.5).GetX()};
        const float y(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->UWtoY(u, w));
        const float z(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->UWtoZ(u, w));
        vtxPosUW.SetValues(x, y, z);
    }
    if (haveV && haveW) 
    {
        const float v{viewVectorMap.at(TPC_VIEW_V).GetZ()};
        const float w{viewVectorMap.at(TPC_VIEW_W).GetZ()};
        const float x{((viewVectorMap.at(TPC_VIEW_V) + viewVectorMap.at(TPC_VIEW_W)) * 0.5).GetX()};
        const float y(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->VWtoY(v, w));
        const float z(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->VWtoZ(v, w));
        vtxPosVW.SetValues(x, y, z);
    }
    
    CartesianVector seedPosition(0.f, 0.f, 0.f);
    if (haveU && haveV && haveW)
    {
//        std::cout << "Three vertex position choices:" << std::endl;
//        std::cout << vtxPosUV << std::endl;
//        std::cout << vtxPosUW << std::endl;
//        std::cout << vtxPosVW << std::endl;
        seedPosition = (vtxPosUV + vtxPosUW + vtxPosVW) * (1.f / 3.f);
    }
    else if (haveU && haveV)
        seedPosition = vtxPosUV;
    else if (haveU && haveW)
        seedPosition = vtxPosUW;
    else if (haveV && haveW)
        seedPosition = vtxPosVW;
    else
        throw STATUS_CODE_NOT_FOUND; // Should be impossible

    return seedPosition;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void KalmanThreeDTrackHitCreationAlgorithm::GetPfoEndPoints(const ParticleFlowObject *const pPfo)
{
    ViewVectorMap viewFirstHitPositions;
    ViewVectorMap viewFinalHitPositions;

    this->GetPfoEndPoints2D(pPfo, viewFirstHitPositions, viewFinalHitPositions, TPC_VIEW_U);
    this->GetPfoEndPoints2D(pPfo, viewFirstHitPositions, viewFinalHitPositions, TPC_VIEW_V);
    this->GetPfoEndPoints2D(pPfo, viewFirstHitPositions, viewFinalHitPositions, TPC_VIEW_W);
    
    m_pfoVertex = this->Calculate3DPosition(viewFirstHitPositions);
    m_pfoEnd = this->Calculate3DPosition(viewFinalHitPositions);
//    std::cout << m_nuVtxPosition << ", " << m_pfoVertex << ", " << m_pfoEnd << std::endl;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void KalmanThreeDTrackHitCreationAlgorithm::GetPfoEndPoints2D(const pandora::ParticleFlowObject *const pPfo, ViewVectorMap &vertices, ViewVectorMap &ends, const pandora::HitType view)
{
    CaloHitList caloHitList;
    LArPfoHelper::GetCaloHits(pPfo, view, caloHitList);

    if (caloHitList.size() < 2)
        return;

    const CartesianVector start = (*(caloHitList.begin()))->GetPositionVector();
    const CartesianVector end = (*std::prev(caloHitList.end()))->GetPositionVector();
    
    bool swapEnds{false};
    if (m_hasNeutrinoVertex)
    {
        const CartesianVector projVtx(LArGeometryHelper::ProjectPosition(this->GetPandora(), m_nuVtxPosition, view));
        const float startDiffSq{(projVtx-start).GetMagnitudeSquared()};
        const float endDiffSq{(projVtx-end).GetMagnitudeSquared()};
        swapEnds = startDiffSq > endDiffSq ? true : false;
//        std::cout << "Checking ends for view " << view << ": " << startDiffSq << ", " << endDiffSq << ", " << swapEnds << std::endl;
    }
    
    vertices.emplace(view, !swapEnds ? start : end);
    ends.emplace(view, !swapEnds ? end : start);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void KalmanThreeDTrackHitCreationAlgorithm::GetKalmanSeedPositionAndDirection(CartesianVector &seedPosition, CartesianVector &seedDirection, const bool backwardMode)
{
    seedPosition = backwardMode ? m_pfoEnd : m_pfoVertex;
    const CartesianVector endPosition = backwardMode ? m_pfoVertex : m_pfoEnd;
    seedDirection = (endPosition - seedPosition).GetUnitVector();
}

//------------------------------------------------------------------------------------------------------------------------------------------

void KalmanThreeDTrackHitCreationAlgorithm::CalculateDynamicStepSize(const unsigned int nHits)
{
    if (!m_useDynamicStepSize || nHits < 1)
        return;

    m_kalmanStepSize = (m_pfoEnd - m_pfoVertex).GetMagnitude() / static_cast<float>(nHits);
}

//------------------------------------------------------------------------------------------------------------------------------------------

DistanceBasedKalmanFilter3D KalmanThreeDTrackHitCreationAlgorithm::InitialiseKalmanFilter(const bool backwardMode)
{
    CartesianVector seedPosition{0.f, 0.f, 0.f}, seedDirection{0.f, 0.f, 0.f};
    this->GetKalmanSeedPositionAndDirection(seedPosition, seedDirection, backwardMode);

    // Convert to Eigen for the Kalman Filter
    Eigen::VectorXd x(3);
    x << seedPosition.GetX(), seedPosition.GetY(), seedPosition.GetZ();
    Eigen::VectorXd v(3);
    v << seedDirection.GetX(), seedDirection.GetY(), seedDirection.GetZ();

    Eigen::VectorXd x_err(3);
    x_err << 1.0, 1.0, 1.0;
    Eigen::VectorXd v_err(3);
    v_err << 1.0, 1.0, 1.0;
    DistanceBasedKalmanFilter3D distanceKalmanFilter(x, v, x_err, v_err);
    return distanceKalmanFilter;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool KalmanThreeDTrackHitCreationAlgorithm::KalmanFilterTreatment(const std::set<const CaloHit*> &preToolCaloHitSet, ProtoHitVector &protoHitVector, const ProtoHitVector &potentialProtoHits, DistanceBasedKalmanFilter3D &kalmanFilter) const
{
    std::set<const CaloHit*> usedHits;
    ProtoHitVector goodHitsToAdd;
    ProtoHitVector allHitsToConsider{protoHitVector};
    allHitsToConsider.insert(allHitsToConsider.end(), potentialProtoHits.begin(), potentialProtoHits.end());

    // Sort the hits by distance from the seed position
    const CartesianVector referencePos{kalmanFilter.GetPositionAsCartesianVector()};
    std::sort(allHitsToConsider.begin(), allHitsToConsider.end(), [&referencePos](const ProtoHit &a, const ProtoHit &b)
          {return (a.GetPosition3D() - referencePos).GetMagnitudeSquared() < (b.GetPosition3D() - referencePos).GetMagnitudeSquared();});

    unsigned int nPredictionsWithoutMatch{0};
    while (nPredictionsWithoutMatch < m_stepsWithoutMatch)
    {
        const float step{m_kalmanStepSize * (1 + nPredictionsWithoutMatch)};
        kalmanFilter.Predict(step);
        const CartesianVector prediction{kalmanFilter.GetPredictedPositionAsCartesianVector()};

        std::map<float, ProtoHit> potentialMatches;
        for (const ProtoHit &hit : allHitsToConsider)
        {
            if (usedHits.count(hit.GetParentCaloHit2D()))
                continue;

            const CartesianVector measurement{hit.GetPosition3D()};
            Eigen::VectorXd m(3);
            m << measurement.GetX(), measurement.GetY(), measurement.GetZ();
            const float distance{static_cast<float>(kalmanFilter.GetMahalanobisDistance(m, true))};
            if (distance < m_tolerance)
                potentialMatches.insert(std::make_pair(distance, hit));
        }

       if (potentialMatches.empty())
        {
            ++nPredictionsWithoutMatch;
            continue;
        }

        // Choose the closest hit
        const ProtoHit &bestHit = potentialMatches.begin()->second;
        Eigen::VectorXd m(3);
        Eigen::VectorXd mError(3);
        const CartesianVector measurement{bestHit.GetPosition3D()};
        m << measurement.GetX(), measurement.GetY(), measurement.GetZ();
        mError << 1.0, 1.0, 1.0;
        kalmanFilter.Update(m, mError);
        if (!preToolCaloHitSet.count(bestHit.GetParentCaloHit2D()))
            goodHitsToAdd.emplace_back(bestHit);
        usedHits.insert(bestHit.GetParentCaloHit2D());
        nPredictionsWithoutMatch = 0;
    }

    protoHitVector.insert(protoHitVector.end(), goodHitsToAdd.begin(), goodHitsToAdd.end());
    return !goodHitsToAdd.empty();
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode KalmanThreeDTrackHitCreationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "Tolerance", m_tolerance));

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "KalmanStepSize", m_kalmanStepSize));

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "StepsWithoutMatch", m_stepsWithoutMatch));

    return ThreeDHitCreationAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_content
