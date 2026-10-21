/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/LongitudinalTrackHitsBaseTool.cc
 *
 *  @brief  Implementation of the longitudinal track hits base tool.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/LongitudinalTrackHitsBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

using namespace pandora;

namespace lar_content
{

LongitudinalTrackHitsBaseTool::LongitudinalTrackHitsBaseTool() :
    m_vtxDisplacementCutSquared(5.f * 5.f),
    m_minTrackLengthSquared(7.5f * 7.5f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LongitudinalTrackHitsBaseTool::GetTrackHits3D(
    const CaloHitVector &inputTwoDHits, const MatchedSlidingFitMap &inputSlidingFitMap, ProtoHitVector &protoHitVector) const
{
    MatchedSlidingFitMap matchedSlidingFitMap;
    CartesianVector vtx3D(0.f, 0.f, 0.f), end3D(0.f, 0.f, 0.f);
    this->GetVertexAndEndPositions(inputSlidingFitMap, matchedSlidingFitMap, vtx3D, end3D);

    for (const CaloHit *const pCaloHit2D : inputTwoDHits)
    {
        try
        {
            ProtoHit protoHit(pCaloHit2D);
            this->GetLongitudinalTrackHit3D(matchedSlidingFitMap, vtx3D, end3D, protoHit);

            if (protoHit.IsPositionSet() && (protoHit.GetChi2() < m_chiSquaredCut))
                protoHitVector.push_back(protoHit);
        }
        catch (StatusCodeException &)
        {
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LongitudinalTrackHitsBaseTool::GetVertexAndEndPositions(const MatchedSlidingFitMap &inputSlidingFitMap,
    MatchedSlidingFitMap &outputSlidingFitMap, CartesianVector &bestVtx3D, CartesianVector &bestEnd3D) const
{
    const std::array<HitType, 3> hitTypes{TPC_VIEW_U, TPC_VIEW_V, TPC_VIEW_W};
    const std::array<MatchedSlidingFitMap::const_iterator, 3> iterators{inputSlidingFitMap.find(TPC_VIEW_U), inputSlidingFitMap.find(TPC_VIEW_V), inputSlidingFitMap.find(TPC_VIEW_W)};
    const std::array<bool, 3> found{iterators[0] != inputSlidingFitMap.end(), iterators[1] != inputSlidingFitMap.end(), iterators[2] != inputSlidingFitMap.end()};

    std::array<bool, 3> useView{false, false, false};
    float bestChi2(std::numeric_limits<float>::max());

    // Iterate over the forward / backward possibilities
    for (unsigned int iPermutation = 0; iPermutation < 4; ++iPermutation)
    {
        const std::array<bool, 3> isForward{iPermutation != 1, iPermutation != 2, iPermutation != 3};

        std::array<CartesianVector, 3> vtx{CartesianVector(0.f, 0.f, 0.f), CartesianVector(0.f, 0.f, 0.f), CartesianVector(0.f, 0.f, 0.f)};
        std::array<CartesianVector, 3> end{CartesianVector(0.f, 0.f, 0.f), CartesianVector(0.f, 0.f, 0.f), CartesianVector(0.f, 0.f, 0.f)};

        // Iterate over the three views to get the start and end points
        for (unsigned int v = 0; v < 3; ++v)
        {
            if (!found[v])
                continue;

            const TwoDSlidingFitResult &slidingFitResult = iterators[v]->second;

            vtx[v] = isForward[v] ? slidingFitResult.GetGlobalMinLayerPosition() : slidingFitResult.GetGlobalMaxLayerPosition();
            end[v] = isForward[v] ? slidingFitResult.GetGlobalMaxLayerPosition() : slidingFitResult.GetGlobalMinLayerPosition();
        }

        // Iterate over pairs of views (v1, v2) to get the best vertex and end points
        CartesianVector vtx3D(0.f, 0.f, 0.f), end3D(0.f, 0.f, 0.f);
        float vtxChi2(std::numeric_limits<float>::max()), endChi2(std::numeric_limits<float>::max());
        for (unsigned int v1 = 0; v1 < 3; ++v1)
        {
            const unsigned int v2{(v1 + 1) % 3};
        
            if (!found[v1] || !found[v2])
                continue;
        
            this->UpdateBestPosition(hitTypes[v1], hitTypes[v2], vtx[v1], vtx[v2], vtx3D, vtxChi2);
            this->UpdateBestPosition(hitTypes[v1], hitTypes[v2], end[v1], end[v2], end3D, endChi2);
        }

        std::array<bool, 3> matched{false, false, false};
        unsigned int matchedViews{0};
        
        for (unsigned int v = 0; v < 3; ++v)
        {
            if (!found[v])
                continue;
        
            const CartesianVector projVtx{LArGeometryHelper::ProjectPosition(this->GetPandora(), vtx3D, hitTypes[v])};
            const CartesianVector projEnd{LArGeometryHelper::ProjectPosition(this->GetPandora(), end3D, hitTypes[v])};
        
            if ((end[v] - vtx[v]).GetMagnitudeSquared() > m_minTrackLengthSquared &&
                (projVtx - vtx[v]).GetMagnitudeSquared() < std::min(m_vtxDisplacementCutSquared, (projVtx - end[v]).GetMagnitudeSquared()) &&
                (projEnd - end[v]).GetMagnitudeSquared() < std::min(m_vtxDisplacementCutSquared, (projEnd - vtx[v]).GetMagnitudeSquared()))
            {
                matched[v] = true;
                ++matchedViews;
            }
        }

        if (matchedViews < 2)
            continue;
        
        if (vtxChi2 + endChi2 < bestChi2)
        {
            bestVtx3D = vtx3D;
            bestEnd3D = end3D;
            bestChi2 = vtxChi2 + endChi2;
            useView = matched;
        }
    }

    // Store the outputs        
    for (unsigned int v = 0; v < 3; ++v)
        if (useView[v])
            outputSlidingFitMap.insert(MatchedSlidingFitMap::value_type(iterators[v]->first, iterators[v]->second));

    if (outputSlidingFitMap.empty())
        throw StatusCodeException(STATUS_CODE_NOT_FOUND);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LongitudinalTrackHitsBaseTool::UpdateBestPosition(const HitType hitType1, const HitType hitType2, const CartesianVector &vtx1,
    const CartesianVector &vtx2, CartesianVector &bestVtx, float &bestChi2) const
{
    CartesianVector mergedVtx(0.f, 0.f, 0.f);
    float mergedChi2(std::numeric_limits<float>::max());

    LArGeometryHelper::MergeTwoPositions3D(this->GetPandora(), hitType1, hitType2, vtx1, vtx2, mergedVtx, mergedChi2);

    if (mergedChi2 < bestChi2)
    {
        bestVtx = mergedVtx;
        bestChi2 = mergedChi2;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode LongitudinalTrackHitsBaseTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    float vtxDisplacementCut = std::sqrt(m_vtxDisplacementCutSquared);
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "VertexDisplacementCut", vtxDisplacementCut));
    m_vtxDisplacementCutSquared = vtxDisplacementCut * vtxDisplacementCut;

    float minTrackLength = std::sqrt(m_minTrackLengthSquared);
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "MinTrackLength", minTrackLength));
    m_minTrackLengthSquared = minTrackLength * minTrackLength;

    return TrackHitsBaseTool::ReadSettings(xmlHandle);
}

} // namespace lar_content
