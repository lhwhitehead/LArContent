/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/PlaneSolverHitsTool.cc
 *
 *  @brief  Implementation of the plane solver-based hit creation tool.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArObjects/LArPlaneContextObject.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/PlaneSolverHitsTool.h"

using namespace pandora;

namespace lar_content
{

PlaneSolverHitsTool::PlaneSolverHitsTool() :
    m_tripletsOnly{false},
    m_collectionOnly{false},
    m_applyChi2Cut{false},
    m_chi2Cut{std::numeric_limits<float>::max()}
{

}

void PlaneSolverHitsTool::Run(ThreeDHitCreationAlgorithm *const pAlgorithm, const pandora::ParticleFlowObject *const pPfo,
        const pandora::CaloHitVector &inputTwoDHits, ProtoHitVector &protoHitVector)
{

    if (PandoraContentApi::GetSettings(*pAlgorithm)->ShouldDisplayAlgorithmInfo())
        std::cout << "----> Running Algorithm Tool: " << this->GetInstanceName() << ", " << this->GetType() << std::endl;

    const LArPlaneContextObject *pPlaneContextObject{dynamic_cast<const LArPlaneContextObject*>(PandoraContentApi::GetEventContextObject(
        *pAlgorithm, "PlaneContext"))};

    if (!pPlaneContextObject)
        return;

    for (auto const *pCaloHit2D : inputTwoDHits)
    {
        const LArPlaneContextObject::HitTriplet *pTriplet{pPlaneContextObject->GetHitTriplet(pCaloHit2D)};
        if (!pTriplet)
            continue;

        if (m_applyChi2Cut && (pTriplet->m_chi2 > m_chi2Cut))
            continue;

        const HitType hitType{pCaloHit2D->GetHitType()};
        if (m_collectionOnly && hitType != TPC_VIEW_W)
            continue;

        std::unordered_map<HitType, const CaloHit*> hitTypeToCaloHitMap;
        hitTypeToCaloHitMap[TPC_VIEW_U] = pTriplet->m_uHit;
        hitTypeToCaloHitMap[TPC_VIEW_V] = pTriplet->m_vHit;
        hitTypeToCaloHitMap[TPC_VIEW_W] = pTriplet->m_wHit;

        const HitType hitType1((TPC_VIEW_U == hitType) ? TPC_VIEW_V : (TPC_VIEW_V == hitType) ? TPC_VIEW_W : TPC_VIEW_U);
        const HitType hitType2((TPC_VIEW_U == hitType) ? TPC_VIEW_W : (TPC_VIEW_V == hitType) ? TPC_VIEW_U : TPC_VIEW_V);

        bool canUseHit1{false};
        if (hitTypeToCaloHitMap[hitType1] != nullptr)
            canUseHit1 = std::find(inputTwoDHits.begin(), inputTwoDHits.end(), hitTypeToCaloHitMap[hitType1]) != inputTwoDHits.end();

        bool canUseHit2{false};
        if (hitTypeToCaloHitMap[hitType2] != nullptr)
            canUseHit2 = std::find(inputTwoDHits.begin(), inputTwoDHits.end(), hitTypeToCaloHitMap[hitType2]) != inputTwoDHits.end();

        // If we don't have any hits then move on
        if (!canUseHit1 && !canUseHit2)
            continue;

        const bool isTriplet{canUseHit1 && canUseHit2};
        if (m_tripletsOnly && !isTriplet)
            continue;

        CartesianPointVector fitPositions1, fitPositions2;
        if (canUseHit1)
            fitPositions1.emplace_back(hitTypeToCaloHitMap[hitType1]->GetPositionVector());
        if (canUseHit2) 
            fitPositions2.emplace_back(hitTypeToCaloHitMap[hitType2]->GetPositionVector());

        ProtoHit protoHit(pCaloHit2D);
        this->GetBestPosition3D(hitType1, hitType2, fitPositions1, fitPositions2, protoHit);

        if (protoHit.IsPositionSet() && (protoHit.GetChi2() < m_chiSquaredCut))
            protoHitVector.emplace_back(protoHit);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode PlaneSolverHitsTool::ReadSettings(const pandora::TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "TripletsOnly", m_tripletsOnly));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "CollectionOnly", m_collectionOnly));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "ApplyChi2Cut", m_applyChi2Cut));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "Chi2Cut", m_chi2Cut));

    return HitCreationBaseTool::ReadSettings(xmlHandle);
}

} // namespace lar_content
