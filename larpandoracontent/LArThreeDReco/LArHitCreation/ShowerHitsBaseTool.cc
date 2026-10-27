/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/ShowerHitsBaseTool.cc
 *
 *  @brief  Implementation of the shower hit creation tool.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"
#include "Geometry/LArReadoutVolume.h"
#include "Geometry/LArReadoutChannel.h"

#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/ShowerHitsBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

using namespace pandora;

namespace lar_content
{

ShowerHitsBaseTool::ShowerHitsBaseTool() :
    m_xTolerance(1.f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerHitsBaseTool::Run(ThreeDHitCreationAlgorithm *const pAlgorithm, const ParticleFlowObject *const pPfo,
    const CaloHitVector &inputTwoDHits, ProtoHitVector &protoHitVector)
{
    if (PandoraContentApi::GetSettings(*pAlgorithm)->ShouldDisplayAlgorithmInfo())
        std::cout << "----> Running Algorithm Tool: " << this->GetInstanceName() << ", " << this->GetType() << std::endl;

    m_useGeometryConstraints = pAlgorithm->GetUseDetailedGeometry();

    try
    {
        if (!LArPfoHelper::IsShower(pPfo))
            return;

        CaloHitVector caloHitVectorU, caloHitVectorV, caloHitVectorW;
        pAlgorithm->FilterCaloHitsByType(inputTwoDHits, TPC_VIEW_U, caloHitVectorU);
        pAlgorithm->FilterCaloHitsByType(inputTwoDHits, TPC_VIEW_V, caloHitVectorV);
        pAlgorithm->FilterCaloHitsByType(inputTwoDHits, TPC_VIEW_W, caloHitVectorW);

        this->GetShowerHits3D(caloHitVectorU, caloHitVectorV, caloHitVectorW, protoHitVector);
        this->GetShowerHits3D(caloHitVectorV, caloHitVectorU, caloHitVectorW, protoHitVector);
        this->GetShowerHits3D(caloHitVectorW, caloHitVectorU, caloHitVectorV, protoHitVector);
    }
    catch (StatusCodeException &)
    {
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerHitsBaseTool::GetShowerHits3D(const CaloHitVector &inputTwoDHits, const CaloHitVector &caloHitVector1,
    const CaloHitVector &caloHitVector2, ProtoHitVector &protoHitVector) const
{
    for (const CaloHit *const pCaloHit2D : inputTwoDHits)
    {
        try
        {
            CaloHitVector filteredHits1, filteredHits2;
            this->FilterCaloHits(pCaloHit2D, caloHitVector1, filteredHits1);
            this->FilterCaloHits(pCaloHit2D, caloHitVector2, filteredHits2);

            ProtoHit protoHit(pCaloHit2D);
            this->GetShowerHit3D(filteredHits1, filteredHits2, protoHit);

            if (protoHit.IsPositionSet() && (protoHit.GetChi2() < m_chiSquaredCut))
                protoHitVector.push_back(protoHit);
        }
        catch (StatusCodeException &)
        {
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ShowerHitsBaseTool::FilterCaloHits(
    const CaloHit *const pSeedHit, const CaloHitVector &inputCaloHitVector, CaloHitVector &outputCaloHitVector) const
{
    for (const CaloHit *const pCaloHit : inputCaloHitVector)
    {
        const float x{pSeedHit->GetPositionVector().GetX()};

        if (m_useGeometryConstraints)
        {
            const LArCaloHit *pLArSeedHit{dynamic_cast<const LArCaloHit *>(pSeedHit)};
            const LArCaloHit *pLArCaloHit{dynamic_cast<const LArCaloHit *>(pCaloHit)};

            if (pLArSeedHit == nullptr || pLArCaloHit == nullptr)
                throw StatusCodeException(STATUS_CODE_FAILURE);

            // Check the readout volume
            const LArReadoutVolume &seedHitVolume{LArGeometryHelper::GetReadoutVolume(this->GetPandora(), *pLArSeedHit)};
            const LArReadoutVolume &caloHitVolume{LArGeometryHelper::GetReadoutVolume(this->GetPandora(), *pLArCaloHit)};
            if (seedHitVolume.GetId() != caloHitVolume.GetId())
                continue;           

            // Check the readout channel of the calo hit falls within the range of geometric matches to the seed channel
            const LArReadoutChannel &seedHitChannel{LArGeometryHelper::GetReadoutChannel(this->GetPandora(), *pLArSeedHit)};
            const LArReadoutChannel::ChannelInterval &channelInterval{seedHitChannel.GetChannelInterval(pLArCaloHit->GetHitType())};
            const unsigned int caloHitChannelId{LArGeometryHelper::GetReadoutChannel(this->GetPandora(), *pLArCaloHit).GetId()};
            if (caloHitChannelId < channelInterval.first || caloHitChannelId > channelInterval.second)
                continue;
        }

        const float deltaX{pCaloHit->GetPositionVector().GetX() - x};

        if (std::fabs(deltaX) < m_xTolerance)
            outputCaloHitVector.push_back(pCaloHit);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ShowerHitsBaseTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "XTolerance", m_xTolerance));

    return HitCreationBaseTool::ReadSettings(xmlHandle);
}

} // namespace lar_content
