/**
 *  @file   larpandoracontent/LArTrackShowerId/DlClusterCharacterisationAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning based cluster characterisation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoradlcontent/LArTrackShowerId/DlClusterCharacterisationAlgorithm.h"

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

#include <numeric>

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlClusterCharacterisationAlgorithm::DlClusterCharacterisationAlgorithm() :
  m_trackLikelihoodThreshold(0.5f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

DlClusterCharacterisationAlgorithm::~DlClusterCharacterisationAlgorithm()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool DlClusterCharacterisationAlgorithm::IsClearTrack(const Cluster *const pCluster) const
{
    const OrderedCaloHitList &orderedCaloHitList{pCluster->GetOrderedCaloHitList()};
    CaloHitList caloHits;
    orderedCaloHitList.FillCaloHitList(caloHits);
    const CaloHitList &isolatedHits{pCluster->GetIsolatedCaloHitList()};
    caloHits.insert(caloHits.end(), isolatedHits.begin(), isolatedHits.end());

    const float meanTrackLikelihood = LArDLHelper::GetMeanTrackLikelihood(caloHits);
    if (meanTrackLikelihood >= m_trackLikelihoodThreshold)
        return true;
    else
        return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlClusterCharacterisationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "TrackLikelihoodThreshold", m_trackLikelihoodThreshold)); 
    return ClusterCharacterisationBaseAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_dl_content
