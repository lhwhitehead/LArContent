/**
 *  @file   larpandoradlcontent/LArTwoDReco/DlTrackShowerStreamSelectionAlgorithm.cc
 *
 *  @brief  Implementation of the deep learning track shower cluster streaming algorithm.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoradlcontent/LArTwoDReco/DlTrackShowerStreamSelectionAlgorithm.h"

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

#include <numeric>

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlTrackShowerStreamSelectionAlgorithm::DlTrackShowerStreamSelectionAlgorithm() :
  m_trackLikelihoodThreshold(0.5f)
{
}

StatusCode DlTrackShowerStreamSelectionAlgorithm::AllocateToStreams(const Cluster *const pCluster)
{
    const OrderedCaloHitList &orderedCaloHitList{pCluster->GetOrderedCaloHitList()};
    CaloHitList caloHits;
    orderedCaloHitList.FillCaloHitList(caloHits);
    const CaloHitList &isolatedHits{pCluster->GetIsolatedCaloHitList()};
    caloHits.insert(caloHits.end(), isolatedHits.begin(), isolatedHits.end());

    const float meanTrackLikelihood = LArDLHelper::GetMeanTrackLikelihood(caloHits);
    if (meanTrackLikelihood >= m_trackLikelihoodThreshold)
        m_clusterListMap.at(m_trackListName).emplace_back(pCluster);
    else
        m_clusterListMap.at(m_showerListName).emplace_back(pCluster);

    return STATUS_CODE_SUCCESS;
}
//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlTrackShowerStreamSelectionAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, StreamSelectionAlgorithm::ReadSettings(xmlHandle));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "TrackListName", m_trackListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "ShowerListName", m_showerListName));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "TrackLikelihoodThreshold", m_trackLikelihoodThreshold));
    m_listNames.emplace_back(m_trackListName);
    m_listNames.emplace_back(m_showerListName);

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_dl_content
