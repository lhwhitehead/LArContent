/**
 *  @file   larpandoracontent/LArTrackShowerId/DlPfoCharacterisationAlgorithm.cc
 *
 *  @brief  Implementation of the cut based pfo characterisation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

#include "larpandoradlcontent/LArTrackShowerId/DlPfoCharacterisationAlgorithm.h"

#include <numeric>

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DlPfoCharacterisationAlgorithm::DlPfoCharacterisationAlgorithm() :
    m_trackLikelihoodThreshold(0.5f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool DlPfoCharacterisationAlgorithm::IsClearTrack(const Cluster *const pCluster) const
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

bool DlPfoCharacterisationAlgorithm::IsClearTrack(const pandora::ParticleFlowObject *const pPfo) const
{
    ClusterList allClusters;
    LArPfoHelper::GetTwoDClusterList(pPfo, allClusters);
    FloatVector clusterMeanTrackLikelihoods;
    for (const Cluster *pCluster : allClusters)
    {
        const OrderedCaloHitList &orderedCaloHitList{pCluster->GetOrderedCaloHitList()};
        CaloHitList caloHits;
        orderedCaloHitList.FillCaloHitList(caloHits);
        const CaloHitList &isolatedHits{pCluster->GetIsolatedCaloHitList()};
        caloHits.insert(caloHits.end(), isolatedHits.begin(), isolatedHits.end());

        const float meanTrackLikelihood = LArDLHelper::GetMeanTrackLikelihood(caloHits);
        clusterMeanTrackLikelihoods.emplace_back(meanTrackLikelihood);
    }

    const unsigned long N{clusterMeanTrackLikelihoods.size()};
    if (N > 0)
    {
        const float mean{std::accumulate(std::begin(clusterMeanTrackLikelihoods), std::end(clusterMeanTrackLikelihoods), 0.f) / N};
        if (mean >= m_trackLikelihoodThreshold)
            return true;
        else
            return false;
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DlPfoCharacterisationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "TrackLikelihoodThreshold", m_trackLikelihoodThreshold));
    return PfoCharacterisationBaseAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_dl_content
