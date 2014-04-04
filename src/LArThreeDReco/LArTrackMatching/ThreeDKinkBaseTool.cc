/**
 *  @file   LArContent/src/LArThreeDReco/LArTrackMatching/ThreeDKinkBaseTool.cc
 * 
 *  @brief  Implementation of the three d kink base tool class.
 * 
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "LArHelpers/LArThreeDHelper.h"

#include "LArObjects/LArPointingCluster.h"

#include "LArThreeDReco/LArTrackMatching/ThreeDKinkBaseTool.h"

using namespace pandora;

namespace lar
{

ThreeDKinkBaseTool::ThreeDKinkBaseTool(const unsigned int nCommonClusters) :
    m_nCommonClusters(nCommonClusters)
{
    if (!((1 == m_nCommonClusters) || (2 == m_nCommonClusters)))
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDKinkBaseTool::PassesElementCuts(TensorType::ElementList::const_iterator eIter, const ClusterList &usedClusters) const
{
    if (usedClusters.count(eIter->GetClusterU()) || usedClusters.count(eIter->GetClusterV()) || usedClusters.count(eIter->GetClusterW()))
        return false;

    if (eIter->GetOverlapResult().GetMatchedFraction() < m_minMatchedFraction)
        return false;

    if (eIter->GetOverlapResult().GetNMatchedSamplingPoints() < m_minMatchedSamplingPoints)
        return false;

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDKinkBaseTool::Run(ThreeDTransverseTracksAlgorithm *pAlgorithm, TensorType &overlapTensor)
{
    if (PandoraSettings::ShouldDisplayAlgorithmInfo())
       std::cout << "----> Running Algorithm Tool: " << this << ", " << m_algorithmToolType << std::endl;

    ModificationList modificationList;
    this->GetModifications(pAlgorithm, overlapTensor, modificationList);
    const bool changesMade(this->ApplyChanges(pAlgorithm, modificationList));

    return changesMade;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDKinkBaseTool::GetModifications(ThreeDTransverseTracksAlgorithm *pAlgorithm, const TensorType &overlapTensor, ModificationList &modificationList) const
{
    ClusterList usedClusters;

    for (TensorType::const_iterator iterU = overlapTensor.begin(), iterUEnd = overlapTensor.end(); iterU != iterUEnd; ++iterU)
    {
        if (!iterU->first->IsAvailable())
            continue;

        unsigned int nU(0), nV(0), nW(0);
        TensorType::ElementList elementList;
        overlapTensor.GetConnectedElements(iterU->first, true, elementList, nU, nV, nW);

        if (nU * nV * nW < 2)
            continue;

        std::sort(elementList.begin(), elementList.end(), ThreeDTransverseTracksAlgorithm::SortByNMatchedSamplingPoints);

        for (TensorType::ElementList::const_iterator eIter = elementList.begin(); eIter != elementList.end(); ++eIter)
        {
            if (!this->PassesElementCuts(eIter, usedClusters))
                continue;

            IteratorList iteratorList;
            this->SelectTensorElements(eIter, elementList, usedClusters, iteratorList);

            if (iteratorList.size() < 2)
                continue;

            ModificationList localModificationList;
            this->GetIteratorListModifications(pAlgorithm, iteratorList, localModificationList);

            if (localModificationList.empty())
                continue;

            for (ModificationList::const_iterator mIter = localModificationList.begin(), mIterEnd = localModificationList.end(); mIter != mIterEnd; ++mIter)
            {
                usedClusters.insert(mIter->m_affectedClusters.begin(), mIter->m_affectedClusters.end());
            }

            modificationList.insert(modificationList.end(), localModificationList.begin(), localModificationList.end());
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDKinkBaseTool::ApplyChanges(ThreeDTransverseTracksAlgorithm *pAlgorithm, const ModificationList &modificationList) const
{
    ClusterMergeMap consolidatedMergeMap;
    SplitPositionMap consolidatedSplitMap;

    for (ModificationList::const_iterator iter = modificationList.begin(), iterEnd = modificationList.end(); iter != iterEnd; ++iter)
    {
        for (ClusterMergeMap::const_iterator cIter = iter->m_clusterMergeMap.begin(), cIterEnd = iter->m_clusterMergeMap.end(); cIter != cIterEnd; ++cIter)
        {
            const ClusterList &daughterClusters(cIter->second);

            for (ClusterList::const_iterator dIter = daughterClusters.begin(), dIterEnd = daughterClusters.end(); dIter != dIterEnd; ++dIter)
            {
                if (consolidatedMergeMap.count(*dIter))
                    throw StatusCodeException(STATUS_CODE_FAILURE);
            }

            ClusterList &targetClusterList(consolidatedMergeMap[cIter->first]);
            targetClusterList.insert(daughterClusters.begin(), daughterClusters.end());
        }

        for (SplitPositionMap::const_iterator cIter = iter->m_splitPositionMap.begin(), cIterEnd = iter->m_splitPositionMap.end(); cIter != cIterEnd; ++cIter)
        {
            CartesianPointList &cartesianPointList(consolidatedSplitMap[cIter->first]);
            cartesianPointList.insert(cartesianPointList.end(), cIter->second.begin(), cIter->second.end());
        }
    }

    bool changesMade(false);
    changesMade |= this->MakeClusterMerges(pAlgorithm, consolidatedMergeMap);
    changesMade |= this->MakeClusterSplits(pAlgorithm, consolidatedSplitMap);

    return changesMade;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDKinkBaseTool::SelectTensorElements(TensorType::ElementList::const_iterator eIter, const TensorType::ElementList &elementList,
    const ClusterList &usedClusters, IteratorList &iteratorList) const
{
    iteratorList.push_back(eIter);

    for (TensorType::ElementList::const_iterator eIter2 = elementList.begin(); eIter2 != elementList.end(); ++eIter2)
    {
        if (eIter == eIter2)
            continue;

        if (!this->PassesElementCuts(eIter2, usedClusters))
            continue;

        for (IteratorList::const_iterator iIter = iteratorList.begin(); iIter != iteratorList.end(); ++iIter)
        {
            if ((*iIter) == eIter2)
                continue;

            unsigned int nMatchedClusters(0);

            if ((*iIter)->GetClusterU() == eIter2->GetClusterU())
                ++nMatchedClusters;

            if ((*iIter)->GetClusterV() == eIter2->GetClusterV())
                ++nMatchedClusters;

            if ((*iIter)->GetClusterW() == eIter2->GetClusterW())
                ++nMatchedClusters;

            if (m_nCommonClusters == nMatchedClusters)
            {
                iteratorList.push_back(eIter2);
                return;
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDKinkBaseTool::MakeClusterMerges(ThreeDTransverseTracksAlgorithm *pAlgorithm, const ClusterMergeMap &clusterMergeMap) const
{
    ClusterList deletedClusters;

    for (ClusterMergeMap::const_iterator iter = clusterMergeMap.begin(), iterEnd = clusterMergeMap.end(); iter != iterEnd; ++iter)
    {
        Cluster *pParentCluster = iter->first;

        const HitType hitType(LArThreeDHelper::GetClusterHitType(pParentCluster));
        const std::string clusterListName((TPC_VIEW_U == hitType) ? pAlgorithm->GetClusterListNameU() : (TPC_VIEW_V == hitType) ? pAlgorithm->GetClusterListNameV() : pAlgorithm->GetClusterListNameW());

        if (!((TPC_VIEW_U == hitType) || (TPC_VIEW_V == hitType) || (TPC_VIEW_W == hitType)))
            throw StatusCodeException(STATUS_CODE_FAILURE);

        for (ClusterList::const_iterator dIter = iter->second.begin(), dIterEnd = iter->second.end(); dIter != dIterEnd; ++dIter)
        {
            Cluster *pDaughterCluster = *dIter;

            if (deletedClusters.count(pParentCluster) || deletedClusters.count(pDaughterCluster))
                throw StatusCodeException(STATUS_CODE_FAILURE);

            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::MergeAndDeleteClusters(*pAlgorithm, pParentCluster, pDaughterCluster, clusterListName, clusterListName));
            pAlgorithm->UpdateUponMerge(pParentCluster, pDaughterCluster);
            deletedClusters.insert(pDaughterCluster);
        }
    }

    return !(deletedClusters.empty());
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDKinkBaseTool::MakeClusterSplits(ThreeDTransverseTracksAlgorithm *pAlgorithm, const SplitPositionMap &splitPositionMap) const
{
    bool changesMade(false);

    for (SplitPositionMap::const_iterator iter = splitPositionMap.begin(), iterEnd = splitPositionMap.end(); iter != iterEnd; ++iter)
    {
        Cluster *pCurrentCluster = iter->first;
        CartesianPointList splitPositions(iter->second);
        std::sort(splitPositions.begin(), splitPositions.end(), ThreeDKinkBaseTool::SortSplitPositions);

        const HitType hitType(LArThreeDHelper::GetClusterHitType(pCurrentCluster));
        const std::string clusterListName((TPC_VIEW_U == hitType) ? pAlgorithm->GetClusterListNameU() : (TPC_VIEW_V == hitType) ? pAlgorithm->GetClusterListNameV() : pAlgorithm->GetClusterListNameW());

        if (!((TPC_VIEW_U == hitType) || (TPC_VIEW_V == hitType) || (TPC_VIEW_W == hitType)))
            throw StatusCodeException(STATUS_CODE_FAILURE);

        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Cluster>(*pAlgorithm, clusterListName));

        for (CartesianPointList::const_iterator sIter = splitPositions.begin(), sIterEnd = splitPositions.end(); sIter != sIterEnd; ++sIter)
        {
            const CartesianVector &splitPosition(*sIter);

            Cluster *pLowXCluster(NULL), *pHighXCluster(NULL);
            this->MakeClusterSplit(pAlgorithm, *sIter, pCurrentCluster, pLowXCluster, pHighXCluster);

            pAlgorithm->UpdateUponSplit(pLowXCluster, pHighXCluster, pCurrentCluster);
            changesMade = true;
            pCurrentCluster = pHighXCluster;
        }
    }

    return changesMade;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDKinkBaseTool::MakeClusterSplit(ThreeDTransverseTracksAlgorithm *pAlgorithm, const CartesianVector &splitPosition,
    Cluster *&pCurrentCluster, Cluster *&pLowXCluster, Cluster *&pHighXCluster) const
{
    pLowXCluster = NULL;
    pHighXCluster = NULL;

    std::string originalListName, fragmentListName;
    ClusterList clusterList; clusterList.insert(pCurrentCluster);
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::InitializeFragmentation(*pAlgorithm, clusterList, originalListName, fragmentListName));

    CaloHitList caloHitList;
    pCurrentCluster->GetOrderedCaloHitList().GetCaloHitList(caloHitList);

    LArPointingCluster pointingCluster(pCurrentCluster);
    const bool innerIsLowX(pointingCluster.GetInnerVertex().GetPosition().GetX() < pointingCluster.GetOuterVertex().GetPosition().GetX());
    const CartesianVector &lowXEnd(innerIsLowX ? pointingCluster.GetInnerVertex().GetPosition() : pointingCluster.GetOuterVertex().GetPosition());
    const CartesianVector &highXEnd(innerIsLowX ? pointingCluster.GetOuterVertex().GetPosition() : pointingCluster.GetInnerVertex().GetPosition());

    const CartesianVector lowXUnitVector((lowXEnd -splitPosition).GetUnitVector());
    const CartesianVector highXUnitVector((highXEnd -splitPosition).GetUnitVector());

    for (CaloHitList::const_iterator hIter = caloHitList.begin(), hIterEnd = caloHitList.end(); hIter != hIterEnd; ++hIter)
    {
        CaloHit *pCaloHit = *hIter;
        const CartesianVector unitVector((pCaloHit->GetPositionVector() - splitPosition).GetUnitVector());

        const float dotProductLowX(unitVector.GetDotProduct(lowXUnitVector));
        const float dotProductHighX(unitVector.GetDotProduct(highXUnitVector));
        Cluster *&pClusterToModify((dotProductLowX > dotProductHighX) ? pLowXCluster : pHighXCluster);

        if (NULL == pClusterToModify)
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*pAlgorithm, pCaloHit, pClusterToModify));
        }
        else
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToCluster(*pAlgorithm, pClusterToModify, pCaloHit));
        }
    }

    if ((NULL == pLowXCluster) || (NULL == pHighXCluster))
        throw StatusCodeException(STATUS_CODE_FAILURE);

    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::EndFragmentation(*pAlgorithm, fragmentListName, originalListName));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDKinkBaseTool::SortSplitPositions(const pandora::CartesianVector &lhs, const pandora::CartesianVector &rhs)
{
    return (lhs.GetX() < rhs.GetX());
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeDKinkBaseTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    m_minMatchedFraction = 0.75f;
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinMatchedFraction", m_minMatchedFraction));

    m_minMatchedSamplingPoints = 10;
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinMatchedSamplingPoints", m_minMatchedSamplingPoints));

    m_minLongitudinalImpactParameter = -1.f;
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinLongitudinalImpactParameter", m_minLongitudinalImpactParameter));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar
