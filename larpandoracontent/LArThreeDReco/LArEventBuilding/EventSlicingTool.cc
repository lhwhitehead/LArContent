/**
 *  @file   larpandoracontent/LArThreeDReco/LArEventBuilding/EventSlicingTool.cc
 *
 *  @brief  Implementation of the event slicing tool class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArPointingClusterHelper.h"

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"
#include "larpandoracontent/LArObjects/LArThreeDSlidingConeFitResult.h"

#include "larpandoracontent/LArThreeDReco/LArEventBuilding/EventSlicingTool.h"

#include "larpandoracontent/LArUtility/KDTreeLinkerAlgoT.h"

using namespace pandora;

namespace lar_content
{

typedef SlicingAlgorithm::HitTypeToNameMap HitTypeToNameMap;
typedef SlicingAlgorithm::SliceList SliceList;
typedef SlicingAlgorithm::Slice Slice;

//------------------------------------------------------------------------------------------------------------------------------------------

EventSlicingTool::EventSlicingTool() :
    m_minHitsPer3DCluster(20),
    m_min3DHitsToSeedNewSlice(50),
    m_halfWindowLayers(20),
    m_usePointingAssociation(true),
    m_minVertexLongitudinalDistance(-7.5f),
    m_maxVertexLongitudinalDistance(60.f),
    m_maxVertexTransverseDistance(10.5f),
    m_vertexAngularAllowance(9.f),
    m_maxClosestApproach(15.f),
    m_maxInterceptDistance(60.f),
    m_useProximityAssociation(true),
    m_maxHitSeparationSquared(25.f * 25.f),
    m_useShowerConeAssociation(true),
    m_nConeFitLayers(20),
    m_nConeFits(5),
    m_coneLengthMultiplier(7.f),
    m_maxConeLength(126.f),
    m_coneTanHalfAngle1(0.5f),
    m_coneBoundedFraction1(0.5f),
    m_coneTanHalfAngle2(0.75f),
    m_coneBoundedFraction2(0.75f),
    m_use3DProjectionsInHitPickUp(true),
    m_useSliceSplit(true),
    m_sliceSplitTrackMaxDistance(50.),
    m_sliceSplitShowerMaxDistance(100.),
    m_sliceSplitVertexMaxDistance(10.)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::RunSlicing(const Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames, const HitTypeToNameMap &clusterListNames,
    SliceList &sliceList)
{
    if (PandoraContentApi::GetSettings(*pAlgorithm)->ShouldDisplayAlgorithmInfo())
       std::cout << "----> Running Algorithm Tool: " << this->GetInstanceName() << ", " << this->GetType() << std::endl;

    ClusterToPfoMap clusterToPfoMap;

    ClusterList trackClusters3D;
    this->GetThreeDClusters(pAlgorithm, m_trackPfoListName, trackClusters3D, clusterToPfoMap);

    ClusterList showerClusters3D;
    this->GetThreeDClusters(pAlgorithm, m_showerPfoListName, showerClusters3D, clusterToPfoMap);

    ClusterSliceList clusterSliceList;
    this->GetClusterSliceList(trackClusters3D, showerClusters3D, clusterSliceList);

    if (clusterSliceList.size() < 2)
    {
        return this->CopyAllHitsToSingleSlice(pAlgorithm, caloHitListNames, sliceList);
    }
    else
    {
        ClusterToSliceIndexMap clusterToSliceIndexMap;
        this->CreateSlices(clusterSliceList, sliceList, clusterToSliceIndexMap);

        ClusterSet assignedClusters;
        this->CopyPfoHitsToSlices(clusterToSliceIndexMap, clusterToPfoMap, sliceList, assignedClusters);

        ClusterList remainingClusters;
        this->GetRemainingClusters(pAlgorithm, clusterListNames, assignedClusters, remainingClusters);

        //std::cout << "Total of " << remainingClusters.size() << " clusters to have their hits added to slices" << std::endl;

        this->AssignRemainingHitsToSlices(remainingClusters, clusterToSliceIndexMap, sliceList);

        //std::cout << "Final number of slices = " << sliceList.size() << std::endl;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::CopyAllHitsToSingleSlice(const Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames, SliceList &sliceList) const
{
    if (!sliceList.empty())
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

    const CaloHitList *pCaloHitListU(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*pAlgorithm,
        caloHitListNames.at(TPC_VIEW_U), pCaloHitListU));

    const CaloHitList *pCaloHitListV(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*pAlgorithm,
        caloHitListNames.at(TPC_VIEW_V), pCaloHitListV));

    const CaloHitList *pCaloHitListW(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*pAlgorithm,
        caloHitListNames.at(TPC_VIEW_W), pCaloHitListW));

    if (pCaloHitListU || pCaloHitListV || pCaloHitListW)
    {
        sliceList.push_back(Slice());
        Slice &slice(sliceList.at(0));

        if (pCaloHitListU) slice.m_caloHitListU = *pCaloHitListU;
        if (pCaloHitListV) slice.m_caloHitListV = *pCaloHitListV;
        if (pCaloHitListW) slice.m_caloHitListW = *pCaloHitListW;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetThreeDClusters(const Algorithm *const pAlgorithm, const std::string &pfoListName, ClusterList &clusters3D,
    ClusterToPfoMap &clusterToPfoMap) const
{
    const PfoList *pPfoList(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*pAlgorithm, pfoListName, pPfoList));

    if (!pPfoList || pPfoList->empty())
    {
        if (PandoraContentApi::GetSettings(*pAlgorithm)->ShouldDisplayAlgorithmInfo())
            std::cout << "EventSlicingTool: unable to find pfo list " << pfoListName << std::endl;

        return;
    }

    for (const ParticleFlowObject *const pPfo : *pPfoList)
    {
        ClusterList pfoClusters3D;
        LArPfoHelper::GetThreeDClusterList(pPfo, pfoClusters3D);

        if (pPfo->GetDaughterPfoList().size() > 0)
            std::cout << "Pfo " << pPfo << " has at least one daughter" << std::endl;
        
        if (pfoClusters3D.size() > 1)
            std::cout << "Pfo " << pPfo << " has at least two clusters" << std::endl;

        for (const Cluster *const pCluster3D : pfoClusters3D)
        {
            if (pCluster3D->GetNCaloHits() < m_minHitsPer3DCluster)
                continue;

            if (!clusterToPfoMap.insert(ClusterToPfoMap::value_type(pCluster3D, pPfo)).second)
                throw StatusCodeException(STATUS_CODE_ALREADY_PRESENT);

            if (clusters3D.end() != std::find(clusters3D.begin(), clusters3D.end(), pCluster3D))
                throw StatusCodeException(STATUS_CODE_ALREADY_PRESENT);

            clusters3D.push_back(pCluster3D);
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetClusterSliceList(const ClusterList &trackClusters3D, const ClusterList &showerClusters3D,
    ClusterSliceList &clusterSliceList) const
{
    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));

    ThreeDSlidingFitResultMap trackFitResults;

    for (const Cluster *const pCluster3D : trackClusters3D)
    {
        try {trackFitResults.insert(ThreeDSlidingFitResultMap::value_type(pCluster3D, ThreeDSlidingFitResult(pCluster3D, m_halfWindowLayers, layerPitch)));}
        catch (StatusCodeException &) {std::cout << "EventSlicingTool: ThreeDSlidingFitResult failure for track cluster." << std::endl;}
    }

    ThreeDSlidingConeFitResultMap showerConeFitResults;

    for (const Cluster *const pCluster3D : showerClusters3D)
    {
        try {showerConeFitResults.insert(ThreeDSlidingConeFitResultMap::value_type(pCluster3D, ThreeDSlidingConeFitResult(pCluster3D, m_halfWindowLayers, layerPitch)));}
        catch (StatusCodeException &) {std::cout << "EventSlicingTool: ThreeDSlidingConeFitResult failure for shower cluster." << std::endl;}
    }

    ClusterVector sortedClusters3D(trackClusters3D.begin(), trackClusters3D.end());
    sortedClusters3D.insert(sortedClusters3D.end(), showerClusters3D.begin(), showerClusters3D.end());
    std::sort(sortedClusters3D.begin(), sortedClusters3D.end(), LArClusterHelper::SortByNHits);

    ClusterSet usedClusters;

    for (const Cluster *const pCluster3D : sortedClusters3D)
    {
        if (usedClusters.count(pCluster3D))
            continue;

        if (pCluster3D->GetNCaloHits() < m_min3DHitsToSeedNewSlice)
            continue;

        clusterSliceList.push_back(ClusterVector(1, pCluster3D));
        usedClusters.insert(pCluster3D);

        ClusterVector &clusterSlice(clusterSliceList.back());
        this->CollectAssociatedClusters(pCluster3D, sortedClusters3D, trackFitResults, showerConeFitResults, clusterSlice, usedClusters);

        //std::cout << "Slice has " << clusterSlice.size() << " clusters" << std::endl;

        if (m_useSliceSplit)
          this->CheckAndDivideSlice(clusterSlice,usedClusters, trackFitResults, showerConeFitResults);
    }

    //std::cout << "Used " << usedClusters.size() << " / " << sortedClusters3D.size() << " clusters" << std::endl;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::CollectAssociatedClusters(const Cluster *const pClusterInSlice, const ClusterVector &candidateClusters,
    const ThreeDSlidingFitResultMap &trackFitResults, const ThreeDSlidingConeFitResultMap &showerConeFitResults, ClusterVector &clusterSlice,
    ClusterSet &usedClusters) const
{
    ClusterVector addedClusters;

    for (const Cluster *const pCandidateCluster : candidateClusters)
    {
        if (usedClusters.count(pCandidateCluster) || (pClusterInSlice == pCandidateCluster))
            continue;

        if ((m_usePointingAssociation && this->PassPointing(pClusterInSlice, pCandidateCluster, trackFitResults)) ||
            (m_useProximityAssociation && this->PassProximity(pClusterInSlice, pCandidateCluster)) ||
            (m_useShowerConeAssociation && (this->PassShowerCone(pClusterInSlice, pCandidateCluster, showerConeFitResults) || this->PassShowerCone(pCandidateCluster, pClusterInSlice, showerConeFitResults))) )
        {
            addedClusters.push_back(pCandidateCluster);
            (void) usedClusters.insert(pCandidateCluster);
        }
    }

    clusterSlice.insert(clusterSlice.end(), addedClusters.begin(), addedClusters.end());

    for (const Cluster *const pAddedCluster : addedClusters)
        this->CollectAssociatedClusters(pAddedCluster, candidateClusters, trackFitResults, showerConeFitResults, clusterSlice, usedClusters);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::CheckAndDivideSlice(pandora::ClusterVector &sliceClusters, pandora::ClusterSet &usedClusters, const ThreeDSlidingFitResultMap &trackFitResults, const ThreeDSlidingConeFitResultMap &showerConeFitResults) const
{
    //std::cout << "===== Checking slice with primary cluster " << sliceClusters.at(0) << " with " << sliceClusters.at(0)->GetOrderedCaloHitList().size() << " hits" << std::endl;

    // The above will miss slices with two hierarchies, so tackle that now
    const unsigned int origSize = sliceClusters.size();
    this->RemoveDistantHierarchy(sliceClusters,usedClusters,trackFitResults,showerConeFitResults);
    if(origSize != sliceClusters.size()){
        std::cout << "We made a change to this slice." << std::endl;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::RemoveDistantHierarchy(pandora::ClusterVector &sliceClusters, pandora::ClusterSet &usedClusters, const ThreeDSlidingFitResultMap &trackFitResults, const ThreeDSlidingConeFitResultMap &showerConeFitResults) const
{
    if (2 > sliceClusters.size())
        return;

    // Get the cluster extremes first to avoid multiple calls
    std::map<const Cluster*,std::vector<pandora::CartesianVector>> clusterVerticesMap;
    for (const Cluster *const pCluster : sliceClusters)
    {
        if (clusterVerticesMap.find(pCluster) == clusterVerticesMap.end())
        {
            std::vector<pandora::CartesianVector> vertices;
            this->GetClusterExtremes(pCluster,trackFitResults,showerConeFitResults,vertices);
            clusterVerticesMap.insert(std::make_pair(pCluster,vertices));
        }
    }

    std::map<const Cluster*,std::map<const Cluster*,float>> clusterDistanceMap;
    std::map<const Cluster*,std::vector<const Cluster*>> closestClusters;
    for (const Cluster *const pCluster1 : sliceClusters)
    {
        std::map<const Cluster*,float> cluster1Distances;
        const std::vector<pandora::CartesianVector> cluster1Vertices = clusterVerticesMap.at(pCluster1);

//        for (const Cluster *const pCluster2 : sliceClusters)
        closestClusters.insert(std::make_pair(pCluster1,std::vector<const Cluster*>()));          
        for (unsigned int c = 0; c < sliceClusters.size(); ++c)
        {
            const Cluster *pCluster2 = sliceClusters.at(c);
            if (pCluster1 == pCluster2)
                continue;

            const std::vector<pandora::CartesianVector> cluster2Vertices = clusterVerticesMap.at(pCluster2);
            // Get the minimum distance between the vertices
            float minDist = std::numeric_limits<float>::max();
            for (auto const &v1 : cluster1Vertices)
            {
                for (auto const &v2 : cluster2Vertices)
                {
                    const float dist = (v1-v2).GetMagnitude();
                    if (dist < minDist)
                        minDist = dist;       
                }
            }
            cluster1Distances.insert(std::make_pair(pCluster2,minDist));

            const float vtxApp = this->GetClosestApproachToVertex(cluster2Vertices,pCluster1);
            if (vtxApp < m_sliceSplitVertexMaxDistance)
            {
                closestClusters.at(pCluster1).push_back(pCluster2);
            }
        }
        clusterDistanceMap.insert(std::make_pair(pCluster1,cluster1Distances));
        //std::cout << "Cluster " << pCluster1 << " at (" << cluster1Vertices.at(0).GetX() << ", " << cluster1Vertices.at(0).GetZ() << ") " << " with " << pCluster1->GetOrderedCaloHitList().size() << " hits has " << closestClusters.at(pCluster1).size() << " clusters with vertices within " << m_sliceSplitVertexMaxDistance << " cm of its hits: ";
//        for(const Cluster *const pCloseCluster : closestClusters.at(pCluster1))
//        {
            //std::cout << pCloseCluster << ", ";
//        }
        //std::cout << std::endl;
        
    }

    //std::cout << usedClusters.size() << ", " << clusterDistanceMap.size() << std::endl;
    if( 2 > clusterDistanceMap.size())
        return;

    std::vector<const Cluster*> clusteredClusters;
    std::vector<std::vector<const Cluster*>> allHierarchies;
    std::map<const Cluster*, unsigned int> clusterHierarchyMap;
    while (clusteredClusters.size() < clusterDistanceMap.size())
    {
        std::vector<const Cluster*> newHierarchy;
        bool changeMade = true;
        unsigned int counter = 0;
        while (changeMade)
        {
//            //std::cout << "Starting iteration " << ++counter << std::endl;
            bool localChangeMade = false;
            for (auto const *c1 : sliceClusters)
            {
                if (std::find(clusteredClusters.begin(),clusteredClusters.end(),c1) != clusteredClusters.end())
                    continue;

                if (newHierarchy.empty())
                {   
                    //std::cout << "Seeding hierarchy with cluster " << c1 << std::endl;
                    newHierarchy.push_back(c1);
                    clusteredClusters.push_back(c1);
                    clusterHierarchyMap.insert(std::make_pair(c1,allHierarchies.size()));
                    localChangeMade = true;
                    break;
                }
                else
                {
                    for (auto const *pHierCluster : newHierarchy)
                    {
                        if (pHierCluster == c1)
                            continue;
                        const float dist = clusterDistanceMap.at(c1).at(pHierCluster);
                        const bool isTrkSliceCluster = (trackFitResults.end() != trackFitResults.find(c1));
                        const bool isShwSliceCluster = (showerConeFitResults.end() != showerConeFitResults.find(c1));

                        const bool isTrkHierCluster = (trackFitResults.end() != trackFitResults.find(pHierCluster));
                        const bool isShwHierCluster = (showerConeFitResults.end() != showerConeFitResults.find(pHierCluster));

                        const bool isCloseCluster = std::find(closestClusters.at(pHierCluster).begin(),closestClusters.at(pHierCluster).end(),c1) != closestClusters.at(pHierCluster).end();

                        // Use different cut values for tracks and showers
                        float cutValue = m_sliceSplitTrackMaxDistance;

                        // Use the shower distance if we are comparing shower -> X, or X -> shower
                        if (isShwHierCluster) cutValue = m_sliceSplitShowerMaxDistance;
                        if (isTrkHierCluster && isShwSliceCluster) cutValue = m_sliceSplitShowerMaxDistance;
                        if (!isTrkSliceCluster && !isShwSliceCluster) cutValue = m_sliceSplitShowerMaxDistance;
                        if (!isTrkHierCluster && !isShwHierCluster) cutValue = m_sliceSplitShowerMaxDistance;

                        if (dist < cutValue || isCloseCluster)// || vtxdist1 < m_sliceSplitVtxApproachMaxDistance || vtxdist2 < m_sliceSplitVtxApproachMaxDistance)
                        {
                            //std::cout << isTrkHierCluster << isShwHierCluster << isTrkSliceCluster << isShwSliceCluster << std::endl;
                            //std::cout << "Adding cluster " << c1 << " to hierarchy with " << dist << " cm from " << pHierCluster << ". Was it a special close cluster? " << isCloseCluster << std::endl;
                            newHierarchy.push_back(c1);
                            clusteredClusters.push_back(c1);
                            clusterHierarchyMap.insert(std::make_pair(c1,allHierarchies.size()));
                            localChangeMade = true;
                            break;
                        }
                    }        
                }
            }
            changeMade = localChangeMade;
            //std::cout << "Ending iteraction " << counter << " and used " << clusteredClusters.size() << " of " << clusterDistanceMap.size() << " clusters" << std::endl;
            ++counter;
        }
        //std::cout << "Made hierarchy with " << newHierarchy.size() << " clusters" << std::endl;
        allHierarchies.push_back(newHierarchy);
    }
    //std::cout << "Made " << allHierarchies.size() << " hierarchies" << std::endl;

    if (allHierarchies.size() < 2)
        return;

/*
    // Check the clusters flagged for removal to see if any other cluster have a vertex very close to the cluster
//    for (size_t h = 0; h < allHierarchies.size(); ++h)
//    {
//    for (const Cluster *const pClusterToRemove : allHierarchies.at(h))
      std::vector<const Cluster*> clustersToAdd;
        for (size_t c = 0; c < allHierarchies.at(0).size(); ++c)
        {
            const Cluster *const pClusterToKeep = allHierarchies.at(0).at(c);
                
            // We only want to use track-like or small clusters here since showers are easy to break
//            if (showerConeFitResults.end() == showerConeFitResults.find(pClusterToKeep))
//                continue;

            //std::cout << "Looking for associations for cluster " << pClusterToKeep << " in hierarchy 0" << std::endl;
            for (const Cluster *const pSliceCluster : sliceClusters)
            {
                if (pClusterToKeep == pSliceCluster)
                    continue;
    
                const unsigned int oldHierarchy = clusterHierarchyMap.at(pSliceCluster);
                if (oldHierarchy == 0)
                    continue;
    
                // We only want to use track-like or small clusters here since showers are easy to break
//                if (showerConeFitResults.end() == showerConeFitResults.find(pSliceCluster))
//                    continue;

                const float minDist = this->GetClosestApproachToVertex(clusterVerticesMap.at(pSliceCluster),pClusterToKeep);
                if ((minDist < m_sliceSplitVtxApproachMaxDistance) && (std::find(closestClusters.at(pClusterToKeep).begin(),closestClusters.at(pClusterToKeep).end(),pSliceCluster) != closestClusters.at(pClusterToKeep).end()))
                {
                    //std::cout << "Moving cluster " << pSliceCluster << " from hierarchy " << oldHierarchy << " to 0" << std::endl;
                    // Remove from other hierarchy but don't add to the new one yet to prevent recursion
                    allHierarchies.at(oldHierarchy).erase(std::remove(allHierarchies.at(oldHierarchy).begin(),allHierarchies.at(oldHierarchy).end(),pSliceCluster),allHierarchies.at(oldHierarchy).end());
                    clustersToAdd.push_back(pSliceCluster);
                    clusterHierarchyMap.at(pSliceCluster) = 0;
                }
            }
    
        }
        allHierarchies.at(0).insert(allHierarchies.at(0).end(),clustersToAdd.begin(),clustersToAdd.end());
//    }
*/
    for (size_t h = 1; h < allHierarchies.size(); ++h)
    {
        for (const Cluster *const pClusterToRemove : allHierarchies.at(h))
        {   
            //std::cout << "Removing cluster with address " << pClusterToRemove << " and " << pClusterToRemove->GetOrderedCaloHitList().size() << " hits from the slice" << std::endl;
            usedClusters.erase(pClusterToRemove);
            sliceClusters.erase(std::remove(sliceClusters.begin(), sliceClusters.end(), pClusterToRemove),sliceClusters.end());
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::PassPointing(const Cluster *const pClusterInSlice, const Cluster *const pCandidateCluster, const ThreeDSlidingFitResultMap &trackFitResults) const
{
    ThreeDSlidingFitResultMap::const_iterator inSliceIter = trackFitResults.find(pClusterInSlice);
    ThreeDSlidingFitResultMap::const_iterator candidateIter = trackFitResults.find(pCandidateCluster);

    if ((trackFitResults.end() == inSliceIter) || (trackFitResults.end() == candidateIter))
        return false;

    const LArPointingCluster inSlicePointingCluster(inSliceIter->second);
    const LArPointingCluster candidatePointingCluster(candidateIter->second);

    if (this->CheckClosestApproach(inSlicePointingCluster, candidatePointingCluster) ||
        this->IsEmission(inSlicePointingCluster, candidatePointingCluster) ||
        this->IsNode(inSlicePointingCluster, candidatePointingCluster))
    {
        return true;
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::PassProximity(const Cluster *const pClusterInSlice, const Cluster *const pCandidateCluster) const
{
    for (const auto &orderedList1 : pClusterInSlice->GetOrderedCaloHitList())
    {
        for (const CaloHit *const pCaloHit1 : *(orderedList1.second))
        {
            const CartesianVector &positionVector1(pCaloHit1->GetPositionVector());

            for (const auto &orderedList2 : pCandidateCluster->GetOrderedCaloHitList())
            {
                for (const CaloHit *const pCaloHit2 : *(orderedList2.second))
                {
                    if ((positionVector1 - pCaloHit2->GetPositionVector()).GetMagnitudeSquared() < m_maxHitSeparationSquared)
                        return true;
                }
            }
        }
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float EventSlicingTool::GetProximity(const Cluster *const pCluster1, const Cluster *const pCluster2) const
{
    float minDist = std::numeric_limits<float>::max();
    for (const auto &orderedList1 : pCluster1->GetOrderedCaloHitList())
    {   
        for (const CaloHit *const pCaloHit1 : *(orderedList1.second))
        {   
            const CartesianVector &positionVector1(pCaloHit1->GetPositionVector());
            
            for (const auto &orderedList2 : pCluster2->GetOrderedCaloHitList())
            {   
                for (const CaloHit *const pCaloHit2 : *(orderedList2.second))
                {   
                    const float dist = (positionVector1 - pCaloHit2->GetPositionVector()).GetMagnitudeSquared();
                    if (dist < minDist)
                        minDist = dist;
                }
            }
        }
    }
    
    return minDist;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float EventSlicingTool::GetVertexProximity(const std::vector<pandora::CartesianVector> &cluster1Vertices, const std::vector<pandora::CartesianVector> &cluster2Vertices) const
{

    float minDistance = std::numeric_limits<float>::max();

    if (cluster1Vertices.size() == 0 || cluster2Vertices.size() == 0)
        return std::numeric_limits<float>::max();

    for (const pandora::CartesianVector &v1 : cluster1Vertices)
    {
        for (const pandora::CartesianVector &v2 : cluster2Vertices)
        {
            const float dist = (v2-v1).GetMagnitude();
            if (minDistance > dist)
                minDistance = dist;
        }
    }

    return minDistance;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetClusterExtremes(const pandora::Cluster *const pCluster, const ThreeDSlidingFitResultMap &trackFitResults, const ThreeDSlidingConeFitResultMap &showerConeFitResults, std::vector<pandora::CartesianVector> &vertices) const
{
   // Search for track fits first
    ThreeDSlidingFitResultMap::const_iterator inSliceIter = trackFitResults.find(pCluster);
    if (trackFitResults.end() != inSliceIter)
    {
//        //std::cout << " - Looks like we have a track..." << std::endl;
        const LArPointingCluster inSlicePointingCluster(inSliceIter->second);
        vertices.push_back(inSlicePointingCluster.GetInnerVertex().GetPosition());
        vertices.push_back(inSlicePointingCluster.GetOuterVertex().GetPosition());
    }
    else
    {
        // Otherwise find the shower information
        ThreeDSlidingConeFitResultMap::const_iterator inSliceShwIter = showerConeFitResults.find(pCluster);
        if (showerConeFitResults.end() != inSliceShwIter)
        {
//            //std::cout << " - Looks like we have a shower..." << std::endl;
            const ThreeDSlidingConeFitResult &slidingConeFitResult3D(inSliceShwIter->second);
            SimpleConeList simpleConeList;
//            //std::cout << "   - Getting cone list" << std::endl;
            try
            {
                slidingConeFitResult3D.GetSimpleConeList(m_nConeFitLayers, m_nConeFits, CONE_BOTH_DIRECTIONS, simpleConeList);
                vertices.push_back(simpleConeList.front().GetConeApex());
                if(simpleConeList.size() > 1)
                    vertices.push_back(simpleConeList.back().GetConeApex());
            }
            catch (const StatusCodeException &)
            {
                // Fall back on just getting the extreme positions (ordered by z)
                pandora::CartesianVector minPos(0.,0.,0.);
                pandora::CartesianVector maxPos(0.,0.,0.);
                LArClusterHelper::GetExtremalCoordinates(pCluster,minPos,maxPos);
                vertices.push_back(minPos);
                vertices.push_back(maxPos);
                //std::cout << "Had to use fall-back method to get vertices for cluster " << pCluster << " with " << pCluster->GetOrderedCaloHitList().size() << " hits" << std::endl;
            }
//            //std::cout << "   - Got " << simpleConeList.size() << " cones" << std::endl;
//            //std::cout << "Looking at cones..." << std::endl;
//            for (auto const & cone : simpleConeList)
//            {
//              //std::cout << " - Cone position = " << cone.GetConeApex().GetX() << ", " 
//                                                 << cone.GetConeApex().GetY() << ", " 
//                                                 << cone.GetConeApex().GetZ() << std::endl; 
//            }
        }
    }
}


//------------------------------------------------------------------------------------------------------------------------------------------
float EventSlicingTool::GetClosestApproachToVertex(const std::vector<pandora::CartesianVector> &vertices, const Cluster *const pCluster) const
{
    float minDist = std::numeric_limits<float>::max();
    for (const auto &vtx : vertices)
    {
        for (const auto &orderedList : pCluster->GetOrderedCaloHitList())
        {
            for (const CaloHit *const pCaloHit : *(orderedList.second))
            {
                const float dist = (vtx - pCaloHit->GetPositionVector()).GetMagnitude();
                if (minDist > dist)
                    minDist = dist;
            }
        }
    }
    return minDist;   
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::PassShowerCone(const Cluster *const pConeCluster, const Cluster *const pNearbyCluster, const ThreeDSlidingConeFitResultMap &showerConeFitResults) const
{
    ThreeDSlidingConeFitResultMap::const_iterator fitIter = showerConeFitResults.find(pConeCluster);

    if (showerConeFitResults.end() == fitIter)
        return false;

    float clusterLength(0.f);
    SimpleConeList simpleConeList;

    try
    {
        const ThreeDSlidingConeFitResult &slidingConeFitResult3D(fitIter->second);
        const ThreeDSlidingFitResult &slidingFitResult3D(slidingConeFitResult3D.GetSlidingFitResult());
        slidingConeFitResult3D.GetSimpleConeList(m_nConeFitLayers, m_nConeFits, CONE_BOTH_DIRECTIONS, simpleConeList);
        clusterLength = (slidingFitResult3D.GetGlobalMaxLayerPosition() - slidingFitResult3D.GetGlobalMinLayerPosition()).GetMagnitude();
    }
    catch (const StatusCodeException &)
    {
        return false;
    }

    for (const SimpleCone &simpleCone : simpleConeList)
    {
        const float coneLength(std::min(m_coneLengthMultiplier * clusterLength, m_maxConeLength));

        if (simpleCone.GetBoundedHitFraction(pNearbyCluster, coneLength, m_coneTanHalfAngle1) < m_coneBoundedFraction1)
            continue;

        if (simpleCone.GetBoundedHitFraction(pNearbyCluster, coneLength, m_coneTanHalfAngle2) < m_coneBoundedFraction2)
            continue;

        return true;
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::CheckClosestApproach(const LArPointingCluster &cluster1, const LArPointingCluster &cluster2) const
{
    return (this->CheckClosestApproach(cluster1.GetInnerVertex(), cluster2.GetInnerVertex()) ||
        this->CheckClosestApproach(cluster1.GetOuterVertex(), cluster2.GetInnerVertex()) ||
        this->CheckClosestApproach(cluster1.GetInnerVertex(), cluster2.GetOuterVertex()) ||
        this->CheckClosestApproach(cluster1.GetOuterVertex(), cluster2.GetOuterVertex()));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::CheckClosestApproach(const LArPointingCluster::Vertex &vertex1, const LArPointingCluster::Vertex &vertex2) const
{
    CartesianVector intersectionPoint(0.f, 0.f, 0.f);
    float displacement1(std::numeric_limits<float>::max()), displacement2(std::numeric_limits<float>::max());

    try
    {
        LArPointingClusterHelper::GetIntersection(vertex1, vertex2, intersectionPoint, displacement1, displacement2);
    }
    catch (const StatusCodeException &)
    {
        return false;
    }

    const CartesianVector approach1(vertex1.GetPosition() + vertex1.GetDirection() * displacement1);
    const CartesianVector approach2(vertex2.GetPosition() + vertex2.GetDirection() * displacement2);
    const float closestApproach((approach1 - approach2).GetMagnitude());

    return ((closestApproach < m_maxClosestApproach) && (std::fabs(displacement1) < m_maxInterceptDistance) && (std::fabs(displacement2) < m_maxInterceptDistance));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::IsNode(const LArPointingCluster &cluster1, const LArPointingCluster &cluster2) const
{
    return (LArPointingClusterHelper::IsNode(cluster1.GetInnerVertex().GetPosition(), cluster2.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster1.GetOuterVertex().GetPosition(), cluster2.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster1.GetInnerVertex().GetPosition(), cluster2.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster1.GetOuterVertex().GetPosition(), cluster2.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster2.GetInnerVertex().GetPosition(), cluster1.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster2.GetOuterVertex().GetPosition(), cluster1.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster2.GetInnerVertex().GetPosition(), cluster1.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance) ||
        LArPointingClusterHelper::IsNode(cluster2.GetOuterVertex().GetPosition(), cluster1.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexTransverseDistance));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::IsEmission(const LArPointingCluster &cluster1, const LArPointingCluster &cluster2) const
{
    return (LArPointingClusterHelper::IsEmission(cluster1.GetInnerVertex().GetPosition(), cluster2.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster1.GetOuterVertex().GetPosition(), cluster2.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster1.GetInnerVertex().GetPosition(), cluster2.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster1.GetOuterVertex().GetPosition(), cluster2.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster2.GetInnerVertex().GetPosition(), cluster1.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster2.GetOuterVertex().GetPosition(), cluster1.GetInnerVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster2.GetInnerVertex().GetPosition(), cluster1.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance) ||
        LArPointingClusterHelper::IsEmission(cluster2.GetOuterVertex().GetPosition(), cluster1.GetOuterVertex(), m_minVertexLongitudinalDistance, m_maxVertexLongitudinalDistance, m_maxVertexTransverseDistance, m_vertexAngularAllowance));
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::CreateSlices(const ClusterSliceList &clusterSliceList, SliceList &sliceList, ClusterToSliceIndexMap &clusterToSliceIndexMap) const
{
    unsigned int index(0);

    for (const ClusterVector &clusterList : clusterSliceList)
    {
        for (const Cluster *const pCluster3D : clusterList)
        {
            if (TPC_3D != LArClusterHelper::GetClusterHitType(pCluster3D))
                throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

            if (!clusterToSliceIndexMap.insert(ClusterToSliceIndexMap::value_type(pCluster3D, index)).second)
                throw StatusCodeException(STATUS_CODE_ALREADY_PRESENT);
        }

        sliceList.push_back(Slice());
        ++index;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::CopyPfoHitsToSlices(const ClusterToSliceIndexMap &clusterToSliceIndexMap, const ClusterToPfoMap &clusterToPfoMap,
    SliceList &sliceList, ClusterSet &assignedClusters) const
{
    ClusterList clusterList;
    for (const auto &mapEntry : clusterToSliceIndexMap) clusterList.push_back(mapEntry.first);
    clusterList.sort(LArClusterHelper::SortByNHits);

    for (const Cluster *const pCluster3D : clusterList)
    {
        const unsigned int index(clusterToSliceIndexMap.at(pCluster3D));

        const Pfo *const pPfo(clusterToPfoMap.at(pCluster3D));
        Slice &slice(sliceList.at(index));

        ClusterList clusters2D;
        LArPfoHelper::GetTwoDClusterList(pPfo, clusters2D);

        for (const Cluster *const pCluster2D : clusters2D)
        {
            const HitType hitType(LArClusterHelper::GetClusterHitType(pCluster2D));

            if ((TPC_VIEW_U != hitType) && (TPC_VIEW_V != hitType) && (TPC_VIEW_W != hitType))
                throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

            CaloHitList &targetList((TPC_VIEW_U == hitType) ? slice.m_caloHitListU : (TPC_VIEW_V == hitType) ? slice.m_caloHitListV : slice.m_caloHitListW);

            pCluster2D->GetOrderedCaloHitList().FillCaloHitList(targetList);
            targetList.insert(targetList.end(), pCluster2D->GetIsolatedCaloHitList().begin(), pCluster2D->GetIsolatedCaloHitList().end());

            if (!assignedClusters.insert(pCluster2D).second)
                throw StatusCodeException(STATUS_CODE_ALREADY_PRESENT);
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetRemainingClusters(const Algorithm *const pAlgorithm, const HitTypeToNameMap &clusterListNames,
    const ClusterSet &assignedClusters, ClusterList &remainingClusters) const
{
    this->GetRemainingClusters(pAlgorithm, clusterListNames.at(TPC_VIEW_U), assignedClusters, remainingClusters);
    this->GetRemainingClusters(pAlgorithm, clusterListNames.at(TPC_VIEW_V), assignedClusters, remainingClusters);
    this->GetRemainingClusters(pAlgorithm, clusterListNames.at(TPC_VIEW_W), assignedClusters, remainingClusters);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetRemainingClusters(const Algorithm *const pAlgorithm, const std::string &clusterListName,
    const ClusterSet &assignedClusters, ClusterList &remainingClusters) const
{
    const ClusterList *pClusterList(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*pAlgorithm, clusterListName, pClusterList));

    if (!pClusterList || pClusterList->empty())
    {
        if (PandoraContentApi::GetSettings(*pAlgorithm)->ShouldDisplayAlgorithmInfo())
            std::cout << "EventSlicingTool: unable to find cluster list " << clusterListName << std::endl;

        return;
    }

    for (const Cluster *const pCluster2D : *pClusterList)
    {
        const HitType hitType(LArClusterHelper::GetClusterHitType(pCluster2D));

        if ((TPC_VIEW_U != hitType) && (TPC_VIEW_V != hitType) && (TPC_VIEW_W != hitType))
            throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

        if (assignedClusters.count(pCluster2D))
            continue;

        if (remainingClusters.end() != std::find(remainingClusters.begin(), remainingClusters.end(), pCluster2D))
            throw StatusCodeException(STATUS_CODE_ALREADY_PRESENT);

        remainingClusters.push_back(pCluster2D);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::AssignRemainingHitsToSlices(const ClusterList &remainingClusters, const ClusterToSliceIndexMap &clusterToSliceIndexMap,
    SliceList &sliceList) const
{
    PointToSliceIndexMap pointToSliceIndexMap;

    try
    {
        PointList pointsU, pointsV, pointsW;
        this->GetKDTreeEntries2D(sliceList, pointsU, pointsV, pointsW, pointToSliceIndexMap);

        if (m_use3DProjectionsInHitPickUp)
            this->GetKDTreeEntries3D(clusterToSliceIndexMap, pointsU, pointsV, pointsW, pointToSliceIndexMap);

        pointsU.sort(EventSlicingTool::SortPoints);
        pointsV.sort(EventSlicingTool::SortPoints);
        pointsW.sort(EventSlicingTool::SortPoints);

        PointKDNode2DList kDNode2DListU, kDNode2DListV, kDNode2DListW;
        KDTreeBox boundingRegionU = fill_and_bound_2d_kd_tree(pointsU, kDNode2DListU);
        KDTreeBox boundingRegionV = fill_and_bound_2d_kd_tree(pointsV, kDNode2DListV);
        KDTreeBox boundingRegionW = fill_and_bound_2d_kd_tree(pointsW, kDNode2DListW);

        PointKDTree2D kdTreeU, kdTreeV, kdTreeW;
        kdTreeU.build(kDNode2DListU, boundingRegionU);
        kdTreeV.build(kDNode2DListV, boundingRegionV);
        kdTreeW.build(kDNode2DListW, boundingRegionW);

        ClusterVector sortedRemainingClusters(remainingClusters.begin(), remainingClusters.end());
        std::sort(sortedRemainingClusters.begin(), sortedRemainingClusters.end(), LArClusterHelper::SortByNHits);

        for (const Cluster *const pCluster2D : sortedRemainingClusters)
        {
            const HitType hitType(LArClusterHelper::GetClusterHitType(pCluster2D));

            if ((TPC_VIEW_U != hitType) && (TPC_VIEW_V != hitType) && (TPC_VIEW_W != hitType))
                throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

            PointKDTree2D &kdTree((TPC_VIEW_U == hitType) ? kdTreeU : (TPC_VIEW_V == hitType) ? kdTreeV : kdTreeW);
            const PointKDNode2D *pBestResultPoint(this->MatchClusterToSlice(pCluster2D, kdTree));

            if (!pBestResultPoint)
                continue;

            Slice &slice(sliceList.at(pointToSliceIndexMap.at(pBestResultPoint->data)));
            CaloHitList &targetList((TPC_VIEW_U == hitType) ? slice.m_caloHitListU : (TPC_VIEW_V == hitType) ? slice.m_caloHitListV : slice.m_caloHitListW);

            pCluster2D->GetOrderedCaloHitList().FillCaloHitList(targetList);
            targetList.insert(targetList.end(), pCluster2D->GetIsolatedCaloHitList().begin(), pCluster2D->GetIsolatedCaloHitList().end());
        }
    }
    catch (...)
    {
        //std::cout << "EventSlicingTool::AssignRemainingHitsToSlices - exception " << std::endl;
        for (const auto &pointMap : pointToSliceIndexMap) delete pointMap.first;
        throw;
    }

    for (const auto &pointMap : pointToSliceIndexMap) delete pointMap.first;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetKDTreeEntries2D(const SliceList &sliceList, PointList &pointsU, PointList &pointsV,
    PointList &pointsW, PointToSliceIndexMap &pointToSliceIndexMap) const
{
    unsigned int sliceIndex(0);

    for (const Slice &slice : sliceList)
    {
        for (const CaloHit *const pCaloHit : slice.m_caloHitListU)
        {
            const CartesianVector *const pPoint(new CartesianVector(pCaloHit->GetPositionVector()));
            pointToSliceIndexMap.insert(PointToSliceIndexMap::value_type(pPoint, sliceIndex));
            pointsU.push_back(pPoint);
        }

        for (const CaloHit *const pCaloHit : slice.m_caloHitListV)
        {
            const CartesianVector *const pPoint(new CartesianVector(pCaloHit->GetPositionVector()));
            pointToSliceIndexMap.insert(PointToSliceIndexMap::value_type(pPoint, sliceIndex));
            pointsV.push_back(pPoint);
        }

        for (const CaloHit *const pCaloHit : slice.m_caloHitListW)
        {
            const CartesianVector *const pPoint(new CartesianVector(pCaloHit->GetPositionVector()));
            pointToSliceIndexMap.insert(PointToSliceIndexMap::value_type(pPoint, sliceIndex));
            pointsW.push_back(pPoint);
        }

        ++sliceIndex;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void EventSlicingTool::GetKDTreeEntries3D(const ClusterToSliceIndexMap &clusterToSliceIndexMap, PointList &pointsU, PointList &pointsV,
    PointList &pointsW, PointToSliceIndexMap &pointToSliceIndexMap) const
{
    ClusterList clusterList;
    for (const auto &mapEntry : clusterToSliceIndexMap) clusterList.push_back(mapEntry.first);
    clusterList.sort(LArClusterHelper::SortByNHits);

    for (const Cluster *const pCluster3D : clusterList)
    {
        const unsigned int sliceIndex(clusterToSliceIndexMap.at(pCluster3D));

        CaloHitList caloHitList;
        pCluster3D->GetOrderedCaloHitList().FillCaloHitList(caloHitList);

        for (const CaloHit *const pCaloHit3D : caloHitList)
        {
            if (TPC_3D != pCaloHit3D->GetHitType())
                throw StatusCodeException(STATUS_CODE_FAILURE);

            const CartesianVector &position3D(pCaloHit3D->GetPositionVector());

            const CartesianVector *const pProjectionU(new CartesianVector(LArGeometryHelper::ProjectPosition(this->GetPandora(), position3D, TPC_VIEW_U)));
            const CartesianVector *const pProjectionV(new CartesianVector(LArGeometryHelper::ProjectPosition(this->GetPandora(), position3D, TPC_VIEW_V)));
            const CartesianVector *const pProjectionW(new CartesianVector(LArGeometryHelper::ProjectPosition(this->GetPandora(), position3D, TPC_VIEW_W)));

            pointsU.push_back(pProjectionU);
            pointsV.push_back(pProjectionV);
            pointsW.push_back(pProjectionW);

            pointToSliceIndexMap.insert(PointToSliceIndexMap::value_type(pProjectionU, sliceIndex));
            pointToSliceIndexMap.insert(PointToSliceIndexMap::value_type(pProjectionV, sliceIndex));
            pointToSliceIndexMap.insert(PointToSliceIndexMap::value_type(pProjectionW, sliceIndex));
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

const EventSlicingTool::PointKDNode2D *EventSlicingTool::MatchClusterToSlice(const Cluster *const pCluster2D, PointKDTree2D &kdTree) const
{
    PointList clusterPointList;
    const PointKDNode2D *pBestResultPoint(nullptr);

    try
    {
        clusterPointList.push_back(new CartesianVector(pCluster2D->GetCentroid(pCluster2D->GetInnerPseudoLayer())));
        clusterPointList.push_back(new CartesianVector(pCluster2D->GetCentroid(pCluster2D->GetOuterPseudoLayer())));
        clusterPointList.push_back(new CartesianVector((pCluster2D->GetCentroid(pCluster2D->GetInnerPseudoLayer()) + pCluster2D->GetCentroid(pCluster2D->GetOuterPseudoLayer())) * 0.5f));

        float bestDistance(std::numeric_limits<float>::max());

        for (const CartesianVector *const pClusterPoint : clusterPointList)
        {
            const PointKDNode2D *pResultPoint(nullptr);
            float resultDistance(std::numeric_limits<float>::max());
            const PointKDNode2D targetPoint(pClusterPoint, pClusterPoint->GetX(), pClusterPoint->GetZ());
            kdTree.findNearestNeighbour(targetPoint, pResultPoint, resultDistance);

            if (pResultPoint && (resultDistance < bestDistance))
            {
                pBestResultPoint = pResultPoint;
                bestDistance = resultDistance;
            }
        }
    }
    catch (...)
    {
        //std::cout << "EventSlicingTool::MatchClusterToSlice - exception " << std::endl;
        for (const CartesianVector *const pPoint : clusterPointList) delete pPoint;
        throw;
    }

    for (const CartesianVector *const pPoint : clusterPointList) delete pPoint;

    return pBestResultPoint;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool EventSlicingTool::SortPoints(const CartesianVector *const pLhs, const CartesianVector *const pRhs)
{
    const CartesianVector deltaPosition(*pRhs - *pLhs);

    if (std::fabs(deltaPosition.GetZ()) > std::numeric_limits<float>::epsilon())
        return (deltaPosition.GetZ() > std::numeric_limits<float>::epsilon());

    if (std::fabs(deltaPosition.GetX()) > std::numeric_limits<float>::epsilon())
        return (deltaPosition.GetX() > std::numeric_limits<float>::epsilon());

    return (deltaPosition.GetY() > std::numeric_limits<float>::epsilon());
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode EventSlicingTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle,
        "TrackPfoListName", m_trackPfoListName));

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle,
        "ShowerPfoListName", m_showerPfoListName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinHitsPer3DCluster", m_minHitsPer3DCluster));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "Min3DHitsToSeedNewSlice", m_min3DHitsToSeedNewSlice));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SlidingFitHalfWindow", m_halfWindowLayers));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UsePointingAssociation", m_usePointingAssociation));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MinVertexLongitudinalDistance", m_minVertexLongitudinalDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxVertexLongitudinalDistance", m_maxVertexLongitudinalDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxVertexTransverseDistance", m_maxVertexTransverseDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "VertexAngularAllowance", m_vertexAngularAllowance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxClosestApproach", m_maxClosestApproach));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxInterceptDistance", m_maxInterceptDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UseProximityAssociation", m_useProximityAssociation));

    float maxHitSeparation = std::sqrt(m_maxHitSeparationSquared);
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxHitSeparation", maxHitSeparation));
    m_maxHitSeparationSquared = maxHitSeparation * maxHitSeparation;

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UseShowerConeAssociation", m_useShowerConeAssociation));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "NConeFitLayers", m_nConeFitLayers));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "NConeFits", m_nConeFits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ConeLengthMultiplier", m_coneLengthMultiplier));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxConeLength", m_maxConeLength));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ConeTanHalfAngle1", m_coneTanHalfAngle1));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ConeBoundedFraction1", m_coneBoundedFraction1));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ConeTanHalfAngle2", m_coneTanHalfAngle2));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "ConeBoundedFraction2", m_coneBoundedFraction2));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "Use3DProjectionsInHitPickUp", m_use3DProjectionsInHitPickUp));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UseSliceSplit", m_useSliceSplit));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SliceSplitTrackMaxDistance", m_sliceSplitTrackMaxDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SliceSplitShowerMaxDistance", m_sliceSplitShowerMaxDistance));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SliceSplitVertexMaxDistance", m_sliceSplitVertexMaxDistance));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
