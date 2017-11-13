/**
 *  @file   larpandoracontent/LArUtility/SlicingAlgorithm.cc
 *
 *  @brief  Implementation of the slicing algorithm class.
 *
 *  $Log: $
 */

#include "Api/PandoraApi.h"

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArUtility/SlicingAlgorithm.h"

using namespace pandora;

namespace lar_content
{

SlicingAlgorithm::SlicingAlgorithm() :
    m_pEventSlicingTool(nullptr)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SlicingAlgorithm::Run()
{
    SliceList sliceList;
    m_pEventSlicingTool->RunSlicing(this, m_caloHitListNames, m_clusterListNames, sliceList);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::RunDaughterAlgorithm(*this, m_slicingListDeletionAlgorithm));

    if (sliceList.empty())
        return STATUS_CODE_SUCCESS;

    std::string clusterListName;
    const ClusterList *pClusterList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pClusterList, clusterListName));

    std::string pfoListName;
    const PfoList *pPfoList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pPfoList, pfoListName));

    for (const Slice &slice : sliceList)
    {
        const Cluster *pClusterU(nullptr), *pClusterV(nullptr), *pClusterW(nullptr);
        PandoraContentApi::Cluster::Parameters clusterParametersU, clusterParametersV, clusterParametersW;
        clusterParametersU.m_caloHitList = slice.m_caloHitListU;
        clusterParametersV.m_caloHitList = slice.m_caloHitListV;
        clusterParametersW.m_caloHitList = slice.m_caloHitListW;
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, clusterParametersU, pClusterU));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, clusterParametersV, pClusterV));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, clusterParametersW, pClusterW));

        const Pfo *pSlicePfo(nullptr);
        PandoraContentApi::ParticleFlowObject::Parameters pfoParameters;
        pfoParameters.m_clusterList.push_back(pClusterU);
        pfoParameters.m_clusterList.push_back(pClusterV);
        pfoParameters.m_clusterList.push_back(pClusterW);
        pfoParameters.m_charge = 0;
        pfoParameters.m_energy = 0.f;
        pfoParameters.m_mass = 0.f;
        pfoParameters.m_momentum = CartesianVector(0.f, 0.f, 0.f);
        pfoParameters.m_particleId = 0;
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ParticleFlowObject::Create(*this, pfoParameters, pSlicePfo));
    }

    if (!pClusterList->empty())
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_sliceClusterListName));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Cluster>(*this, m_sliceClusterListName));
    }

    if (!pPfoList->empty())
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<ParticleFlowObject>(*this, m_slicePfoListName));
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<ParticleFlowObject>(*this, m_slicePfoListName));
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SlicingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    AlgorithmTool *pAlgorithmTool(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ProcessAlgorithmTool(*this, xmlHandle, "Slicing", pAlgorithmTool));
    m_pEventSlicingTool = dynamic_cast<EventSlicingBaseTool*>(pAlgorithmTool);

    if (!m_pEventSlicingTool)
        return STATUS_CODE_INVALID_PARAMETER;

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ProcessAlgorithm(*this, xmlHandle, "SlicingListDeletion", m_slicingListDeletionAlgorithm));

    std::string caloHitListNameU, caloHitListNameV, caloHitListNameW;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputCaloHitListNameU", caloHitListNameU));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputCaloHitListNameV", caloHitListNameV));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputCaloHitListNameW", caloHitListNameW));
    m_caloHitListNames[TPC_VIEW_U] = caloHitListNameU;
    m_caloHitListNames[TPC_VIEW_V] = caloHitListNameV;
    m_caloHitListNames[TPC_VIEW_W] = caloHitListNameW;

    std::string clusterListNameU, clusterListNameV, clusterListNameW;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputClusterListNameU", clusterListNameU));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputClusterListNameV", clusterListNameV));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputClusterListNameW", clusterListNameW));
    m_clusterListNames[TPC_VIEW_U] = clusterListNameU;
    m_clusterListNames[TPC_VIEW_V] = clusterListNameV;
    m_clusterListNames[TPC_VIEW_W] = clusterListNameW;

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputClusterListName", m_sliceClusterListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputPfoListName", m_slicePfoListName));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
