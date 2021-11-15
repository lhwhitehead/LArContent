/**
 *  @file   larpandoradlcontent/LArHelpers/LArDLHelper.cc
 *
 *  @brief  Implementation of the lar deep learning helper helper class.
 *
 *  $Log: $
 */

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

#include "Objects/Cluster.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

StatusCode LArDLHelper::LoadModel(const std::string &filename, LArDLHelper::TorchModel &model)
{
    try
    {
        model = torch::jit::load(filename);
        std::cout << "Loaded the TorchScript model \'" << filename << "\'" << std::endl;
    }
    catch (...)
    {
        std::cout << "Error loading the TorchScript model \'" << filename << "\'" << std::endl;
        return STATUS_CODE_FAILURE;
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::InitialiseInput(const at::IntArrayRef dimensions, TorchInput &tensor)
{
    tensor = torch::zeros(dimensions);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::Forward(TorchModel &model, const TorchInputVector &input, TorchOutput &output)
{
    output = model.forward(input).toTensor();
}

//------------------------------------------------------------------------------------------------------------------------------------------
float LArDLHelper::GetTrackLikelihood(const CaloHit *const pCaloHit)
{
    float trackLikelihood{0.f};
    try{
        const LArCaloHit *pLArCaloHit{dynamic_cast<const LArCaloHit *>(pCaloHit)};
        const float pTrack{pLArCaloHit->GetTrackProbability()};
        const float pShower{pLArCaloHit->GetShowerProbability()};
        if ((pTrack + pShower) > std::numeric_limits<float>::epsilon())
            trackLikelihood = pTrack / (pTrack + pShower);
    }
    catch (const StatusCodeException &)
    {
    }
    return trackLikelihood;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float LArDLHelper::GetMeanTrackLikelihood(const CaloHitList &caloHits)
{
    FloatVector trackLikelihoods;
    try
    {
        for (const CaloHit *pCaloHit : caloHits)
            trackLikelihoods.emplace_back(LArDLHelper::GetTrackLikelihood(pCaloHit));
    }
    catch (const StatusCodeException &)
    {
    }
    
    const unsigned long N{trackLikelihoods.size()};
    const float meanTrackLikelihood{N > 0 ? std::accumulate(std::begin(trackLikelihoods), std::end(trackLikelihoods), 0.f) / N : 0.f};
    return meanTrackLikelihood;
}

//------------------------------------------------------------------------------------------------------------------------------------------

float LArDLHelper::GetMeanTrackLikelihood(const Cluster *const pCluster)
{
    const OrderedCaloHitList &orderedCaloHitList{pCluster->GetOrderedCaloHitList()};
    CaloHitList caloHits;
    orderedCaloHitList.FillCaloHitList(caloHits);
    const CaloHitList &isolatedHits{pCluster->GetIsolatedCaloHitList()};
    caloHits.insert(caloHits.end(), isolatedHits.begin(), isolatedHits.end());

    return LArDLHelper::GetMeanTrackLikelihood(caloHits);
}

} // namespace lar_dl_content
