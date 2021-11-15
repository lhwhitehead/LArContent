/**
 *  @file   larpandoradlcontent/LArHelpers/LArDLHelper.h
 *
 *  @brief  Header file for the lar deep learning helper helper class.
 *
 *  $Log: $
 */
#ifndef LAR_DL_HELPER_H
#define LAR_DL_HELPER_H 1

#include <torch/script.h>
#include <torch/torch.h>

#include "Pandora/StatusCodes.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

namespace pandora
{
class Cluster;
}

namespace lar_dl_content
{

/**
 *  @brief  LArDLHelper class
 */
class LArDLHelper
{
public:
    typedef torch::jit::script::Module TorchModel;
    typedef torch::Tensor TorchInput;
    typedef std::vector<torch::jit::IValue> TorchInputVector;
    typedef at::Tensor TorchOutput;

    /**
     *  @brief  Loads a deep learning model
     *
     *  @param  filename the filename of the model to load
     *  @param  model the TorchModel in which to store the loaded model
     *
     *  @return STATUS_CODE_SUCCESS upon successful loading of the model. STATUS_CODE_FAILURE otherwise.
     */
    static pandora::StatusCode LoadModel(const std::string &filename, TorchModel &model);

    /**
     *  @brief  Create a torch input tensor
     *
     *  @param  dimensions the size of each dimension of the tensor: pass as {a, b, c, d} for example
     *  @param  tensor the tensor to be initialised
     */
    static void InitialiseInput(const at::IntArrayRef dimensions, TorchInput &tensor);

    /**
     *  @brief  Run a deep learning model
     *
     *  @param  model the model to run
     *  @param  input the input to run over
     *  @param  output the tensor to store the output in
     */
    static void Forward(TorchModel &model, const TorchInputVector &input, TorchOutput &output);

    /**
     *  @brief  Get the track likelihood for a CaloHit
     *
     *  @param  caloHit input CaloHit
     *
     *  @return track likelihood score
     */
    static float GetTrackLikelihood(const pandora::CaloHit *const caloHit);

    /**
     *  @brief  Get the mean track likelihood for a CaloHitList
     *
     *  @param  caloHits input CaloHitList
     *
     *  @return mean track likelihood score
     */
    static float GetMeanTrackLikelihood(const pandora::CaloHitList &caloHits); 

    /**
     *  @brief  Get the mean track likelihood for a Cluster
     *
     *  @param  pCluster input Cluster object 
     *
     *  @return mean track likelihood score
     */
    static float GetMeanTrackLikelihood(const pandora::Cluster *const pCluster);
};

} // namespace lar_dl_content

#endif // #ifndef LAR_DL_HELPER_H
