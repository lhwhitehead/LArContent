/**
 *  @file   larpandoradlcontent/LArVertexing/DlVertexingAlgorithm.h
 *
 *  @brief  Header file for the deep learning vertexing algorithm.
 *
 *  $Log: $
 */
#ifndef LAR_DL_VERTEXING_ALGORITHM_H
#define LAR_DL_VERTEXING_ALGORITHM_H 1

#include "Pandora/Algorithm.h"
#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

using namespace lar_content;

namespace lar_dl_content
{
/**
 *  @brief  DeepLearningTrackShowerIdAlgorithm class
 */
class DlVertexingAlgorithm: public pandora::Algorithm
{
public:
    typedef std::map<std::pair<int, int>, std::vector<const pandora::CaloHit*>> PixelToCaloHitsMap;

    /**
     *  @brief Default constructor
     */
    DlVertexingAlgorithm();

    virtual ~DlVertexingAlgorithm();

private:
    class CaloHitTuple
    {
    public:
        CaloHitTuple(const pandora::Pandora &pandora, const pandora::CaloHit *pCaloHitU, const pandora::CaloHit *pCaloHitV, const pandora::CaloHit *pCaloHitW);

        const pandora::CartesianVector &GetPosition() const;
        float GetChi2() const;
        std::string ToString() const;
        const pandora::CaloHit *GetCaloHitU() const;
        const pandora::CaloHit *GetCaloHitV() const;
        const pandora::CaloHit *GetCaloHitW() const;

    private:
        const pandora::CaloHit *m_pCaloHitU;    ///< The hit in the U view
        const pandora::CaloHit *m_pCaloHitV;    ///< The hit in the V view
        const pandora::CaloHit *m_pCaloHitW;    ///< The hit in the W view
        pandora::CartesianVector m_pos;         ///< Calculated 3D position
        float                   m_chi2;         ///< Chi squared of calculated position
    };

    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
    pandora::StatusCode Train();
    pandora::StatusCode Infer();

    /*
     *  @brief  Retrieve the map from MC to calo hits for reconstructable particles
     *
     *  @param  mcToHitsMap The map to populate
     *
     *  @return The StatusCode resulting from the function
     **/
    pandora::StatusCode GetMCToHitsMap(LArMCParticleHelper::MCContributionMap &mcToHitsMap) const;

    /*
     *  @brief  Construct a list of the MC particles from the MC to calo hits map, completing the interaction hierarchy with the invisible
     *          upstream particles.
     *
     *  @param  mcToHitsMap The map of reconstructible MC particles to calo hits
     *  @param  mcHierarchy The output list of MC particles representing the interaction
     *
     *  @return The StatusCode resulting from the function
     **/
    pandora::StatusCode CompleteMCHierarchy(const LArMCParticleHelper::MCContributionMap &mcToHitsMap, pandora::MCParticleList& mcHierarchy)
        const;

    /*
     *  @brief  Determine the physical bounds associated with a CaloHitList.
     *
     *  @param  caloHitList The CaloHitList for which bounds should be determined
     *  @param  xMin The output minimum x value
     *  @param  xMax The output maximum x value
     *  @param  zMin The output minimum z value
     *  @param  zMax The output maximum z value
     *
     *  @return The StatusCode resulting from the function
     **/
    void GetHitRegion(const pandora::CaloHitList& caloHitList, float& xMin, float& xMax, float& zMin, float& zMax) const;

    /**
     *  @brief Create a vertex list from the candidate vertices.
     *
     *  @param  candidates The candidate positions with which to create the list.
     *
     *  @return The StatusCode resulting from the function
     */
    pandora::StatusCode MakeCandidateVertexList(const pandora::CartesianPointVector &positions);

    bool                    m_trainingMode;         ///< Training mode
    std::string             m_trainingOutputFile;   ///< Output file name for training examples
    std::string             m_outputVertexListName; ///< Output vertex list name
    pandora::StringVector   m_caloHitListNames;     ///< Names of input calo hit lists
    LArDLHelper::TorchModel m_modelU;               ///< The model for the U view
    LArDLHelper::TorchModel m_modelV;               ///< The model for the V view
    LArDLHelper::TorchModel m_modelW;               ///< The model for the W view
    float                   m_pixelShift;           ///< Pixel normalisation shift
    float                   m_pixelScale;           ///< Pixel normalisation scale
    bool                    m_visualise;            ///< Whether or not to visualise the candidate vertices
};

} // namespace lar_dl_content

#endif // LAR_DL_VERTEXING_ALGORITHM_H

