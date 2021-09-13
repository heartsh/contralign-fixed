//////////////////////////////////////////////////////////////////////
// ComputationEngine.cpp
//////////////////////////////////////////////////////////////////////

#include "ComputationEngine.hpp"

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputationEngine()
// ComputationEngine::~ComputationEngine()
//
// Constructor and destructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
ComputationEngine<RealT>::ComputationEngine(const Options &options,
                                            const std::vector<FileDescription> &descriptions,
                                            InferenceEngine<RealT> &inference_engine,
                                            ParameterManager<RealT> &parameter_manager) :
    DistributedComputation<RealT, SharedInfo<RealT>, NonSharedInfo>(options.GetBoolValue("verbose_output")),
    options(options),
    descriptions(descriptions),
    inference_engine(inference_engine),
    parameter_manager(parameter_manager)
{}

template<class RealT>
ComputationEngine<RealT>::~ComputationEngine()
{}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::DoComputation()
//
// Decide what type of computation needs to be done and then
// pass the work on to the appropriate routine.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::DoComputation(std::vector<RealT> &result, 
                                             const SharedInfo<RealT> &shared,
                                             const NonSharedInfo &nonshared)
{
    switch (shared.command)
    {
        case CHECK_PARSABILITY:
            CheckParsability(result, nonshared);
            break;
        case COMPUTE_SOLUTION_NORM_BOUND:
            ComputeSolutionNormBound(result, shared, nonshared);
            break;
        case COMPUTE_GRADIENT_NORM_BOUND:
            ComputeGradientNormBound(result, nonshared);
            break;
        case COMPUTE_LOSS:
            ComputeLoss(result, shared, nonshared);
            break;    
        case COMPUTE_FUNCTION:
            ComputeFunctionAndGradient(result, shared, nonshared, false);
            break;    
        case COMPUTE_GRADIENT:
            ComputeFunctionAndGradient(result, shared, nonshared, true);
            break;    
        case COMPUTE_HV:
            ComputeHessianVectorProduct(result, shared, nonshared);
            break;    
        case PREDICT:
            Predict(result, shared, nonshared);
            break;
        default: 
            Assert(false, "Unknown command type.");
            break;
    }
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::CheckParsability()
//
// Check to see if an alignment is parsable or not.  Return a
// vector with a "0" in the appropriate spot indicating that a
// file is not parsable.
//////////////////////////////////////////////////////////////////////

template <class RealT>
void ComputationEngine<RealT>::CheckParsability(std::vector<RealT> &result, 
                                                const NonSharedInfo &nonshared)
{
    // load training example
    const MultiSequence &multi_seqs = descriptions[nonshared.index].seqs;
    std::vector<int> indices(2);
    bool parsable = true;
    
    for (indices[0] = 0; parsable && indices[0] < multi_seqs.GetNumSequences(); indices[0]++)
    {
        for (indices[1] = indices[0]+1; parsable && indices[1] < multi_seqs.GetNumSequences(); indices[1]++)
        {
            MultiSequence seqs(multi_seqs, indices);
            inference_engine.LoadSequences(seqs);

            // conditional inference
            inference_engine.LoadValues(std::vector<RealT>(parameter_manager.GetNumLogicalParameters()));
            inference_engine.UseConstraints(seqs.GetAlignedTo());
            inference_engine.ComputeViterbi();
            RealT conditional_score = inference_engine.GetViterbiScore();

            // check for bad parse
            if (conditional_score < RealT(NEG_INF/2)) parsable = false;
        }
    }

    // check for bad parse
    result.clear();
    result.resize(descriptions.size());
    result[nonshared.index] = parsable ? 1 : 0;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeSolutionNormBound()
//
// Compute the max entropy and loss possible for an example.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeSolutionNormBound(std::vector<RealT> &result, 
                                                        const SharedInfo<RealT> &shared,
                                                        const NonSharedInfo &nonshared)
{
    RealT max_entropy = RealT(0);
    RealT max_loss = RealT(0);
    
    // load training example
    const MultiSequence &multi_seqs = descriptions[nonshared.index].seqs;
    std::vector<int> indices(2);

    for (indices[0] = 0; indices[0] < multi_seqs.GetNumSequences(); indices[0]++)
    {
        for (indices[1] = indices[0]+1; indices[1] < multi_seqs.GetNumSequences(); indices[1]++)
        {
            MultiSequence seqs(multi_seqs, indices);
            inference_engine.LoadSequences(seqs);

            // load parameters
            const std::vector<RealT> w(parameter_manager.GetNumLogicalParameters(), RealT(0));
            inference_engine.LoadValues(w);

            // perform computation
#if !SMOOTH_MAX_MARGIN
            if (!options.GetBoolValue("viterbi_parsing"))
#endif
            {
                inference_engine.ComputeForward();
                max_entropy += inference_engine.ComputeLogPartitionCoefficient() * RealT(multi_seqs.GetPairWeight(indices[0], indices[1]));
            }
            
#if defined(HAMMING_LOSS)
            inference_engine.UseLoss(seqs.GetAlignedTo(), RealT(HAMMING_LOSS));
            inference_engine.ComputeViterbi();
            max_loss += inference_engine.GetViterbiScore() * RealT(multi_seqs.GetPairWeight(indices[0], indices[1]));
#endif
        }
    }
            
    result.clear();
    result.resize(descriptions.size());
    result[nonshared.index] = max_entropy / shared.log_base + max_loss;
    
    result *= RealT(descriptions[nonshared.index].weight);
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeGradientNormBound()
//
// Compute the max L1 norm for the features of an example.
// Return a vector with this value in the appropriate spot for
// this example.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeGradientNormBound(std::vector<RealT> &result,
                                                        const NonSharedInfo &nonshared)
{
    RealT max_L1_norm = RealT(0);
    
    // load training example
    const MultiSequence &multi_seqs = descriptions[nonshared.index].seqs;
    std::vector<int> indices(2);

    for (indices[0] = 0; indices[0] < multi_seqs.GetNumSequences(); indices[0]++)
    {
        for (indices[1] = indices[0]+1; indices[1] < multi_seqs.GetNumSequences(); indices[1]++)
        {
            MultiSequence seqs(multi_seqs, indices);
            inference_engine.LoadSequences(seqs);

            // load parameters
            const std::vector<RealT> w(parameter_manager.GetNumLogicalParameters(), RealT(0));
            inference_engine.LoadValues(w);

            // perform computation
            inference_engine.ComputeViterbi();
            max_L1_norm = std::max(max_L1_norm, inference_engine.GetViterbiScore());
        }
    }

    result.clear();
    result.resize(descriptions.size());
    result[nonshared.index] = max_L1_norm;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeLoss()
//
// Return a vector containing a single entry with the loss value.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeLoss(std::vector<RealT> &result, 
                                           const SharedInfo<RealT> &shared,
                                           const NonSharedInfo &nonshared)
{
    result.clear();
    result.resize(1);

    // load training example
    const MultiSequence &multi_seqs = descriptions[nonshared.index].seqs;
    std::vector<int> indices(2);

    for (indices[0] = 0; indices[0] < multi_seqs.GetNumSequences(); indices[0]++)
    {
        for (indices[1] = indices[0]+1; indices[1] < multi_seqs.GetNumSequences(); indices[1]++)
        {
            MultiSequence seqs(multi_seqs, indices);
            inference_engine.LoadSequences(seqs);
            
            // load parameters
            std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
            inference_engine.LoadValues(w * shared.log_base);

            // perform computation
            std::string edit_string;
            if (options.GetBoolValue("viterbi_parsing"))
            {
                inference_engine.ComputeViterbi();
                edit_string = inference_engine.PredictAlignmentViterbi();
            }
            else
            {
                inference_engine.ComputeForward();
                inference_engine.ComputeBackward();
                inference_engine.ComputePosterior();
                edit_string = inference_engine.PredictAlignmentPosterior(shared.gamma);
            }
            
            MultiSequence alignment;
            alignment.AddSequence(new Sequence(Sequence(seqs.GetSequence(0), Sequence::COMPRESS_GAPS), Sequence::INSERT_GAPS, edit_string, 'X'));
            alignment.AddSequence(new Sequence(Sequence(seqs.GetSequence(1), Sequence::COMPRESS_GAPS), Sequence::INSERT_GAPS, edit_string, 'Y'));
            
            // compute loss
            if (!shared.use_loss) Error("Must be using loss function in order to compute loss.");
            std::fill(w.begin(), w.end(), RealT(0));
            inference_engine.LoadValues(w);
#if defined(HAMMING_LOSS)
            inference_engine.UseLoss(seqs.GetAlignedTo(), shared.log_base * RealT(HAMMING_LOSS));
#endif                                                          
            inference_engine.UseConstraints(alignment.GetAlignedTo());
            
            inference_engine.ComputeViterbi();
            result.back() += inference_engine.GetViterbiScore() * RealT(multi_seqs.GetPairWeight(indices[0], indices[1]));
        }
    }
    
    result *= RealT(descriptions[nonshared.index].weight);
    result.back() /= shared.log_base;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeFunctionAndGradient();
//
// Return a vector containing the gradient and function value.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeFunctionAndGradient(std::vector<RealT> &result, 
                                                          const SharedInfo<RealT> &shared,
                                                          const NonSharedInfo &nonshared,
                                                          bool need_gradient)
{
    result.clear();
    result.resize(need_gradient ? parameter_manager.GetNumLogicalParameters() + 1 : 1);

    // load training example
    const MultiSequence &multi_seqs = descriptions[nonshared.index].seqs;
    std::vector<int> indices(2);

    for (indices[0] = 0; indices[0] < multi_seqs.GetNumSequences(); indices[0]++)
    {
        for (indices[1] = indices[0]+1; indices[1] < multi_seqs.GetNumSequences(); indices[1]++)
        {
            std::vector<RealT> partial_result;
            MultiSequence seqs(multi_seqs, indices);
            inference_engine.LoadSequences(seqs);

            // load parameters
            const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
            inference_engine.LoadValues(w * shared.log_base);

#if defined(HAMMING_LOSS)
            if (shared.use_loss) inference_engine.UseLoss(seqs.GetAlignedTo(), shared.log_base * RealT(HAMMING_LOSS));
#endif
            
            // unconditional inference
            RealT unconditional_score;
            std::vector<RealT> unconditional_counts;
            
            if (shared.use_nonsmooth)
            {
                inference_engine.ComputeViterbi();
                unconditional_score = inference_engine.GetViterbiScore();
                if (need_gradient) unconditional_counts = inference_engine.ComputeViterbiFeatureCounts();
            }
            else
            {
                inference_engine.ComputeForward();
                unconditional_score = inference_engine.ComputeLogPartitionCoefficient();
                if (need_gradient)
                {
                    inference_engine.ComputeBackward();
                    unconditional_counts = inference_engine.ComputeFeatureCountExpectations();
                }
            }
            
            // conditional inference
            RealT conditional_score;
            std::vector<RealT> conditional_counts;

            inference_engine.UseConstraints(seqs.GetAlignedTo());
            if (shared.use_nonsmooth)
            {
                inference_engine.ComputeViterbi();
                conditional_score = inference_engine.GetViterbiScore();
                if (need_gradient) conditional_counts = inference_engine.ComputeViterbiFeatureCounts();
            }
            else
            {
                inference_engine.ComputeForward();
                conditional_score = inference_engine.ComputeLogPartitionCoefficient();
                if (need_gradient)
                {
                    inference_engine.ComputeBackward();
                    conditional_counts = inference_engine.ComputeFeatureCountExpectations();
                }
            }
            
            partial_result.clear();
            
            // compute subgradient
            if (need_gradient) partial_result = unconditional_counts - conditional_counts;
            
            // compute function value
            Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
            partial_result.push_back(unconditional_score - conditional_score);
            
            // check for bad parse
            if (conditional_score < RealT(NEG_INF/2))
            {
                std::cerr << "Unexpected bad parse for file: " << descriptions[nonshared.index].input_filename << std::endl;
                fill(partial_result.begin(), partial_result.end(), RealT(0));
                return;
            }
            
            if (NONCONVEX_MULTIPLIER != 0)
            {
                
#if STOCHASTIC_GRADIENT
                if (shared.use_loss) inference_engine.UseLoss(seqs.GetAlignedTo(), RealT(0));
                
                // unconditional counts
                inference_engine.UseConstraints(std::vector<int>(seqs.GetLength() + 1, UNKNOWN));
                if (shared.use_nonsmooth)
                {
                    inference_engine.ComputeViterbi();
                    unconditional_score = inference_engine.GetViterbiScore();
                    if (need_gradient) unconditional_counts = inference_engine.ComputeViterbiFeatureCounts();
                }
                else
                {
                    inference_engine.ComputeForward();
                    unconditional_score = inference_engine.ComputeLogPartitionCoefficient();
                    if (need_gradient)
                    {
                        inference_engine.ComputeBackward();
                        unconditional_counts = inference_engine.ComputeFeatureCountExpectations();
                    }
                }
                
                // conditional counts
                inference_engine.UseConstraints(seqs.GetAlignedTo());
                if (shared.use_nonsmooth)
                {
                    inference_engine.ComputeViterbi();
                    unconditional_score = inference_engine.GetViterbiScore();
                    if (need_gradient) unconditional_counts = inference_engine.ComputeViterbiFeatureCounts();
                }
                else
                {
                    inference_engine.ComputeForward();
                    unconditional_score = inference_engine.ComputeLogPartitionCoefficient();
                    if (need_gradient)
                    {
                        inference_engine.ComputeBackward();
                        unconditional_counts = inference_engine.ComputeFeatureCountExpectations();
                    }
                }
                
                std::vector<RealT> partial_result2;
                
                // compute subgradient
                if (need_gradient) partial_result2 = unconditional_counts - conditional_counts;
                
                // compute function value
                Assert(conditional_score <= unconditional_score, "Conditional score cannot exceed unconditional score.");
                partial_result2.push_back(unconditional_score - conditional_score);
                
                // check for bad parse
                if (conditional_score < RealT(NEG_INF/2))
                {
                    std::cerr << "Unexpected bad parse for file: " << descriptions[nonshared.index].input_filename << std::endl;
                    fill(partial_result.begin(), partial_result.end(), 0);
                    return;
                }
                
                partial_result -= NONCONVEX_MULTIPLIER * partial_result2;
#endif
            }

            // avoid precision problems
            if (partial_result.back() < 0)
            {
                std::cerr << "Encountered negative function value for " << descriptions[nonshared.index].input_filename << ": " << partial_result.back() << std::endl;
                std::fill(partial_result.begin(), partial_result.end(), RealT(0));
                return;
            }

            
            result += partial_result * RealT(multi_seqs.GetPairWeight(indices[0], indices[1]));
        }
    }

    result *= RealT(descriptions[nonshared.index].weight);
    result.back() /= shared.log_base;
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::ComputeHessianVectorProduct()
//
// Return a vector containing Hv.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::ComputeHessianVectorProduct(std::vector<RealT> &result, 
                                                           const SharedInfo<RealT> &shared,
                                                           const NonSharedInfo &nonshared)
{
    const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
    const std::vector<RealT> v(shared.v, shared.v + parameter_manager.GetNumLogicalParameters());

    if (options.GetBoolValue("viterbi_parsing"))
    {
        Error("Should not use Hessian-vector products with Viterbi parsing.");
    }
    
    const RealT EPSILON = RealT(1e-8);
    SharedInfo<RealT> shared_temp(shared);
    std::vector<RealT> result2;

    for (size_t i = 0; i < parameter_manager.GetNumLogicalParameters(); i++)
        shared_temp.w[i] = shared.w[i] + EPSILON * v[i];
    ComputeFunctionAndGradient(result, shared_temp, nonshared, true);
    
    for (size_t i = 0; i < parameter_manager.GetNumLogicalParameters(); i++)
        shared_temp.w[i] = shared.w[i] - EPSILON * v[i];
    ComputeFunctionAndGradient(result2, shared_temp, nonshared, true);
    
    result = (result - result2) / (RealT(2) * EPSILON);
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::Predict()
//
// Predict structure of a single sequence.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::Predict(std::vector<RealT> &result, 
                                       const SharedInfo<RealT> &shared,
                                       const NonSharedInfo &nonshared)
{
    result.clear();
    
    // load sequence, with constraints if necessary
    const MultiSequence &multi_seqs = descriptions[nonshared.index].seqs;
    const int m = multi_seqs.GetNumSequences();
    std::vector<SparseMatrix<RealT> *> posteriors(m*m, static_cast<SparseMatrix<RealT> *>(NULL));

    // retrieve options
    const bool toggle_viterbi = options.GetBoolValue("viterbi_parsing");
    const bool toggle_pairwise = options.GetBoolValue("pairwise") || (m == 2);
    const bool toggle_use_suffix = options.GetBoolValue("pairwise");
    const bool toggle_partition = options.GetBoolValue("partition_function_only");
    const bool toggle_use_constraints = options.GetBoolValue("use_constraints");
    const bool toggle_annealing = options.GetBoolValue("annealing");
    const bool toggle_verbose = options.GetBoolValue("verbose_output");
    const bool toggle_posteriors = (options.GetStringValue("output_posteriors_destination") != "");
    const int probabilistic_consistency_iterations = toggle_viterbi ? 0 : options.GetIntValue("probabilistic_consistency_iterations");
    const int spectral_consistency_iterations = toggle_viterbi ? 0 : options.GetIntValue("spectral_consistency_iterations");
    const RealT posterior_cutoff = options.GetRealValue("output_posteriors_cutoff");

    // if using Viterbi, we must be producing pairwise alignments
    if (toggle_viterbi && !toggle_pairwise)
        Error("Viterbi mode not allowed for multiple sequence alignments.");

    RealT log_partition_coefficient = RealT(0);
    
    // perform all-pairs pairwise comparisons
    for (int i = 0; i < m; i++)
    {
        for (int j = i+1; j < m; j++)
        {
            // print feedback
            if (toggle_verbose)
            {
                WriteProgressMessage(SPrintF("Computing pairwise %s: (%d) %s vs (%d) %s...",
                                             (toggle_viterbi ? "alignment" : "posteriors"),
                                             i+1, multi_seqs.GetSequence(i).GetName().c_str(),
                                             j+1, multi_seqs.GetSequence(j).GetName().c_str()));
            }
            
            // load pair of sequences (and constraints if needed)
            MultiSequence seqs;
            seqs.AddSequence(new Sequence(multi_seqs.GetSequence(i)));
            seqs.AddSequence(new Sequence(multi_seqs.GetSequence(j)));

            inference_engine.LoadSequences(seqs);
            if (toggle_use_constraints) inference_engine.UseConstraints(seqs.GetAlignedTo());

            // load parameters
            const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
            inference_engine.LoadValues(w * shared.log_base);

            // perform inference (3 possibilities)

            // (1) compute partition coefficient only
            if (toggle_partition)
            {
                RealT pair_coefficient = RealT(0);
                if (toggle_viterbi)
                {
                    inference_engine.ComputeViterbi();
                    pair_coefficient = inference_engine.GetViterbiScore();
                }
                else
                {
                    inference_engine.ComputeForward();
                    pair_coefficient = inference_engine.ComputeLogPartitionCoefficient();
                }

                if (toggle_pairwise)
                {
                    std::cout << "Partition coefficient: " << pair_coefficient << std::endl;
                }
                else
                {
                    log_partition_coefficient += pair_coefficient;
                }                
            }
            
            // (2) perform alignment immediately
            else if (toggle_pairwise &&
                     probabilistic_consistency_iterations == 0 &&
                     spectral_consistency_iterations == 0)
            {
                SparseMatrix<RealT> *sparse = static_cast<SparseMatrix<RealT> *>(NULL);
                std::string edit_string;
                
                if (toggle_viterbi)
                {
                    inference_engine.ComputeViterbi();
                    edit_string = inference_engine.PredictAlignmentViterbi();
                }
                else
                {
                    inference_engine.ComputeForward();
                    inference_engine.ComputeBackward();
                    inference_engine.ComputePosterior();
                    edit_string = inference_engine.PredictAlignmentPosterior(shared.gamma);

                    // save posterior matrix if needed

                    if (toggle_posteriors)
                    {
                        RealT *posterior = inference_engine.GetPosterior(posterior_cutoff);
                        sparse = new SparseMatrix<RealT>(posterior,
                                                         seqs.GetSequence(0).GetCompressedLength()+1,
                                                         seqs.GetSequence(1).GetCompressedLength()+1,
                                                         RealT(0));
                        delete [] posterior;
                    }                    
                }

                MultiSequence alignment;
                alignment.AddSequence(new Sequence(Sequence(seqs.GetSequence(0), Sequence::COMPRESS_GAPS), Sequence::INSERT_GAPS, edit_string, 'X'));
                alignment.AddSequence(new Sequence(Sequence(seqs.GetSequence(1), Sequence::COMPRESS_GAPS), Sequence::INSERT_GAPS, edit_string, 'Y'));
                
                OutputPrediction(shared, nonshared, alignment, toggle_use_suffix ? SPrintF(".%d.%d", i+1, j+1) : std::string(""), sparse);
                
                delete sparse;
            }
            
            // (3) save posteriors for later
            else
            {
                inference_engine.ComputeForward();
                inference_engine.ComputeBackward();
                inference_engine.ComputePosterior();

                RealT *posterior = inference_engine.GetPosterior(posterior_cutoff);
                posteriors[i*m + j] = 
                    new SparseMatrix<RealT>(posterior,
                                            seqs.GetSequence(0).GetCompressedLength()+1,
                                            seqs.GetSequence(1).GetCompressedLength()+1,
                                            RealT(0));
                delete [] posterior;                
            }
        }
    }

    // clear progress messages
    if (toggle_verbose) WriteProgressMessage("");
    
    // print out partition coefficient
    if (toggle_partition && !toggle_pairwise)
    {
        std::cout << "Partition coefficient: " << log_partition_coefficient << std::endl;
        return;
    }

    // check if we're done already with needed output
    else if (toggle_pairwise &&
             probabilistic_consistency_iterations == 0 &&
             spectral_consistency_iterations == 0)
    {
        return;
    }

    // consistency transformations
    if (probabilistic_consistency_iterations > 0)
    {
        ProbabilisticConsistency<RealT> consistency(posteriors, toggle_verbose, probabilistic_consistency_iterations);
        consistency.Transform();
    }

    if (spectral_consistency_iterations > 0)
    {
        SpectralConsistency<RealT> consistency(posteriors, toggle_verbose, spectral_consistency_iterations);
        consistency.Transform();
    }
    
    // perform final alignments

    if (toggle_pairwise)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = i+1; j < m; j++)
            {
                // print feedback
                if (toggle_verbose)
                {
                    WriteProgressMessage(SPrintF("Computing pairwise alignments: (%d) %s vs (%d) %s...",
                                                 i+1, multi_seqs.GetSequence(i).GetName().c_str(),
                                                 j+1, multi_seqs.GetSequence(j).GetName().c_str()));
                }

                // load pair of sequences
                MultiSequence seqs;
                seqs.AddSequence(new Sequence(multi_seqs.GetSequence(i)));
                seqs.AddSequence(new Sequence(multi_seqs.GetSequence(j)));
                
                inference_engine.LoadSequences(seqs);
                if (toggle_use_constraints) inference_engine.UseConstraints(seqs.GetAlignedTo());
                
                // load parameters
                const std::vector<RealT> w(shared.w, shared.w + parameter_manager.GetNumLogicalParameters());
                inference_engine.LoadValues(w * shared.log_base);
            
                // use constraints if requested
                if (toggle_use_constraints) inference_engine.UseConstraints(seqs.GetAlignedTo());

                std::vector<RealT> posterior = posteriors[i*m + j]->GetUnsparse();
                std::string edit_string = inference_engine.PredictAlignmentPosterior(shared.gamma, &posterior[0]);

                MultiSequence alignment;
                alignment.AddSequence(new Sequence(Sequence(seqs.GetSequence(0), Sequence::COMPRESS_GAPS), Sequence::INSERT_GAPS, edit_string, 'X'));
                alignment.AddSequence(new Sequence(Sequence(seqs.GetSequence(1), Sequence::COMPRESS_GAPS), Sequence::INSERT_GAPS, edit_string, 'Y'));
                
                OutputPrediction(shared, nonshared, alignment, toggle_use_suffix ? SPrintF(".%d.%d", i+1, j+1) : std::string(""), posteriors[i*m + j]);
            }
        }
    }

    // multiple alignment
    else
    {
        MultiSequence alignment;

        if (toggle_annealing)
        {
            Annealing<RealT> annealing(multi_seqs, posteriors, toggle_verbose);
            annealing.DoAlignment();
            alignment = annealing.GetAlignment();
        }
        else
        {
            Progressive<RealT> progressive(multi_seqs, posteriors, toggle_verbose);
            progressive.DoAlignment();
            alignment = progressive.GetAlignment();
        }

        OutputPrediction(shared, nonshared, alignment, "", static_cast<SparseMatrix<RealT> *>(NULL));
    }

    // clear progress messages
    if (toggle_verbose) WriteProgressMessage("");
    
    // free memory
    for (int i = 0; i < m; i++)
        for (int j = i+1; j < m; j++)
            delete posteriors[i*m + j];
}

/////////////////////////////////////////////////////////////////
// ComputationEngine<RealT>::OutputPrediction()
//
// Print alignment prediction.
/////////////////////////////////////////////////////////////////

template<class RealT>
void ComputationEngine<RealT>::OutputPrediction(const SharedInfo<RealT> &shared,
                                                const NonSharedInfo &nonshared,
                                                const MultiSequence &alignment,
                                                const std::string &suffix,
                                                const SparseMatrix<RealT> *sparse)
{
    // write output
    if (options.GetStringValue("output_mfa_destination") != "")
    {
        const std::string filename = MakeOutputFilename(descriptions[nonshared.index].input_filename,
                                                        options.GetStringValue("output_mfa_destination"),
                                                        options.GetRealValue("gamma") < 0,
                                                        shared.gamma) + suffix;
        std::ofstream outfile(filename.c_str());
        if (outfile.fail()) Error("Unable to open output mfa file '%s' for writing.", filename.c_str());
        alignment.WriteMFA(outfile);
        outfile.close();
    }
  
    if (options.GetStringValue("output_clustalw_destination") != "")
    {
        const std::string filename = MakeOutputFilename(descriptions[nonshared.index].input_filename,
                                                        options.GetStringValue("output_clustalw_destination"),
                                                        options.GetRealValue("gamma") < 0,
                                                        shared.gamma) + suffix;
        std::ofstream outfile(filename.c_str());
        if (outfile.fail()) Error("Unable to open output clustalw file '%s' for writing.", filename.c_str());
        alignment.WriteCLUSTALW(outfile);
        outfile.close();
    }
    
    if (options.GetStringValue("output_posteriors_destination") != "")
    {
        const std::string filename = MakeOutputFilename(descriptions[nonshared.index].input_filename,
                                                        options.GetStringValue("output_posteriors_destination"),
                                                        options.GetRealValue("gamma") < 0,
                                                        shared.gamma) + suffix;
        if (!sparse) Error("Posteriors not computed.");
        std::ofstream outfile(filename.c_str());
        if (outfile.fail()) Error("Unable to open output posteriors file for writing.");
        alignment.WriteMFA(outfile);
        outfile << "#" << std::endl;
        sparse->PrintSparse(outfile);
        outfile.close();
    }
   
    if (options.GetStringValue("output_mfa_destination") == "" &&
        options.GetStringValue("output_clustalw_destination") == "" &&
        options.GetStringValue("output_posteriors_destination") == "")
    {
        WriteProgressMessage("");
        alignment.WriteMFA(std::cout);
    }
}

//////////////////////////////////////////////////////////////////////
// ComputationEngine::MakeOutputFilename()
//
// Decide on output filename, if any.  The arguments to this function
// consist of (1) a boolean variable indicating whether the output
// destination should be treated as the name of an output directory
// (and the output filename is chosen to match the input file) or
// whether the output destination should be interpreted as the output
// filename; (2) the name of the input file to be processed; and (3)
// the supplied output destination.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::string ComputationEngine<RealT>::MakeOutputFilename(const std::string &input_filename,
                                                         const std::string &output_destination,
                                                         const bool cross_validation,
                                                         const RealT gamma) const 
{
    if (output_destination == "") return "";

    const std::string dir_name = GetDirName(output_destination);
    const std::string base_name = GetBaseName(output_destination);

    const std::string prefix = (dir_name != "" ? (dir_name + DIR_SEPARATOR_CHAR) : std::string(""));
    
    // check if output directory required
    if (descriptions.size() > 1)
    {
        if (cross_validation)
        {
            return SPrintF("%s%s%c%s.gamma=%lf%c%s",
                           prefix.c_str(),
                           base_name.c_str(),
                           DIR_SEPARATOR_CHAR,
                           base_name.c_str(),
                           double(gamma),
                           DIR_SEPARATOR_CHAR,
                           GetBaseName(input_filename).c_str());
        }
        return SPrintF("%s%s%c%s",
                       prefix.c_str(),
                       base_name.c_str(),
                       DIR_SEPARATOR_CHAR,
                       GetBaseName(input_filename).c_str());
    }
    
    if (cross_validation)
    {
        return SPrintF("%s%s%c%s.gamma=%lf",
                       prefix.c_str(),
                       base_name.c_str(),
                       DIR_SEPARATOR_CHAR,
                       base_name.c_str(),
                       double(gamma));
    }
    return SPrintF("%s%s",
                   prefix.c_str(),
                   base_name.c_str());
}
