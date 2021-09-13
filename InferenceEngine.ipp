//////////////////////////////////////////////////////////////////////
// InferenceEngine.ipp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// UPDATE_MAX()
//
// Macro for updating a score/traceback pointer which does not
// evaluate t unless an update is needed.  Make sure that this is
// used as a stand-alone statement (i.e., not the "if" condition
// of an if-then-else statement.)
//////////////////////////////////////////////////////////////////////

#define UPDATE_MAX(bs,bt,s,t) do { RealT work(s); if ((work)>(bs)) { (bs)=(work); (bt)=(t); } } while(0)

//////////////////////////////////////////////////////////////////////
// FillScores()
// FillCounts()
// 
// Routines for setting scores and counts quickly. 
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::FillScores(typename std::vector<std::pair<RealT, RealT> >::iterator begin, typename std::vector<std::pair<RealT, RealT> >::iterator end, RealT value)
{
    while (begin != end)
    {
        begin->first = value;
        ++begin;
    }
}

template<class RealT>
void InferenceEngine<RealT>::FillCounts(typename std::vector<std::pair<RealT,RealT> >::iterator begin, typename std::vector<std::pair<RealT,RealT> >::iterator end, RealT value)
{
    while (begin != end)
    {
        begin->second = value;
        ++begin;
    }
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::InferenceEngine()
//
// Constructor
//////////////////////////////////////////////////////////////////////

template<class RealT>
InferenceEngine<RealT>::InferenceEngine() :
    parameter_manager(NULL),
    LX(0),
    LY(0),
    SIZE(0)
{
    // precompute mapping from characters to index representation
    std::fill(char_mapping, char_mapping + 256, BYTE(alphabet.size()));
    for (size_t i = 0; i < alphabet.size(); i++)
    {
        char_mapping[BYTE(tolower(alphabet[i]))] = 
            char_mapping[BYTE(toupper(alphabet[i]))] = i;
    }

#if PARAMS_HYDROPATHY
    // precompute whether each character is hydrophilic or not
    std::fill(is_hydrophilic, is_hydrophilic + 256, false);
    for (size_t i = 0; i < hydrophilic_alphabet.size(); i++)
        is_hydrophilic[BYTE(tolower(hydrophilic_alphabet[i]))] =
            is_hydrophilic[BYTE(toupper(hydrophilic_alphabet[i]))] = true;
#endif

}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::~InferenceEngine()
//
// Destructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
InferenceEngine<RealT>::~InferenceEngine()
{}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::RegisterParameters()
//
// Establish a mapping between parameters in the inference
// engine and parameters in the parameter manager.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::RegisterParameters(ParameterManager<RealT> &parameter_manager)
{
    char buffer[1000];
    char buffer2[1000];

    this->parameter_manager = &parameter_manager;
    parameter_manager.ClearParameters();
    
#if SINGLE_HYPERPARAMETER
    parameter_manager.AddParameterGroup("all_params");
#endif
    
#if PARAMS_MATCH
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("match");
#endif
    for (int i = 0; i <= M; i++)
    {
        for (int j = 0; j <= M; j++)
        {
            if (i == M || j == M)
            {
                score_match[i][j] = std::pair<RealT,RealT>(0, 0);
            }
            else
            {
                sprintf(buffer, "match_%c%c", alphabet[i], alphabet[j]);
                sprintf(buffer2, "match_%c%c", alphabet[j], alphabet[i]);
                if (strcmp(buffer, buffer2) < 0)
                    parameter_manager.AddParameterMapping(buffer, &score_match[i][j]);
                else
                    parameter_manager.AddParameterMapping(buffer2, &score_match[i][j]);
            }
        }
    }
#endif

#if PARAMS_MATCH && PARAMS_COMPRESSED
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("compressed_match");
#endif
    for (int i = 0; i <= M; i++)
    {
        for (int j = 0; j <= M; j++)
        {
            score_compressed_match[i][j] = std::pair<RealT,RealT>(0, 0);
        }
    }
          
    for (int ii = 0; ii < COMPRESSED_M; ii++)
    {
        for (int jj = 0; jj < COMPRESSED_M; jj++)
        {
            char *param_name = NULL;
            sprintf(buffer, "match_%s%s", compressed_alphabet[ii].c_str(), compressed_alphabet[jj].c_str());
            sprintf(buffer2, "match_%s%s", compressed_alphabet[jj].c_str(), compressed_alphabet[ii].c_str());
            if (strcmp(buffer, buffer2) < 0)
                param_name = buffer;
            else
                param_name = buffer2;

            for (size_t i = 0; i < compressed_alphabet[ii].length(); i++)
            {
                for (size_t j = 0; j < compressed_alphabet[jj].length(); j++)
                {
                    BYTE c = compressed_alphabet[ii][i];
                    BYTE d = compressed_alphabet[jj][j];
                    parameter_manager.AddParameterMapping(param_name, &score_compressed_match[char_mapping[c]][char_mapping[d]]);
                }
            }
        }
    }
#endif

#if PARAMS_INSERT
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("insert");
#endif
    for (int i = 0; i <= M; i++)
    {
        if (i == M)
        {
            score_insert[i] = std::pair<RealT,RealT>(0, 0);
        }
        else
        {
            sprintf(buffer, "insert_%c", alphabet[i]);
            parameter_manager.AddParameterMapping(buffer, &score_insert[i]);
        }
    }
#endif

#if PARAMS_INSERT && PARAMS_COMPRESSED
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("compressed_insert");
#endif
    for (int i = 0; i <= M; i++)
    {
        score_compressed_insert[i] = std::pair<RealT,RealT>(0, 0);
    }
          
    for (int ii = 0; ii < COMPRESSED_M; ii++)
    {
        sprintf(buffer, "insert_%s", compressed_alphabet[ii].c_str());
        for (size_t i = 0; i < compressed_alphabet[ii].length(); i++)
        {
            BYTE c = compressed_alphabet[ii][i];
            parameter_manager.AddParameterMapping(buffer, &score_compressed_insert[char_mapping[c]]);
        }
    }
#endif

#if PARAMS_SINGLE
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("single");
#endif
    parameter_manager.AddParameterMapping("match",   &score_single[MATCH]);
    parameter_manager.AddParameterMapping("insert",  &score_single[INS_X]);
    parameter_manager.AddParameterMapping("insert",  &score_single[INS_Y]);
#if PARAMS_DOUBLE_AFFINE
    parameter_manager.AddParameterMapping("insert2", &score_single[INS2_X]);
    parameter_manager.AddParameterMapping("insert2", &score_single[INS2_Y]);
#endif
#endif
    
#if PARAMS_PAIR
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("pair");
#endif
    parameter_manager.AddParameterMapping("match_to_match",  &score_pair[MATCH][MATCH]);
    parameter_manager.AddParameterMapping("match_to_insert", &score_pair[MATCH][INS_X]);
    parameter_manager.AddParameterMapping("match_to_insert", &score_pair[MATCH][INS_Y]);
    parameter_manager.AddParameterMapping("match_to_insert", &score_pair[INS_X][MATCH]);
    parameter_manager.AddParameterMapping("insert_extend",   &score_pair[INS_X][INS_X]);
    parameter_manager.AddParameterMapping("insert_change",   &score_pair[INS_X][INS_Y]);
    parameter_manager.AddParameterMapping("match_to_insert", &score_pair[INS_Y][MATCH]);
    parameter_manager.AddParameterMapping("insert_change",   &score_pair[INS_Y][INS_X]);
    parameter_manager.AddParameterMapping("insert_extend",   &score_pair[INS_Y][INS_Y]);
#if PARAMS_DOUBLE_AFFINE
    parameter_manager.AddParameterMapping("match_to_insert2", &score_pair[MATCH][INS2_X]);
    parameter_manager.AddParameterMapping("match_to_insert2", &score_pair[MATCH][INS2_Y]);
    parameter_manager.AddParameterMapping("match_to_insert2", &score_pair[INS2_X][MATCH]);
    parameter_manager.AddParameterMapping("insert2_extend",   &score_pair[INS2_X][INS2_X]);
    parameter_manager.AddParameterMapping("insert2_change",   &score_pair[INS2_X][INS2_Y]);
    parameter_manager.AddParameterMapping("match_to_insert2", &score_pair[INS2_Y][MATCH]);
    parameter_manager.AddParameterMapping("insert2_change",   &score_pair[INS2_Y][INS2_X]);
    parameter_manager.AddParameterMapping("insert2_extend",   &score_pair[INS2_Y][INS2_Y]);
#endif
#endif

#if PARAMS_HYDROPATHY
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("hydropathy");
#endif
    for (int h = 0; h <= H; h++)
    {
        sprintf(buffer, "match_to_insert_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[MATCH][INS_X][h]);
        sprintf(buffer, "match_to_insert_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[MATCH][INS_Y][h]);
        sprintf(buffer, "match_to_insert_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS_X][MATCH][h]);
        sprintf(buffer, "match_to_insert_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS_Y][MATCH][h]);
        sprintf(buffer, "insert_to_insert_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS_X][INS_X][h]);
        sprintf(buffer, "insert_to_insert_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS_Y][INS_Y][h]);

#if PARAMS_DOUBLE_AFFINE
        sprintf(buffer, "match_to_insert2_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[MATCH][INS2_X][h]);
        sprintf(buffer, "match_to_insert2_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[MATCH][INS2_Y][h]);
        sprintf(buffer, "match_to_insert2_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS2_X][MATCH][h]);
        sprintf(buffer, "match_to_insert2_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS2_Y][MATCH][h]);
        sprintf(buffer, "insert2_to_insert2_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS2_X][INS2_X][h]);
        sprintf(buffer, "insert2_to_insert2_hydrophilic_count_%d", h);
        parameter_manager.AddParameterMapping(buffer, &score_hydropathy[INS2_Y][INS2_Y][h]);
#endif
    }
#endif

#if PARAMS_TERMINAL_INSERTS
#if MULTIPLE_HYPERPARAMETERS
    parameter_manager.AddParameterGroup("terminal_inserts");
#endif
    parameter_manager.AddParameterMapping("terminal_insert_boundary", &score_terminal_insert_boundary);
    parameter_manager.AddParameterMapping("terminal_insert", &score_terminal_insert);
#endif

}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::LoadSequences()
//
// Load a pair of unaligned sequences.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::LoadSequences(const MultiSequence &seqs)
{
    Sequence seqX(seqs.GetSequence(0), Sequence::COMPRESS_GAPS);
    Sequence seqY(seqs.GetSequence(1), Sequence::COMPRESS_GAPS);

    // compute dimensions
    LX = seqX.GetLength();
    LY = seqY.GetLength();
    SIZE = (LX+1)*(LY+1);

    // allocate memory
    x.clear();                     x.resize(LX+1);
    y.clear();                     y.resize(LY+1);
    aligned_to_x.clear();          aligned_to_x.resize(LX+1, Sequence::UNKNOWN);
    aligned_to_y.clear();          aligned_to_y.resize(LY+1, Sequence::UNKNOWN);
    true_aligned_to_x.clear();     true_aligned_to_x.resize(LX+1, Sequence::UNKNOWN);
    true_aligned_to_y.clear();     true_aligned_to_y.resize(LY+1, Sequence::UNKNOWN);

    // convert sequences
    const std::string &sx = seqX.GetData();
    x[0] = BYTE(alphabet.size());
    for (int i = 1; i <= LX; i++)
        x[i] = char_mapping[BYTE(sx[i])];
    
    const std::string &sy = seqY.GetData();
    y[0] = BYTE(alphabet.size());
    for (int i = 1; i <= LY; i++)
        y[i] = char_mapping[BYTE(sy[i])];

#if PARAMS_HYDROPATHY
    // precompute hydropathy information 
    hydrophilic_count_x.clear();   hydrophilic_count_x.resize(LX+1, 0);
    hydrophilic_count_y.clear();   hydrophilic_count_y.resize(LY+1, 0);
    
    for (int d = -H/2+1; d <= H/2; d++)
    {
        for (int i = 0; i <= LX; i++) hydrophilic_count_x[i] += (0 < i+d && i+d <= LX ? is_hydrophilic[BYTE(sx[i+d])] : 0);
        for (int j = 0; j <= LY; j++) hydrophilic_count_y[j] += (0 < j+d && j+d <= LY ? is_hydrophilic[BYTE(sy[j+d])] : 0);
    }
#endif
    
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::LoadValues()
//
// Load parameter values.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::LoadValues(const std::vector<RealT> &values)
{
    if (values.size() != parameter_manager->GetNumLogicalParameters()) Error("Parameter size mismatch.");
    
    for (size_t i = 0; i < values.size(); i++)
    {
        std::vector<std::pair<RealT,RealT> *> physical_parameters = parameter_manager->GetPhysicalParameters(i);
        for (size_t j = 0; j < physical_parameters.size(); j++)
            physical_parameters[j]->first = values[i];
    }
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::GetCounts()
//
// Return counts.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> InferenceEngine<RealT>::GetCounts()
{
    std::vector<RealT> counts(parameter_manager->GetNumLogicalParameters());
    
    // clear counts for physical parameters
    for (size_t i = 0; i < parameter_manager->GetNumLogicalParameters(); i++)
    {
        std::vector<std::pair<RealT,RealT> *> physical_parameters = parameter_manager->GetPhysicalParameters(i);
        for (size_t j = 0; j < physical_parameters.size(); j++)
            counts[i] += physical_parameters[j]->second;
    }

    return counts;
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ClearCounts()
//
// Set all counts to zero.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ClearCounts()
{
    // clear counts for physical parameters
    for (size_t i = 0; i < parameter_manager->GetNumLogicalParameters(); i++)
    {
        std::vector<std::pair<RealT,RealT> *> physical_parameters = parameter_manager->GetPhysicalParameters(i);
        for (size_t j = 0; j < physical_parameters.size(); j++)
            physical_parameters[j]->second = RealT(0);
    }
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::UseLoss()
//
// Use per-position loss.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::UseLoss(const std::pair<std::vector<int>, std::vector<int> > &true_aligned_to, RealT example_loss)
{
    Assert(int(true_aligned_to.first.size()) == LX+1, "Constraints of incorrect length.");
    Assert(int(true_aligned_to.second.size()) == LY+1, "Constraints of incorrect length.");
    
    std::copy(true_aligned_to.first.begin(), true_aligned_to.first.end(), true_aligned_to_x.begin());
    std::copy(true_aligned_to.second.begin(), true_aligned_to.second.end(), true_aligned_to_y.begin());
    
    int count = 0;
    for (int i = 1; i <= LX; i++)
        if (true_aligned_to_x[i] != Sequence::UNKNOWN && true_aligned_to_x[i] != Sequence::UNALIGNED) count++;
    for (int i = 1; i <= LY; i++)
        if (true_aligned_to_y[i] != Sequence::UNKNOWN && true_aligned_to_y[i] != Sequence::UNALIGNED) count++;

    per_position_loss = example_loss / RealT(count);
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::UseConstraints()
//
// Use known alignment.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::UseConstraints(const std::pair<std::vector<int>, std::vector<int> > &aligned_to)
{
    Assert(int(aligned_to.first.size()) == LX+1, "Constraints of incorrect length.");
    Assert(int(aligned_to.second.size()) == LY+1, "Constraints of incorrect length.");
    
    std::copy(aligned_to.first.begin(), aligned_to.first.end(), aligned_to_x.begin());
    std::copy(aligned_to.second.begin(), aligned_to.second.end(), aligned_to_y.begin());
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ScoreMatch()
// InferenceEngine::CountMatch()
//
// Returns the score for matching x[i] with y[j].
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ScoreMatch(int i, int j, int s) const
{
    Assert(0 < i && i <= LX && 0 < j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if ((aligned_to_x[i] != Sequence::UNKNOWN && aligned_to_x[i] != j) ||
        (aligned_to_y[j] != Sequence::UNKNOWN && aligned_to_y[j] != i))
        return RealT(NEG_INF);
    
    return RealT(0)
#if defined(HAMMING_LOSS)
        + ((true_aligned_to_x[i] == Sequence::UNKNOWN || true_aligned_to_x[i] == Sequence::UNALIGNED || true_aligned_to_x[i] == j) ? RealT(0) : per_position_loss)
        + ((true_aligned_to_y[j] == Sequence::UNKNOWN || true_aligned_to_y[j] == Sequence::UNALIGNED || true_aligned_to_y[j] == i) ? RealT(0) : per_position_loss)
#endif
#if PARAMS_MATCH
        + score_match[x[i]][y[j]].first
#if PARAMS_COMPRESSED
        + score_compressed_match[x[i]][y[j]].first
#endif
#endif
#if PARAMS_SINGLE
        + score_single[MATCH].first
#endif
#if PARAMS_PAIR
        + (i != 1 || j != 1 ? score_pair[s][MATCH].first : RealT(0))
#endif
#if PARAMS_HYDROPATHY
        + (s == INS_X ? score_hydropathy[INS_X][MATCH][hydrophilic_count_y[j-1]].first : RealT(0))
        + (s == INS_Y ? score_hydropathy[INS_Y][MATCH][hydrophilic_count_x[i-1]].first : RealT(0))
#if PARAMS_DOUBLE_AFFINE
        + (s == INS2_X ? score_hydropathy[INS2_X][MATCH][hydrophilic_count_y[j-1]].first : RealT(0))
        + (s == INS2_Y ? score_hydropathy[INS2_Y][MATCH][hydrophilic_count_x[i-1]].first : RealT(0))
#endif
#endif
        ;
}

template<class RealT>
inline void InferenceEngine<RealT>::CountMatch(int i, int j, int s, RealT value)
{
    Assert(0 < i && i <= LX && 0 < j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if ((aligned_to_x[i] != Sequence::UNKNOWN && aligned_to_x[i] != j) ||
        (aligned_to_y[j] != Sequence::UNKNOWN && aligned_to_y[j] != i))
        return;
    
#if PARAMS_MATCH
    score_match[x[i]][y[j]].second += value;
#if PARAMS_COMPRESSED
    score_compressed_match[x[i]][y[j]].second += value;
#endif
#endif
#if PARAMS_SINGLE
    score_single[MATCH].second += value;
#endif
#if PARAMS_PAIR
    if (i != 1 || j != 1) score_pair[s][MATCH].second += value;
#endif
#if PARAMS_HYDROPATHY
    if (s == INS_X) score_hydropathy[INS_X][MATCH][hydrophilic_count_y[j-1]].second += value;
    if (s == INS_Y) score_hydropathy[INS_Y][MATCH][hydrophilic_count_x[i-1]].second += value;
#if PARAMS_DOUBLE_AFFINE
    if (s == INS2_X) score_hydropathy[INS2_X][MATCH][hydrophilic_count_y[j-1]].second += value;
    if (s == INS2_Y) score_hydropathy[INS2_Y][MATCH][hydrophilic_count_x[i-1]].second += value;
#endif
#endif
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ScoreInsertX()
// InferenceEngine::CountInsertX()
//
// Returns the score for inserting x[i].
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ScoreInsertX(int i, int j, int s) const
{
    Assert(0 < i && i <= LX && 0 <= j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_x[i] != Sequence::UNKNOWN && aligned_to_x[i] != Sequence::UNALIGNED)
        return RealT(NEG_INF);
    
    return RealT(0)
#if defined(HAMMING_LOSS)
        + ((true_aligned_to_x[i] == Sequence::UNKNOWN || true_aligned_to_x[i] == Sequence::UNALIGNED) ? RealT(0) : per_position_loss)
#endif
#if PARAMS_INSERT
        + score_insert[x[i]].first
#if PARAMS_COMPRESSED
        + score_compressed_insert[x[i]].first
#endif
#endif
#if PARAMS_SINGLE
        + score_single[INS_X].first
#endif
#if PARAMS_PAIR
        + (i != 1 || j != 0 ? score_pair[s][INS_X].first : RealT(0))
#endif
#if PARAMS_HYDROPATHY
        + (s == MATCH ? score_hydropathy[MATCH][INS_X][hydrophilic_count_y[j]].first : RealT(0))
        + (s == INS_X ? score_hydropathy[INS_X][INS_X][hydrophilic_count_y[j]].first : RealT(0))
#endif
#if PARAMS_TERMINAL_INSERTS
        + (j == 0 || j == LY ? score_terminal_insert.first : RealT(0))
        + ((i == 1 && j == 0) || (i == LX && j == LY) ? score_terminal_insert_boundary.first : RealT(0))
#endif
        ;
}

template<class RealT>
inline void InferenceEngine<RealT>::CountInsertX(int i, int j, int s, RealT value)
{
    Assert(0 < i && i <= LX && 0 <= j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_x[i] != Sequence::UNKNOWN && aligned_to_x[i] != Sequence::UNALIGNED)
        return;
    
#if PARAMS_INSERT
    score_insert[x[i]].second += value;
#if PARAMS_COMPRESSED
    score_compressed_insert[x[i]].second += value;
#endif
#endif
#if PARAMS_SINGLE
    score_single[INS_X].second += value;
#endif
#if PARAMS_PAIR
    if (i != 1 || j != 0) score_pair[s][INS_X].second += value;
#endif
#if PARAMS_HYDROPATHY
    if (s == MATCH) score_hydropathy[MATCH][INS_X][hydrophilic_count_y[j]].second += value;
    if (s == INS_X) score_hydropathy[INS_X][INS_X][hydrophilic_count_y[j]].second += value;
#endif
#if PARAMS_TERMINAL_INSERTS
    if (j == 0 || j == LY) score_terminal_insert.second += value;
    if ((i == 1 && j == 0) || (i == LX && j == LY)) score_terminal_insert_boundary.second += value;
#endif
}

#if PARAMS_DOUBLE_AFFINE
/////////////////////////////////////////////////////////////////
// InferenceEngine::ScoreInsert2X()
// InferenceEngine::CountInsert2X()
//
// Returns the score for inserting x[i].
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ScoreInsert2X(int i, int j, int s) const
{
    Assert(0 < i && i <= LX && 0 <= j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_x[i] != Sequence::UNKNOWN && aligned_to_x[i] != Sequence::UNALIGNED)
        return RealT(NEG_INF);
    
    return RealT(0)
#if defined(HAMMING_LOSS)
        + ((true_aligned_to_x[i] == Sequence::UNKNOWN || true_aligned_to_x[i] == Sequence::UNALIGNED) ? RealT(0) : per_position_loss)
#endif
#if PARAMS_INSERT
        + score_insert[x[i]].first
#if PARAMS_COMPRESSED
        + score_compressed_insert[x[i]].first
#endif
#endif
#if PARAMS_SINGLE
        + score_single[INS2_X].first
#endif
#if PARAMS_PAIR
        + (i != 1 || j != 0 ? score_pair[s][INS2_X].first : RealT(0))
#endif
#if PARAMS_HYDROPATHY
        + (s == MATCH ? score_hydropathy[MATCH][INS2_X][hydrophilic_count_y[j]].first : RealT(0))
        + (s == INS2_X ? score_hydropathy[INS2_X][INS2_X][hydrophilic_count_y[j]].first : RealT(0))
#endif
#if PARAMS_TERMINAL_INSERTS
        + (j == 0 || j == LY ? score_terminal_insert.first : RealT(0))
        + ((i == 1 && j == 0) || (i == LX && j == LY) ? score_terminal_insert_boundary.first : RealT(0))
#endif
        ;
}

template<class RealT>
inline void InferenceEngine<RealT>::CountInsert2X(int i, int j, int s, RealT value)
{
    Assert(0 < i && i <= LX && 0 <= j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_x[i] != Sequence::UNKNOWN && aligned_to_x[i] != Sequence::UNALIGNED)
        return;
    
#if PARAMS_INSERT
    score_insert[x[i]].second += value;
#if PARAMS_COMPRESSED
    score_compressed_insert[x[i]].second += value;
#endif
#endif
#if PARAMS_SINGLE
    score_single[INS2_X].second += value;
#endif
#if PARAMS_PAIR
    if (i != 1 || j != 0) score_pair[s][INS2_X].second += value;
#endif
#if PARAMS_HYDROPATHY
    if (s == MATCH) score_hydropathy[MATCH][INS2_X][hydrophilic_count_y[j]].second += value;
    if (s == INS2_X) score_hydropathy[INS2_X][INS2_X][hydrophilic_count_y[j]].second += value;
#endif
#if PARAMS_TERMINAL_INSERTS
    if (j == 0 || j == LY) score_terminal_insert.second += value;
    if ((i == 1 && j == 0) || (i == LX && j == LY)) score_terminal_insert_boundary.second += value;
#endif
}
#endif

/////////////////////////////////////////////////////////////////
// InferenceEngine::ScoreInsertY()
// InferenceEngine::CountInsertY()
//
// Returns the score for inserting y[i].
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ScoreInsertY(int i, int j, int s) const
{
    Assert(0 <= i && i <= LX && 0 < j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_y[j] != Sequence::UNKNOWN && aligned_to_y[j] != Sequence::UNALIGNED)
        return RealT(NEG_INF);
    
    return RealT(0)
#if defined(HAMMING_LOSS)
        + ((true_aligned_to_y[j] == Sequence::UNKNOWN || true_aligned_to_y[j] == Sequence::UNALIGNED) ? RealT(0) : per_position_loss)
#endif
#if PARAMS_INSERT
        + score_insert[y[j]].first
#if PARAMS_COMPRESSED
        + score_compressed_insert[y[j]].first
#endif
#endif
#if PARAMS_SINGLE
        + score_single[INS_Y].first
#endif
#if PARAMS_PAIR
        + (i != 0 || j != 1 ? score_pair[s][INS_Y].first : RealT(0))
#endif
#if PARAMS_HYDROPATHY
        + (s == MATCH ? score_hydropathy[MATCH][INS_Y][hydrophilic_count_x[i]].first : RealT(0))
        + (s == INS_Y ? score_hydropathy[INS_Y][INS_Y][hydrophilic_count_x[i]].first : RealT(0))
#endif
#if PARAMS_TERMINAL_INSERTS
        + ((i == 0 || i == LX) ? score_terminal_insert.first : RealT(0))
        + ((i == 0 && j == 1) || (i == LX && j == LY) ? score_terminal_insert_boundary.first : RealT(0))
#endif
        ;
}

template<class RealT>
inline void InferenceEngine<RealT>::CountInsertY(int i, int j, int s, RealT value)
{
    Assert(0 <= i && i <= LX && 0 < j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_y[j] != Sequence::UNKNOWN && aligned_to_y[j] != Sequence::UNALIGNED)
        return;
    
#if PARAMS_INSERT
    score_insert[y[j]].second += value;
#if PARAMS_COMPRESSED
    score_compressed_insert[y[j]].second += value;
#endif
#endif
#if PARAMS_SINGLE
    score_single[INS_Y].second += value;
#endif
#if PARAMS_PAIR
    if (i != 0 || j != 1) score_pair[s][INS_Y].second += value;
#endif
#if PARAMS_HYDROPATHY
    if (s == MATCH) score_hydropathy[MATCH][INS_Y][hydrophilic_count_x[i]].second += value;
    if (s == INS_Y) score_hydropathy[INS_Y][INS_Y][hydrophilic_count_x[i]].second += value;
#endif
#if PARAMS_TERMINAL_INSERTS
    if (i == 0 || i == LX) score_terminal_insert.second += value;
    if ((i == 0 && j == 1) || (i == LX && j == LY)) score_terminal_insert_boundary.second += value;
#endif
}

#if PARAMS_DOUBLE_AFFINE
/////////////////////////////////////////////////////////////////
// InferenceEngine::ScoreInsert2Y()
// InferenceEngine::CountInsert2Y()
//
// Returns the score for inserting y[i].
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ScoreInsert2Y(int i, int j, int s) const
{
    Assert(0 <= i && i <= LX && 0 < j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_y[j] != Sequence::UNKNOWN && aligned_to_y[j] != Sequence::UNALIGNED)
        return RealT(NEG_INF);
    
    return RealT(0)
#if defined(HAMMING_LOSS)
        + ((true_aligned_to_y[j] == Sequence::UNKNOWN || true_aligned_to_y[j] == Sequence::UNALIGNED) ? RealT(0) : per_position_loss)
#endif
#if PARAMS_INSERT
        + score_insert[y[j]].first
#if PARAMS_COMPRESSED
        + score_compressed_insert[y[j]].first
#endif
#endif
#if PARAMS_SINGLE
        + score_single[INS2_Y].first
#endif
#if PARAMS_PAIR
        + (i != 0 || j != 1 ? score_pair[s][INS2_Y].first : RealT(0))
#endif
#if PARAMS_HYDROPATHY
        + (s == MATCH ? score_hydropathy[MATCH][INS2_Y][hydrophilic_count_x[i]].first : RealT(0))
        + (s == INS2_Y ? score_hydropathy[INS2_Y][INS2_Y][hydrophilic_count_x[i]].first : RealT(0))
#endif
#if PARAMS_TERMINAL_INSERTS
        + ((i == 0 || i == LX) ? score_terminal_insert.first : RealT(0))
        + ((i == 0 && j == 1) || (i == LX && j == LY) ? score_terminal_insert_boundary.first : RealT(0))
#endif
        ;
}

template<class RealT>
inline void InferenceEngine<RealT>::CountInsert2Y(int i, int j, int s, RealT value)
{
    Assert(0 <= i && i <= LX && 0 < j && j <= LY, "Invalid indices.");
    Assert(0 <= s && s < K, "Invalid state.");
    
    if (aligned_to_y[j] != Sequence::UNKNOWN && aligned_to_y[j] != Sequence::UNALIGNED)
        return;
    
#if PARAMS_INSERT
    score_insert[y[j]].second += value;
#if PARAMS_COMPRESSED
    score_compressed_insert[y[j]].second += value;
#endif
#endif
#if PARAMS_SINGLE
    score_single[INS2_Y].second += value;
#endif
#if PARAMS_PAIR
    if (i != 0 || j != 1) score_pair[s][INS2_Y].second += value;
#endif
#if PARAMS_HYDROPATHY
    if (s == MATCH) score_hydropathy[MATCH][INS2_Y][hydrophilic_count_x[i]].second += value;
    if (s == INS2_Y) score_hydropathy[INS2_Y][INS2_Y][hydrophilic_count_x[i]].second += value;
#endif
#if PARAMS_TERMINAL_INSERTS
    if (i == 0 || i == LX) score_terminal_insert.second += value;
    if ((i == 0 && j == 1) || (i == LX && j == LY)) score_terminal_insert_boundary.second += value;
#endif
}
#endif

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeViterbi()
//
// Run Viterbi algorithm.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputeViterbi()
{
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif

    // initialization
    for (int k = 0; k < K; k++)
    {
        Ft[k].clear();
        Ft[k].resize(SIZE, -1);
        Fv[k].clear();
        Fv[k].resize(SIZE, RealT(NEG_INF));
        Fv[k][0] = RealT(0);
    }
    
    for (int i = 1; i <= LX; i++) { Fv[INS_X][i*(LY+1)+0] = Fv[INS_X][(i-1)*(LY+1)+0] + ScoreInsertX(i,0,INS_X); Ft[INS_X][i*(LY+1)+0] = INS_X; }
    for (int j = 1; j <= LY; j++) { Fv[INS_Y][0*(LY+1)+j] = Fv[INS_Y][0*(LY+1)+(j-1)] + ScoreInsertY(0,j,INS_Y); Ft[INS_Y][0*(LY+1)+j] = INS_Y; }
#if PARAMS_DOUBLE_AFFINE
    for (int i = 1; i <= LX; i++) { Fv[INS2_X][i*(LY+1)+0] = Fv[INS2_X][(i-1)*(LY+1)+0] + ScoreInsert2X(i,0,INS2_X); Ft[INS2_X][i*(LY+1)+0] = INS2_X; }
    for (int j = 1; j <= LY; j++) { Fv[INS2_Y][0*(LY+1)+j] = Fv[INS2_Y][0*(LY+1)+(j-1)] + ScoreInsert2Y(0,j,INS2_Y); Ft[INS2_Y][0*(LY+1)+j] = INS2_Y; }
#endif
    
    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            const int ij = i*(LY+1)+j;
            const int i1j = ij-(LY+1);
            const int ij1 = ij-1;
            const int i1j1 = ij-(LY+1)-1;
            
            UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[MATCH][i1j1] + ScoreMatch(i,j,MATCH), MATCH);
            if (i > 1 || j > 1)
            {
                UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[INS_X][i1j1] + ScoreMatch(i,j,INS_X), INS_X);
                UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[INS_Y][i1j1] + ScoreMatch(i,j,INS_Y), INS_Y);
#if PARAMS_DOUBLE_AFFINE
                UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[INS2_X][i1j1] + ScoreMatch(i,j,INS2_X), INS2_X);
                UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[INS2_Y][i1j1] + ScoreMatch(i,j,INS2_Y), INS2_Y);
#endif
            }
            
            UPDATE_MAX(Fv[INS_X][ij], Ft[INS_X][ij], Fv[MATCH][i1j] + ScoreInsertX(i,j,MATCH), MATCH);
            UPDATE_MAX(Fv[INS_X][ij], Ft[INS_X][ij], Fv[INS_X][i1j] + ScoreInsertX(i,j,INS_X), INS_X);
#if !FORCE_UNIQUE_PARSES
            UPDATE_MAX(Fv[INS_X][ij], Ft[INS_X][ij], Fv[INS_Y][i1j] + ScoreInsertX(i,j,INS_Y), INS_Y);
#endif
            
            UPDATE_MAX(Fv[INS_Y][ij], Ft[INS_Y][ij], Fv[MATCH][ij1] + ScoreInsertY(i,j,MATCH), MATCH);
            UPDATE_MAX(Fv[INS_Y][ij], Ft[INS_Y][ij], Fv[INS_X][ij1] + ScoreInsertY(i,j,INS_X), INS_X);
            UPDATE_MAX(Fv[INS_Y][ij], Ft[INS_Y][ij], Fv[INS_Y][ij1] + ScoreInsertY(i,j,INS_Y), INS_Y);
            
#if PARAMS_DOUBLE_AFFINE
            UPDATE_MAX(Fv[INS2_X][ij], Ft[INS2_X][ij], Fv[MATCH][i1j] + ScoreInsert2X(i,j,MATCH), MATCH);
            UPDATE_MAX(Fv[INS2_X][ij], Ft[INS2_X][ij], Fv[INS2_X][i1j] + ScoreInsert2X(i,j,INS2_X), INS2_X);
#if !FORCE_UNIQUE_PARSES
            UPDATE_MAX(Fv[INS2_X][ij], Ft[INS2_X][ij], Fv[INS2_Y][i1j] + ScoreInsert2X(i,j,INS2_Y), INS2_Y);
#endif
            
            UPDATE_MAX(Fv[INS2_Y][ij], Ft[INS2_Y][ij], Fv[MATCH][ij1] + ScoreInsert2Y(i,j,MATCH), MATCH);
            UPDATE_MAX(Fv[INS2_Y][ij], Ft[INS2_Y][ij], Fv[INS2_X][ij1] + ScoreInsert2Y(i,j,INS2_X), INS2_X);
            UPDATE_MAX(Fv[INS2_Y][ij], Ft[INS2_Y][ij], Fv[INS2_Y][ij1] + ScoreInsert2Y(i,j,INS2_Y), INS2_Y);
#endif
        }
    }

#if SHOW_TIMINGS
    std::cerr << "Viterbi score: " << GetViterbiScore() 
              << " (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::GetViterbiScore()
//
// Return Viterbi score for a sequence.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::GetViterbiScore() const
{
    RealT ret = Fv[MATCH][SIZE-1];
    for (int k = 1; k < K; k++)
        ret = std::max(ret, Fv[k][SIZE-1]);
    return ret;
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::PredictAlignmentViterbi()
// 
// Use Viterbi decoding to predict alignment.
/////////////////////////////////////////////////////////////////

template<class RealT>
std::string InferenceEngine<RealT>::PredictAlignmentViterbi()
{
    std::vector<int> aligned_to_x(LX+1);
    std::vector<int> aligned_to_y(LY+1);
    
    int i = LX;
    int j = LY;
    int state = MATCH;
    for (int k = 1; k < K; k++)
        if (Fv[k][SIZE-1] > Fv[state][SIZE-1])
            state = k;

    std::string edit_string;
    while (i > 0 || j > 0)
    {
        int prev_state = Ft[state][i*(LY+1)+j];
        switch (state)
        {
            case MATCH: edit_string.push_back('M'); i--; j--; break;
            case INS_X: edit_string.push_back('X'); i--; break;
            case INS_Y: edit_string.push_back('Y'); j--; break;
#if PARAMS_DOUBLE_AFFINE
            case INS2_X: edit_string.push_back('X'); i--; break;
            case INS2_Y: edit_string.push_back('Y'); j--; break;
#endif
            default: Assert(false, "Should not get here.");
        }
        state = prev_state;
    }
    
    edit_string.push_back('@');
    std::reverse(edit_string.begin(), edit_string.end());
    
    return edit_string;
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeViterbiFeatureCounts()
// 
// Use feature counts from Viterbi decoding.
/////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> InferenceEngine<RealT>::ComputeViterbiFeatureCounts()
{
    int i = LX;
    int j = LY;
    int state = MATCH;
    for (int k = 1; k < K; k++)
        if (Fv[k][SIZE-1] > Fv[state][SIZE-1])
            state = k;
    
    ClearCounts();

    while (i > 0 || j > 0)
    {
        int prev_state = Ft[state][i*(LY+1)+j];
        switch (state)
        {
            case MATCH: CountMatch(i,j,prev_state,RealT(1)); i--; j--; break;
            case INS_X: CountInsertX(i,j,prev_state,RealT(1)); i--; break;
            case INS_Y: CountInsertY(i,j,prev_state,RealT(1)); j--; break;
#if PARAMS_DOUBLE_AFFINE
            case INS2_X: CountInsert2X(i,j,prev_state,RealT(1)); i--; break;
            case INS2_Y: CountInsert2Y(i,j,prev_state,RealT(1)); j--; break;
#endif
            default: Assert(false, "Should not get here.");
        }
        state = prev_state;
    }

    return GetCounts();
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeForward()
//
// Run forward algorithm.
/////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputeForward()
{
    
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif
    
    // initialization
    
    for (int k = 0; k < K; k++)
    {
        Ff[k].clear();
        Ff[k].resize(SIZE, RealT(NEG_INF));
        Ff[k][0] = RealT(0);
    }
    
    for (int i = 1; i <= LX; i++) Fast_LogPlusEquals(Ff[INS_X][i*(LY+1)+0], Ff[INS_X][(i-1)*(LY+1)+0] + ScoreInsertX(i,0,INS_X));
    for (int j = 1; j <= LY; j++) Fast_LogPlusEquals(Ff[INS_Y][0*(LY+1)+j], Ff[INS_Y][0*(LY+1)+(j-1)] + ScoreInsertY(0,j,INS_Y));
#if PARAMS_DOUBLE_AFFINE
    for (int i = 1; i <= LX; i++) Fast_LogPlusEquals(Ff[INS2_X][i*(LY+1)+0], Ff[INS2_X][(i-1)*(LY+1)+0] + ScoreInsert2X(i,0,INS2_X));
    for (int j = 1; j <= LY; j++) Fast_LogPlusEquals(Ff[INS2_Y][0*(LY+1)+j], Ff[INS2_Y][0*(LY+1)+(j-1)] + ScoreInsert2Y(0,j,INS2_Y));
#endif
    
    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            const int ij = i*(LY+1)+j;
            const int i1j = ij-(LY+1);
            const int ij1 = ij-1;
            const int i1j1 = ij-(LY+1)-1;
            
            Fast_LogPlusEquals(Ff[MATCH][ij], Ff[MATCH][i1j1] + ScoreMatch(i,j,MATCH));
            if (i > 1 || j > 1)
            {
                Fast_LogPlusEquals(Ff[MATCH][ij], Ff[INS_X][i1j1] + ScoreMatch(i,j,INS_X));
                Fast_LogPlusEquals(Ff[MATCH][ij], Ff[INS_Y][i1j1] + ScoreMatch(i,j,INS_Y));
#if PARAMS_DOUBLE_AFFINE
                Fast_LogPlusEquals(Ff[MATCH][ij], Ff[INS2_X][i1j1] + ScoreMatch(i,j,INS2_X));
                Fast_LogPlusEquals(Ff[MATCH][ij], Ff[INS2_Y][i1j1] + ScoreMatch(i,j,INS2_Y));
#endif
            }
            
            Fast_LogPlusEquals(Ff[INS_X][ij], Ff[MATCH][i1j] + ScoreInsertX(i,j,MATCH));
            Fast_LogPlusEquals(Ff[INS_X][ij], Ff[INS_X][i1j] + ScoreInsertX(i,j,INS_X));
#if !FORCE_UNIQUE_PARSES
            Fast_LogPlusEquals(Ff[INS_X][ij], Ff[INS_Y][i1j] + ScoreInsertX(i,j,INS_Y));
#endif
            
            Fast_LogPlusEquals(Ff[INS_Y][ij], Ff[MATCH][ij1] + ScoreInsertY(i,j,MATCH));
            Fast_LogPlusEquals(Ff[INS_Y][ij], Ff[INS_X][ij1] + ScoreInsertY(i,j,INS_X));
            Fast_LogPlusEquals(Ff[INS_Y][ij], Ff[INS_Y][ij1] + ScoreInsertY(i,j,INS_Y));
            
#if PARAMS_DOUBLE_AFFINE
            Fast_LogPlusEquals(Ff[INS2_X][ij], Ff[MATCH][i1j] + ScoreInsert2X(i,j,MATCH));
            Fast_LogPlusEquals(Ff[INS2_X][ij], Ff[INS2_X][i1j] + ScoreInsert2X(i,j,INS2_X));
#if !FORCE_UNIQUE_PARSES
            Fast_LogPlusEquals(Ff[INS2_X][ij], Ff[INS2_Y][i1j] + ScoreInsert2X(i,j,INS2_Y));
#endif
            
            Fast_LogPlusEquals(Ff[INS2_Y][ij], Ff[MATCH][ij1] + ScoreInsert2Y(i,j,MATCH));
            Fast_LogPlusEquals(Ff[INS2_Y][ij], Ff[INS2_X][ij1] + ScoreInsert2Y(i,j,INS2_X));
            Fast_LogPlusEquals(Ff[INS2_Y][ij], Ff[INS2_Y][ij1] + ScoreInsert2Y(i,j,INS2_Y));
#endif
        }
    }
    
#if SHOW_TIMINGS
    std::cerr << "Forward score: " << ComputeForwardLogPartitionCoefficient()
              << " (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeBackward()
//
// Run backward algorithm.
/////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputeBackward()
{
    
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif
    
    // initialization
    
    for (int k = 0; k < K; k++)
    {
        Fb[k].clear();
        Fb[k].resize(SIZE, RealT(NEG_INF));
        Fb[k][SIZE-1] = RealT(0);
    }
    
    for (int i = LX; i >= 1; i--)
    {
        for (int j = LY; j >= 1; j--)
        {
            const int ij = i*(LY+1)+j;
            const int i1j = ij-(LY+1);
            const int ij1 = ij-1;
            const int i1j1 = ij-(LY+1)-1;
            
            Fast_LogPlusEquals(Fb[MATCH][i1j1], Fb[MATCH][ij] + ScoreMatch(i,j,MATCH));
            if (i > 1 || j > 1)
            {
                Fast_LogPlusEquals(Fb[INS_X][i1j1], Fb[MATCH][ij] + ScoreMatch(i,j,INS_X));
                Fast_LogPlusEquals(Fb[INS_Y][i1j1], Fb[MATCH][ij] + ScoreMatch(i,j,INS_Y));
#if PARAMS_DOUBLE_AFFINE
                Fast_LogPlusEquals(Fb[INS2_X][i1j1], Fb[MATCH][ij] + ScoreMatch(i,j,INS2_X));
                Fast_LogPlusEquals(Fb[INS2_Y][i1j1], Fb[MATCH][ij] + ScoreMatch(i,j,INS2_Y));
#endif
            }
            
            Fast_LogPlusEquals(Fb[MATCH][i1j], Fb[INS_X][ij] + ScoreInsertX(i,j,MATCH));
            Fast_LogPlusEquals(Fb[INS_X][i1j], Fb[INS_X][ij] + ScoreInsertX(i,j,INS_X));
#if !FORCE_UNIQUE_PARSES
            Fast_LogPlusEquals(Fb[INS_Y][i1j], Fb[INS_X][ij] + ScoreInsertX(i,j,INS_Y));
#endif
            
            Fast_LogPlusEquals(Fb[MATCH][ij1], Fb[INS_Y][ij] + ScoreInsertY(i,j,MATCH));
            Fast_LogPlusEquals(Fb[INS_X][ij1], Fb[INS_Y][ij] + ScoreInsertY(i,j,INS_X));
            Fast_LogPlusEquals(Fb[INS_Y][ij1], Fb[INS_Y][ij] + ScoreInsertY(i,j,INS_Y));
            
#if PARAMS_DOUBLE_AFFINE
            Fast_LogPlusEquals(Fb[MATCH][i1j], Fb[INS2_X][ij] + ScoreInsert2X(i,j,MATCH));
            Fast_LogPlusEquals(Fb[INS2_X][i1j], Fb[INS2_X][ij] + ScoreInsert2X(i,j,INS2_X));
#if !FORCE_UNIQUE_PARSES
            Fast_LogPlusEquals(Fb[INS2_Y][i1j], Fb[INS2_X][ij] + ScoreInsert2X(i,j,INS2_Y));
#endif
            
            Fast_LogPlusEquals(Fb[MATCH][ij1], Fb[INS2_Y][ij] + ScoreInsert2Y(i,j,MATCH));
            Fast_LogPlusEquals(Fb[INS2_X][ij1], Fb[INS2_Y][ij] + ScoreInsert2Y(i,j,INS2_X));
            Fast_LogPlusEquals(Fb[INS2_Y][ij1], Fb[INS2_Y][ij] + ScoreInsert2Y(i,j,INS2_Y));
#endif
        }
    }
    
    for (int i = LX; i >= 1; i--) Fast_LogPlusEquals(Fb[INS_X][(i-1)*(LY+1)+0], Fb[INS_X][i*(LY+1)+0] + ScoreInsertX(i,0,INS_X));
    for (int j = LY; j >= 1; j--) Fast_LogPlusEquals(Fb[INS_Y][0*(LY+1)+(j-1)], Fb[INS_Y][0*(LY+1)+j] + ScoreInsertY(0,j,INS_Y));
#if PARAMS_DOUBLE_AFFINE
    for (int i = LX; i >= 1; i--) Fast_LogPlusEquals(Fb[INS2_X][(i-1)*(LY+1)+0], Fb[INS2_X][i*(LY+1)+0] + ScoreInsert2X(i,0,INS2_X));
    for (int j = LY; j >= 1; j--) Fast_LogPlusEquals(Fb[INS2_Y][0*(LY+1)+(j-1)], Fb[INS2_Y][0*(LY+1)+j] + ScoreInsert2Y(0,j,INS2_Y));
#endif
    
#if SHOW_TIMINGS
    std::cerr << "Backward score: " << ComputeBackwardLogPartitionCoefficient()
              << " (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeLogPartitionCoefficient()
//
// Return partition coefficient.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ComputeLogPartitionCoefficient() const
{
    return ComputeForwardLogPartitionCoefficient();
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeForwardLogPartitionCoefficient()
//
// Return partition coefficient using the forward algorithm.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ComputeForwardLogPartitionCoefficient() const
{
    RealT ret = Ff[MATCH][SIZE-1];
    for (int k = 1; k < K; k++)
        Fast_LogPlusEquals(ret, Ff[k][SIZE-1]);
    return ret;
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeBackwardLogPartitionCoefficient()
//
// Return partition coefficient using the backward algorithm.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ComputeBackwardLogPartitionCoefficient() const
{
    RealT ret = Fb[MATCH][0];
    for (int k = 1; k < K; k++)
        Fast_LogPlusEquals(ret, Fb[k][0]);
    return ret;
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeFeatureCountExpectations()
// 
// Combine the results of the forward and backward algorithms
// in order to compute feature count expectations.
/////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> InferenceEngine<RealT>::ComputeFeatureCountExpectations()
{
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif
    
    const RealT Z = ComputeLogPartitionCoefficient();
    
    // initialization

    ClearCounts();
    for (int i = 1; i <= LX; i++) CountInsertX(i,0,INS_X,Fast_Exp(Ff[INS_X][(i-1)*(LY+1)+0] + ScoreInsertX(i,0,INS_X) + Fb[INS_X][i*(LY+1)+0] - Z));
    for (int j = 1; j <= LY; j++) CountInsertY(0,j,INS_Y,Fast_Exp(Ff[INS_Y][0*(LY+1)+(j-1)] + ScoreInsertY(0,j,INS_Y) + Fb[INS_Y][0*(LY+1)+j] - Z));
#if PARAMS_DOUBLE_AFFINE
    for (int i = 1; i <= LX; i++) CountInsert2X(i,0,INS2_X,Fast_Exp(Ff[INS2_X][(i-1)*(LY+1)+0] + ScoreInsert2X(i,0,INS2_X) + Fb[INS2_X][i*(LY+1)+0] - Z));
    for (int j = 1; j <= LY; j++) CountInsert2Y(0,j,INS2_Y,Fast_Exp(Ff[INS2_Y][0*(LY+1)+(j-1)] + ScoreInsert2Y(0,j,INS2_Y) + Fb[INS2_Y][0*(LY+1)+j] - Z));
#endif
    
    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            const int ij = i*(LY+1)+j;
            const int i1j = ij-(LY+1);
            const int ij1 = ij-1;
            const int i1j1 = ij-(LY+1)-1;
            
            CountMatch(i,j,MATCH,Fast_Exp(Ff[MATCH][i1j1] + ScoreMatch(i,j,MATCH) + Fb[MATCH][ij] - Z));
            if (i > 1 || j > 1)
            {
                CountMatch(i,j,INS_X,Fast_Exp(Ff[INS_X][i1j1] + ScoreMatch(i,j,INS_X) + Fb[MATCH][ij] - Z));
                CountMatch(i,j,INS_Y,Fast_Exp(Ff[INS_Y][i1j1] + ScoreMatch(i,j,INS_Y) + Fb[MATCH][ij] - Z));
#if PARAMS_DOUBLE_AFFINE
                CountMatch(i,j,INS2_X,Fast_Exp(Ff[INS2_X][i1j1] + ScoreMatch(i,j,INS2_X) + Fb[MATCH][ij] - Z));
                CountMatch(i,j,INS2_Y,Fast_Exp(Ff[INS2_Y][i1j1] + ScoreMatch(i,j,INS2_Y) + Fb[MATCH][ij] - Z));
#endif
            }
            
            CountInsertX(i,j,MATCH,Fast_Exp(Ff[MATCH][i1j] + ScoreInsertX(i,j,MATCH) + Fb[INS_X][ij] - Z));
            CountInsertX(i,j,INS_X,Fast_Exp(Ff[INS_X][i1j] + ScoreInsertX(i,j,INS_X) + Fb[INS_X][ij] - Z));
#if !FORCE_UNIQUE_PARSES
            CountInsertX(i,j,INS_Y,Fast_Exp(Ff[INS_Y][i1j] + ScoreInsertX(i,j,INS_Y) + Fb[INS_X][ij] - Z));
#endif
            
            CountInsertY(i,j,MATCH,Fast_Exp(Ff[MATCH][ij1] + ScoreInsertY(i,j,MATCH) + Fb[INS_Y][ij] - Z));
            CountInsertY(i,j,INS_X,Fast_Exp(Ff[INS_X][ij1] + ScoreInsertY(i,j,INS_X) + Fb[INS_Y][ij] - Z));
            CountInsertY(i,j,INS_Y,Fast_Exp(Ff[INS_Y][ij1] + ScoreInsertY(i,j,INS_Y) + Fb[INS_Y][ij] - Z));
            
#if PARAMS_DOUBLE_AFFINE
            CountInsert2X(i,j,MATCH,Fast_Exp(Ff[MATCH][i1j] + ScoreInsert2X(i,j,MATCH) + Fb[INS2_X][ij] - Z));
            CountInsert2X(i,j,INS2_X,Fast_Exp(Ff[INS2_X][i1j] + ScoreInsert2X(i,j,INS2_X) + Fb[INS2_X][ij] - Z));
#if !FORCE_UNIQUE_PARSES
            CountInsert2X(i,j,INS2_Y,Fast_Exp(Ff[INS2_Y][i1j] + ScoreInsert2X(i,j,INS2_Y) + Fb[INS2_X][ij] - Z));
#endif
            
            CountInsert2Y(i,j,MATCH,Fast_Exp(Ff[MATCH][ij1] + ScoreInsert2Y(i,j,MATCH) + Fb[INS2_Y][ij] - Z));
            CountInsert2Y(i,j,INS2_X,Fast_Exp(Ff[INS2_X][ij1] + ScoreInsert2Y(i,j,INS2_X) + Fb[INS2_Y][ij] - Z));
            CountInsert2Y(i,j,INS2_Y,Fast_Exp(Ff[INS2_Y][ij1] + ScoreInsert2Y(i,j,INS2_Y) + Fb[INS2_Y][ij] - Z));
#endif
            
        }
    }
    
#if SHOW_TIMINGS
    std::cerr << "Feature expectations (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif

    return GetCounts();
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::ComputePosterior()
// 
// Combine the results of the forward and backward algorithms
// in order to compute posterior probabilities of matches.
/////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputePosterior()
{
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif
    
    const RealT Z = ComputeLogPartitionCoefficient();

    // initialization

    posterior.clear(); posterior.resize(SIZE, RealT(0));
    
    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            const int ij = i*(LY+1)+j;
            const int i1j1 = ij-(LY+1)-1;
            for (int k = 0; k < K; k++)
            {
                if (k == MATCH || i > 1 || j > 1)
                    posterior[ij] += Fast_Exp(Ff[k][i1j1] + ScoreMatch(i,j,k) + Fb[MATCH][ij] - Z);
            }
        }
    }

    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            posterior[i*(LY+1)+j] = Clip(posterior[i*(LY+1)+j], RealT(0), RealT(1));
        }
    }
    
#if SHOW_TIMINGS
    std::cerr << "Compute posterior (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::PredictAlignmentPosterior()
//
// Use posterior decoding to predict alignment.
/////////////////////////////////////////////////////////////////

template<class RealT>
std::string InferenceEngine<RealT>::PredictAlignmentPosterior(const RealT gamma)
{
    return PredictAlignmentPosterior(gamma, &posterior[0]);
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::PredictAlignmentPosterior()
//
// Use posterior decoding to predict alignment.
/////////////////////////////////////////////////////////////////

template<class RealT>
std::string InferenceEngine<RealT>::PredictAlignmentPosterior(const RealT gamma, const RealT *posterior)
{
    Assert(gamma > 0, "Non-negative gamma expected.");

    // compute insert scores
    
    std::vector<RealT> insert_x(LX+1, RealT(1));
    std::vector<RealT> insert_y(LY+1, RealT(1));
    insert_x[0] = insert_y[0] = 0;
    
    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            const int ij = i*(LY+1)+j;
            insert_x[i] -= posterior[ij];
            insert_y[j] -= posterior[ij];
        }
    }
    
    for (int i = 1; i <= LX; i++) insert_x[i] = std::max(insert_x[i], RealT(0)) / (RealT(1e-10) + gamma);
    for (int j = 1; j <= LY; j++) insert_y[j] = std::max(insert_y[j], RealT(0)) / (RealT(1e-10) + gamma);
    
    // initialization

    Ft[MATCH].clear();
    Ft[MATCH].resize(SIZE, -1);
    Fv[MATCH].clear();
    Fv[MATCH].resize(SIZE, RealT(NEG_INF));
    Fv[MATCH][0] = RealT(0);
    
    // dynamic programming
    
    for (int i = 1; i <= LX; i++) { Fv[MATCH][i*(LY+1)+0] = Fv[MATCH][(i-1)*(LY+1)+0] + insert_x[i]; Ft[MATCH][i*(LY+1)+0] = INS_X; }
    for (int j = 1; j <= LY; j++) { Fv[MATCH][0*(LY+1)+j] = Fv[MATCH][0*(LY+1)+(j-1)] + insert_y[j]; Ft[MATCH][0*(LY+1)+j] = INS_Y; }
    
    for (int i = 1; i <= LX; i++)
    {
        for (int j = 1; j <= LY; j++)
        {
            const int ij = i*(LY+1)+j;
            const int i1j = ij-(LY+1);
            const int ij1 = ij-1;
            const int i1j1 = ij-(LY+1)-1;
            
            UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[MATCH][i1j1] + posterior[ij], MATCH);
            UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[MATCH][i1j] + insert_x[i], INS_X);
            UPDATE_MAX(Fv[MATCH][ij], Ft[MATCH][ij], Fv[MATCH][ij1] + insert_y[j], INS_Y);
        }
    }
    
    // traceback
    
    int i = LX;
    int j = LY;
    
    std::string edit_string;
    while (i > 0 || j > 0)
    {
        switch (Ft[MATCH][i*(LY+1)+j])
        {
            case MATCH: edit_string.push_back('M'); i--; j--; break;
            case INS_X: edit_string.push_back('X'); i--; break;
            case INS_Y: edit_string.push_back('Y'); j--; break;
        }
    }
    
    edit_string.push_back('@');
    std::reverse(edit_string.begin(), edit_string.end());
    
    return edit_string;
}

/////////////////////////////////////////////////////////////////
// InferenceEngine::GetPosterior()
//
// Return posterior probability matrix, thresholded.
/////////////////////////////////////////////////////////////////

template<class RealT>
RealT *InferenceEngine<RealT>::GetPosterior(const RealT posterior_cutoff) const
{
    RealT *ret = new RealT[SIZE];
    for (int i = 0; i < SIZE; i++)
        ret[i] = (posterior[i] >= posterior_cutoff ? posterior[i] : RealT(0));
    return ret;
}

/*









//////////////////////////////////////////////////////////////////////
// InferenceEngine::GetViterbiScore()
//
// Return Viterbi score for a sequence.
//////////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::GetViterbiScore() const
{
    return F5v[L];
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::PredictPairingsViterbi()
// 
// Use Viterbi decoding to predict pairings.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<int> InferenceEngine<RealT>::PredictPairingsViterbi() const
{
    std::vector<int> solution(L+1,SStruct::UNPAIRED);
    solution[0] = SStruct::Sequence::UNKNOWN;
    //return solution;
    
    std::queue<triple<const int *,int,int> > traceback_queue;
    traceback_queue.push(make_triple(&F5t[0], 0, L));
    
    while (!traceback_queue.empty())
    {
        triple<const int *,int,int> t = traceback_queue.front();
        traceback_queue.pop();
        const int *V = t.first;
        const int i = t.second;
        const int j = t.third;
        
        std::pair<int,int> traceback = DecodeTraceback(V == &F5t[0] ? V[j] : V[offset[i]+j]);
        
        //std::cerr << (V == FCt ? "FC " : V == FMt ? "FM " : V == FM1t ? "FM1 " : "F5 ");
        //std::cerr << i << " " << j << ": " << traceback.first << " " << traceback.second << std::endl;
        
        switch (traceback.first)
        {
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
            case TB_FN_HAIRPIN: 
                break;
            case TB_FN_SINGLE: 
            {
                const int p = i + traceback.second / (C_MAX_SINGLE_LENGTH+1);
                const int q = j - traceback.second % (C_MAX_SINGLE_LENGTH+1);
                solution[p+1] = q;
                solution[q] = p+1;
                traceback_queue.push(make_triple(&FCt[0], p+1, q-1));
            }
            break;
            case TB_FN_BIFURCATION:
            {
                const int k = traceback.second;
                traceback_queue.push(make_triple(&FM1t[0], i, k));
                traceback_queue.push(make_triple(&FMt[0], k, j));
            }
            break;
            case TB_FE_STACKING: 
            {
                solution[i+1] = j;
                solution[j] = i+1;
                traceback_queue.push(make_triple(&FEt[0], i+1, j-1));
            }
            break;
            case TB_FE_FN: 
            {
                traceback_queue.push(make_triple(&FNt[0], i, j));
            }
            break;
            case TB_FC_FN:
            {
                traceback_queue.push(make_triple(&FNt[0], i, j));
            }
            break;
            case TB_FC_HELIX:
            {
                const int m = traceback.second;
                for (int k = 2; k <= m; k++)
                {
                    solution[i+k-1] = j-k+2;
                    solution[j-k+2] = i+k-1;
                }
                traceback_queue.push(make_triple(&FNt[0], i+m-1, j-m+1));
            }
            break;
            case TB_FC_FE:
            {
                const int m = D_MAX_HELIX_LENGTH;
                for (int k = 2; k <= m; k++)
                {
                    solution[i+k-1] = j-k+2;
                    solution[j-k+2] = i+k-1;
                }
                traceback_queue.push(make_triple(&FEt[0], i+m-1, j-m+1));
            }
            break;
#else
            case TB_FC_HAIRPIN: 
                break;
            case TB_FC_SINGLE: 
            {
                const int p = i + traceback.second / (C_MAX_SINGLE_LENGTH+1);
                const int q = j - traceback.second % (C_MAX_SINGLE_LENGTH+1);
                solution[p+1] = q;
                solution[q] = p+1;
                traceback_queue.push(make_triple(&FCt[0], p+1, q-1));
            }
            break;
            case TB_FC_BIFURCATION:
            {
                const int k = traceback.second;
                traceback_queue.push(make_triple(&FM1t[0], i, k));
                traceback_queue.push(make_triple(&FMt[0], k, j));
            }
            break;
#endif
            case TB_FM1_PAIRED:
            {
                solution[i+1] = j;
                solution[j] = i+1;
                traceback_queue.push(make_triple(&FCt[0], i+1, j-1));
            }
      break;
            case TB_FM1_UNPAIRED:
            {
                traceback_queue.push(make_triple(&FM1t[0], i+1, j));
            }
            break;
            case TB_FM_BIFURCATION:
            {
                const int k = traceback.second;
                traceback_queue.push(make_triple(&FM1t[0], i, k));
                traceback_queue.push(make_triple(&FMt[0], k, j));
            }
            break;
            case TB_FM_UNPAIRED:
            {
                traceback_queue.push(make_triple(&FMt[0], i, j-1));
            }
            break;
            case TB_FM_FM1: 
            {
                traceback_queue.push(make_triple(&FM1t[0], i, j));
            }
            break;
            case TB_F5_ZERO:
                break;
            case TB_F5_UNPAIRED:
            {
                traceback_queue.push(make_triple(&F5t[0], 0, j-1));
            }
            break;
            case TB_F5_BIFURCATION:
            {
                const int k = traceback.second;
                solution[k+1] = j;
                solution[j] = k+1;
                traceback_queue.push(make_triple(&F5t[0], 0, k));
                traceback_queue.push(make_triple(&FCt[0], k+1, j-1));
            }
            break;
            default:
                Assert(false, "Bad traceback.");
        }
    }
    
    return solution;
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeViterbiFeatureCounts()
// 
// Use feature counts from Viterbi decoding.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> InferenceEngine<RealT>::ComputeViterbiFeatureCounts()
{
    std::queue<triple<int *,int,int> > traceback_queue;
    traceback_queue.push(make_triple(&F5t[0], 0, L));

    ClearCounts();
    
    while (!traceback_queue.empty())
    {
        triple<int *,int,int> t = traceback_queue.front();
        traceback_queue.pop();
        const int *V = t.first;
        const int i = t.second;
        const int j = t.third;
        
        std::pair<int,int> traceback = DecodeTraceback (V == &F5t[0] ? V[j] : V[offset[i]+j]);
        
        //std::cout << (V == FCt ? "FC " : V == FMt ? "FM " : V == FM1t ? "FM1 " : "F5 ");
        //std::cout << i << " " << j << ": " << traceback.first << " " << traceback.second << std::endl;
        
        switch (traceback.first)
        {
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
            case TB_FN_HAIRPIN: 
                CountHairpin(i,j,RealT(1));
                break;
            case TB_FN_SINGLE: 
            {
                const int p = i + traceback.second / (C_MAX_SINGLE_LENGTH+1);
                const int q = j - traceback.second % (C_MAX_SINGLE_LENGTH+1);
                CountSingle(i,j,p,q,RealT(1));
                traceback_queue.push(make_triple(&FCt[0], p+1, q-1));
            }
            break;
            case TB_FN_BIFURCATION:
            {
                const int k = traceback.second;
                CountJunctionA(i,j,RealT(1));
                CountMultiPaired(RealT(1));
                CountMultiBase(RealT(1));
                traceback_queue.push(make_triple(&FM1t[0], i, k));
                traceback_queue.push(make_triple(&FMt[0], k, j));
            }
            break;
            case TB_FE_STACKING: 
            {
                CountBasePair(i+1,j,RealT(1));
                CountHelixStacking(i,j+1,RealT(1));
                traceback_queue.push(make_triple(&FEt[0], i+1, j-1));
            }
            break;
            case TB_FE_FN: 
            {
                traceback_queue.push(make_triple(&FNt[0], i, j));
            }
            break;
            case TB_FC_FN:
            {
                CountIsolated(RealT(1));
                traceback_queue.push(make_triple(&FNt[0], i, j));
            }
            break;
            case TB_FC_HELIX:
            {
                const int m = traceback.second;
                CountHelix(i-1,j+1,m,RealT(1));
                traceback_queue.push(make_triple(&FNt[0], i+m-1, j-m+1));
            }
            break;
            case TB_FC_FE:
            {
                const int m = D_MAX_HELIX_LENGTH;
                CountHelix(i-1,j+1,m,RealT(1));
                traceback_queue.push(make_triple(&FEt[0], i+m-1, j-m+1));
            }
            break;
#else
            case TB_FC_HAIRPIN: 
                CountHairpin(i,j,RealT(1));
                break;
            case TB_FC_SINGLE: 
            {
                const int p = i + traceback.second / (C_MAX_SINGLE_LENGTH+1);
                const int q = j - traceback.second % (C_MAX_SINGLE_LENGTH+1);

                if (p == i && q == j)
                {
                    CountBasePair(i+1,j,RealT(1));
                    CountHelixStacking(i,j+1,RealT(1));
                }
                else
                {
                    CountSingle(i,j,p,q,RealT(1));
                }
                
                traceback_queue.push(make_triple(&FCt[0], p+1, q-1));
            }
            break;
            case TB_FC_BIFURCATION:
            {
                const int k = traceback.second;
                CountJunctionA(i,j,RealT(1));
                CountMultiPaired(RealT(1));
                CountMultiBase(RealT(1));
                traceback_queue.push(make_triple(&FM1t[0], i, k));
                traceback_queue.push(make_triple(&FMt[0], k, j));
            }
            break;
#endif
            case TB_FM1_PAIRED:
            {
                CountJunctionA(j,i,RealT(1));
                CountMultiPaired(RealT(1));
                CountBasePair(i+1,j,RealT(1));
                traceback_queue.push(make_triple(&FCt[0], i+1, j-1));
            }
            break;
            case TB_FM1_UNPAIRED:
            {
                CountMultiUnpaired(i+1,RealT(1));
                traceback_queue.push(make_triple(&FM1t[0], i+1, j));
            }
            break;
            case TB_FM_BIFURCATION:
            {
                const int k = traceback.second;
                traceback_queue.push(make_triple(&FM1t[0], i, k));
                traceback_queue.push(make_triple(&FMt[0], k, j));
            }
            break;
            case TB_FM_UNPAIRED:
            {
                CountMultiUnpaired(j,RealT(1));
                traceback_queue.push(make_triple(&FMt[0], i, j-1));
            }
            break;
            case TB_FM_FM1: 
                traceback_queue.push(make_triple(&FM1t[0], i, j));
                break;
            case TB_F5_ZERO:
                break;
            case TB_F5_UNPAIRED:
                CountExternalUnpaired(j,RealT(1));
                traceback_queue.push(make_triple(&F5t[0], 0, j-1));
                break;
            case TB_F5_BIFURCATION:
            {
                const int k = traceback.second;
                CountExternalPaired(RealT(1));
                CountBasePair(k+1,j,RealT(1));
                CountJunctionA(j,k,RealT(1));
                traceback_queue.push(make_triple(&F5t[0], 0, k));
                traceback_queue.push(make_triple(&FCt[0], k+1, j-1));
            }
            break;
            default:
                Assert(false, "Bad traceback.");
        }
    }

    return GetCounts();
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeInside()
//
// Run inside algorithm.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputeInside()
{
        
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif
    
    // initialization

    F5i.clear(); F5i.resize(L+1, RealT(NEG_INF));
    FCi.clear(); FCi.resize(SIZE, RealT(NEG_INF));
    FMi.clear(); FMi.resize(SIZE, RealT(NEG_INF));
    FM1i.clear(); FM1i.resize(SIZE, RealT(NEG_INF));
    
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
    FEi.clear(); FEi.resize(SIZE, RealT(NEG_INF));
    FNi.clear(); FNi.resize(SIZE, RealT(NEG_INF));
#endif

    for (int i = L; i >= 0; i--)
    {
        for (int j = i; j <= L; j++)
        {
            
            // FM2[i,j] = SUM (i<k<j : FM1[i,k] + FM[k,j])
            
            RealT FM2i = RealT(NEG_INF);
            
#if SIMPLE_FM2
            
            for (int k = i+1; k < j; k++)
                Fast_LogPlusEquals(FM2i, FM1i[offset[i]+k] + FMi[offset[k]+j]);
            
#else
            
            if (i+2 <= j)
            {
                const RealT *p1 = &(FM1i[offset[i]+i+1]);
                const RealT *p2 = &(FMi[offset[i+1]+j]);
                for (register int k = i+1; k < j; k++)
                {
                    Fast_LogPlusEquals(FM2i, (*p1) + (*p2));
                    ++p1;
                    p2 += L-k;
                }
            }
            
#endif
            
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR

            // FN[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           and the next interaction is not a stacking pair
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                RealT sum_i = RealT(NEG_INF);
                
                // compute ScoreHairpin(i,j)
                
                if (allow_unpaired[offset[i]+j] && j-i >= C_MIN_HAIRPIN_LENGTH)
                    Fast_LogPlusEquals(sum_i, ScoreHairpin(i,j));
                
                // compute SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                
                for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                {
                    if (p > i && !allow_unpaired_position[p]) break;
                    int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                    for (int q = j; q >= q_min; q--)
                    {
                        if (q < j && !allow_unpaired_position[q+1]) break;
                        if (!allow_paired[offset[p+1]+q]) continue;
                        if (i == p && j == q) continue;
                        
                        Fast_LogPlusEquals(sum_i, ScoreSingle(i,j,p,q) + FCi[offset[p+1]+q-1]);
                    }
                }
                
#else

                if (i+2 <= j)                
                {
                    RealT score_other = ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        const RealT *FCptr = &(FCi[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            if (i == p && j == q) continue;
                            
                            RealT score = (score_other + cache_score_single[p-i][j-q].first + FCptr[q] + ScoreBasePair(p+1,q) +
                                           ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                            
                            Fast_LogPlusEquals(sum_i, score);
                        }
                    }
                }
#endif
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c)
                
                Fast_LogPlusEquals(sum_i, FM2i + ScoreJunctionA(i,j) + ScoreMultiPaired() + ScoreMultiBase());
                
                FNi[offset[i]+j] = sum_i;
            }
            
            // FE[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) ... (i-D+1,j+D) are 
            //           already base-paired
            //
            //         = SUM [ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]   if i+2<=j,
            //                FN(i,j)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT sum_i = RealT(NEG_INF);
                
                // compute ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]
                
                if (i+2 <= j && allow_paired[offset[i+1]+j])
                {
                    Fast_LogPlusEquals(sum_i, ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) + FEi[offset[i+1]+j-1]);
                }
                
                // compute FN(i,j)

                Fast_LogPlusEquals(sum_i, FNi[offset[i]+j]);
                
                FEi[offset[i]+j] = sum_i;
            }
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           but (i-1,j+2) are not
            //
            //         = SUM [ScoreIsolated() + FN(i,j),
            //                SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k)),
            //                FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT sum_i = RealT(NEG_INF);
                
                // compute ScoreIsolated() + FN(i,j)
                
                Fast_LogPlusEquals(sum_i, ScoreIsolated() + FNi[offset[i]+j]);
                
                // compute SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k))
                
                bool allowed = true;
                for (int k = 2; k < D_MAX_HELIX_LENGTH; k++)
                {
                    if (i + 2*k - 2 > j) break;
                    if (!allow_paired[offset[i+k-1]+j-k+2]) { allowed = false; break; }
                    Fast_LogPlusEquals(sum_i, ScoreHelix(i-1,j+1,k) + FNi[offset[i+k-1]+j-k+1]);
                }
                
                // compute FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
                
                if (i + 2*D_MAX_HELIX_LENGTH-2 <= j)
                {
                    if (allowed && allow_paired[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+2])
                        Fast_LogPlusEquals(sum_i, ScoreHelix(i-1,j+1,D_MAX_HELIX_LENGTH) + FEi[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+1]);
                }
                
                FCi[offset[i]+j] = sum_i;
            }
            
#else
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT sum_i = RealT(NEG_INF);
                
                // compute ScoreHairpin(i,j)
                
                if (allow_unpaired[offset[i]+j] && j-i >= C_MIN_HAIRPIN_LENGTH)
                    Fast_LogPlusEquals(sum_i, ScoreHairpin(i,j));
                
                // compute SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                
                for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                {
                    if (p > i && !allow_unpaired_position[p]) break;
                    int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                    for (int q = j; q >= q_min; q--)
                    {
                        if (q < j && !allow_unpaired_position[q+1]) break;
                        if (!allow_paired[offset[p+1]+q]) continue;

                        Fast_LogPlusEquals(sum_i,
                                           FCi[offset[p+1]+q-1] +
                                           (p == i && q == j ? ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) : ScoreSingle(i,j,p,q)));
                    }
                }
                
#else

                {
                    RealT score_helix = (i+2 <= j ? ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) : 0);
                    RealT score_other = ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        const RealT *FCptr = &(FCi[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            
                            RealT score = (p == i && q == j) ?
                                (score_helix + FCptr[q]) :
                                (score_other + cache_score_single[p-i][j-q].first + FCptr[q] + ScoreBasePair(p+1,q) +
                                 ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                            
                            Fast_LogPlusEquals(sum_i, score);
                        }
                    }
                }
#endif

                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c)
                
                Fast_LogPlusEquals(sum_i, FM2i + ScoreJunctionA(i,j) + ScoreMultiPaired() + ScoreMultiBase());
                
                FCi[offset[i]+j] = sum_i;
            }
            
#endif
            
            // FM1[i,j] = optimal energy for substructure belonging to a
            //            multibranch loop containing a (k+1,j) base pair
            //            preceded by 5' unpaired nucleotides from i to k
            //            for some i <= k <= j-2
            //
            //          = SUM [FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)  if i+2<=j,
            //                 FM1[i+1,j] + b                                          if i+2<=j]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                
                RealT sum_i = RealT(NEG_INF);
                
                // compute FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)
                
                if (allow_paired[offset[i+1]+j])
                    Fast_LogPlusEquals(sum_i, FCi[offset[i+1]+j-1] + ScoreJunctionA(j,i) + ScoreMultiPaired() + ScoreBasePair(i+1,j));
                
                // compute FM1[i+1,j] + b
                
                if (allow_unpaired_position[i+1])
                    Fast_LogPlusEquals(sum_i, FM1i[offset[i+1]+j] + ScoreMultiUnpaired(i+1));
                
                FM1i[offset[i]+j] = sum_i;
            }
            
            // FM[i,j] = optimal energy for substructure belonging to a
            //           multibranch loop which contains at least one 
            //           helix
            //
            //         = SUM [SUM (i<k<j : FM1[i,k] + FM[k,j]),
            //                FM[i,j-1] + b,
            //                FM1[i,j]]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                
                RealT sum_i = RealT(NEG_INF);
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j])
                
                Fast_LogPlusEquals(sum_i, FM2i);
                
                // compute FM[i,j-1] + b
                
                if (allow_unpaired_position[j])
                    Fast_LogPlusEquals(sum_i, FMi[offset[i]+j-1] + ScoreMultiUnpaired(j));
                
                // compute FM1[i,j]
                
                Fast_LogPlusEquals(sum_i, FM1i[offset[i]+j]);
                
                FMi[offset[i]+j] = sum_i;
            }
        }
    }
    
    F5i[0] = RealT(0);
    for (int j = 1; j <= L; j++)
    {
        
        // F5[j] = optimal energy for substructure between positions 0 and j
        //         (or 0 if j = 0)
        //
        //       = SUM [F5[j-1] + ScoreExternalUnpaired(),
        //              SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))]
        
        RealT sum_i = RealT(NEG_INF);
        
        // compute F5[j-1] + ScoreExternalUnpaired()
        
        if (allow_unpaired_position[j])
            Fast_LogPlusEquals(sum_i, F5i[j-1] + ScoreExternalUnpaired(j));
        
        // compute SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))
        
        for (int k = 0; k < j; k++)
            if (allow_paired[offset[k+1]+j])
                Fast_LogPlusEquals(sum_i, F5i[k] + FCi[offset[k+1]+j-1] + ScoreExternalPaired() + ScoreBasePair(k+1,j) + ScoreJunctionA(j,k));
        
        F5i[j] = sum_i;
    }

#if SHOW_TIMINGS
    std::cerr << "Inside score: " << F5i[L] << " (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeOutside()
//
// Run outside algorithm.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputeOutside()
{
    
#if SHOW_TIMINGS    
    double starting_time = GetSystemTime();
#endif
    
    // initialization
    
    F5o.clear(); F5o.resize(L+1, RealT(NEG_INF));
    FCo.clear(); FCo.resize(SIZE, RealT(NEG_INF));
    FMo.clear(); FMo.resize(SIZE, RealT(NEG_INF));
    FM1o.clear(); FM1o.resize(SIZE, RealT(NEG_INF));
    
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
    FEo.clear(); FEo.resize(SIZE, RealT(NEG_INF));
    FNo.clear(); FNo.resize(SIZE, RealT(NEG_INF));
#endif
    
    F5o[L] = RealT(0);  
    for (int j = L; j >= 1; j--)
    {
        
        // F5[j] = optimal energy for substructure between positions 0 and j
        //         (or 0 if j = 0)
        //
        //       = SUM [F5[j-1] + ScoreExternalUnpaired(),
        //              SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))]
        
        // compute F5[j-1] + ScoreExternalUnpaired()
        
        if (allow_unpaired_position[j])
            Fast_LogPlusEquals(F5o[j-1], F5o[j] + ScoreExternalUnpaired(j));
        
        // compute SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))
        
        {
            for (int k = 0; k < j; k++)
            {
                if (allow_paired[offset[k+1]+j])
                {
                    RealT temp = F5o[j] + ScoreExternalPaired() + ScoreBasePair(k+1,j) + ScoreJunctionA(j,k);
                    Fast_LogPlusEquals(F5o[k], temp + FCi[offset[k+1]+j-1]);
                    Fast_LogPlusEquals(FCo[offset[k+1]+j-1], temp + F5i[k]);
                }
            }
        }
    }
    
    for (int i = 0; i <= L; i++)
    {
        for (int j = L; j >= i; j--)
        {
            RealT FM2o = RealT(NEG_INF);
            
            // FM[i,j] = optimal energy for substructure belonging to a
            //           multibranch loop which contains at least one 
            //           helix
            //
            //         = SUM [SUM (i<k<j : FM1[i,k] + FM[k,j]),
            //                FM[i,j-1] + b,
            //                FM1[i,j]]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j])
                
                Fast_LogPlusEquals(FM2o, FMo[offset[i]+j]);
                
                // compute FM[i,j-1] + b
                
                if (allow_unpaired_position[j])
                    Fast_LogPlusEquals(FMo[offset[i]+j-1], FMo[offset[i]+j] + ScoreMultiUnpaired(j));
                
                // compute FM1[i,j]
                
                Fast_LogPlusEquals(FM1o[offset[i]+j], FMo[offset[i]+j]);
            }
            
            // FM1[i,j] = optimal energy for substructure belonging to a
            //            multibranch loop containing a (k+1,j) base pair
            //            preceded by 5' unpaired nucleotides from i to k
            //            for some i <= k <= j-2
            //
            //          = SUM [FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)  if i+2<=j,
            //                 FM1[i+1,j] + b                                          if i+2<=j]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                // compute FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)
                
                if (allow_paired[offset[i+1]+j])
                    Fast_LogPlusEquals(FCo[offset[i+1]+j-1], FM1o[offset[i]+j] + ScoreJunctionA(j,i) + ScoreMultiPaired() + ScoreBasePair(i+1,j));
                
                // compute FM1[i+1,j] + b
                
                if (allow_unpaired_position[i+1])
                    Fast_LogPlusEquals(FM1o[offset[i+1]+j], FM1o[offset[i]+j] + ScoreMultiUnpaired(i+1));
                
            }
            
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           but (i-1,j+2) are not
            //
            //         = SUM [ScoreIsolated() + FN(i,j),
            //                SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k)),
            //                FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                // compute ScoreIsolated() + FN(i,j)
                
                Fast_LogPlusEquals(FNo[offset[i]+j], ScoreIsolated() + FCo[offset[i]+j]);
                
                // compute SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k))
                
                bool allowed = true;
                for (int k = 2; k < D_MAX_HELIX_LENGTH; k++)
                {
                    if (i + 2*k - 2 > j) break;
                    if (!allow_paired[offset[i+k-1]+j-k+2]) { allowed = false; break; }
                    Fast_LogPlusEquals(FNo[offset[i+k-1]+j-k+1], ScoreHelix(i-1,j+1,k) + FCo[offset[i]+j]);
                }
                
                // compute FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
                
                if (i + 2*D_MAX_HELIX_LENGTH-2 <= j)
                {
                    if (allowed && allow_paired[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+2])
                        Fast_LogPlusEquals(FEo[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+1],
                                           ScoreHelix(i-1,j+1,D_MAX_HELIX_LENGTH) + FCo[offset[i]+j]);
                }
            }
            
            // FE[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) ... (i-D+1,j+D) are 
            //           already base-paired
            //
            //         = SUM [ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]   if i+2<=j,
            //                FN(i,j)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                // compute ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]
                
                if (i+2 <= j && allow_paired[offset[i+1]+j])
                {
                    Fast_LogPlusEquals(FEo[offset[i+1]+j-1], FEo[offset[i]+j] + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1));
                }
                
                // compute FN(i,j)
                
                Fast_LogPlusEquals(FNo[offset[i]+j], FEo[offset[i]+j]);
            }
            
            // FN[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           and the next interaction is not a stacking pair
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                // compute ScoreHairpin(i,j) -- do nothing
                
                // compute SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                {
                    RealT temp = FNo[offset[i]+j];
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            if (i == p && j == q) continue;
                            
                            Fast_LogPlusEquals(FCo[offset[p+1]+q-1], temp + ScoreSingle(i,j,p,q));
                        }
                    }
                }
#else
                
                {
                    RealT score_other = FNo[offset[i]+j] + ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        RealT *FCptr = &(FCo[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            if (i == p && j == q) continue;
                            
                            Fast_LogPlusEquals(FCptr[q], score_other + cache_score_single[p-i][j-q].first + ScoreBasePair(p+1,q) + ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                        }
                    }
                }
#endif
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c)
                
                Fast_LogPlusEquals(FM2o, FNo[offset[i]+j] + ScoreJunctionA(i,j) + ScoreMultiPaired() + ScoreMultiBase());
                
            }
            
#else
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                // compute ScoreHairpin(i,j) -- do nothing
                
                // compute SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1])

#if !FAST_SINGLE_BRANCH_LOOPS
                {
                    RealT temp = FCo[offset[i]+j];
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            
                            Fast_LogPlusEquals(FCo[offset[p+1]+q-1],
                                               temp + (p == i && q == j ? ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) : ScoreSingle(i,j,p,q)));
                        }
                    }
                }
#else
                
                {
                    RealT score_helix = (i+2 <= j ? FCo[offset[i]+j] + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) : 0);
                    RealT score_other = FCo[offset[i]+j] + ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        RealT *FCptr = &(FCo[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            
                            Fast_LogPlusEquals(FCptr[q], 
                                               (p == i && q == j) ?
                                               score_helix :
                                               score_other + cache_score_single[p-i][j-q].first + ScoreBasePair(p+1,q) + ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                        }
                    }
                }
#endif
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c)
                
                Fast_LogPlusEquals(FM2o, FCo[offset[i]+j] + ScoreJunctionA(i,j) + ScoreMultiPaired() + ScoreMultiBase());
                
            }
            
#endif
            
            // FM2[i,j] = SUM (i<k<j : FM1[i,k] + FM[k,j])
            
#if SIMPLE_FM2
            
            for (int k = i+1; k < j; k++)
            {
                Fast_LogPlusEquals(FM1o[offset[i]+k], FM2o + FMi[offset[k]+j]);
                Fast_LogPlusEquals(FMo[offset[k]+j], FM2o + FM1i[offset[i]+k]);
            }

#else
            if (i+2 <= j)
            {
                RealT *p1i = &(FM1i[offset[i]+i+1]);
                RealT *p2i = &(FMi[offset[i+1]+j]);
                RealT *p1o = &(FM1o[offset[i]+i+1]);
                RealT *p2o = &(FMo[offset[i+1]+j]);
                for (register int k = i+1; k < j; k++)
                {
                    Fast_LogPlusEquals(*p1o, FM2o + *p2i);
                    Fast_LogPlusEquals(*p2o, FM2o + *p1i);
                    ++p1i;
                    ++p1o;
                    p2i += L-k;
                    p2o += L-k;
                }
            }
            
#endif
        }
    }
    
#if SHOW_TIMINGS
    std::cerr << "Outside score: " << F5o[0] << " (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeLogPartitionCoefficient()
//
// Return partition coefficient.
//////////////////////////////////////////////////////////////////////

template<class RealT>
inline RealT InferenceEngine<RealT>::ComputeLogPartitionCoefficient() const
{
    // NOTE: This should be equal to F5o[0]. 
    
    return F5i[L];
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputeFeatureCountExpectations()
// 
// Combine the results of the inside and outside algorithms
// in order to compute feature count expectations.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<RealT> InferenceEngine<RealT>::ComputeFeatureCountExpectations()
{
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif

    //std::cerr << "Inside score: " << F5i[L].GetLogRepresentation() << std::endl;
    //std::cerr << "Outside score: " << F5o[0].GetLogRepresentation() << std::endl;
    
    const RealT Z = ComputeLogPartitionCoefficient();

    ClearCounts();
    
    for (int i = L; i >= 0; i--)
    {
        for (int j = i; j <= L; j++)
        {

            // FM2[i,j] = SUM (i<k<j : FM1[i,k] + FM[k,j])
            
            RealT FM2i = RealT(NEG_INF);
            
#if SIMPLE_FM2
            
            for (int k = i+1; k < j; k++)
                Fast_LogPlusEquals(FM2i, FM1i[offset[i]+k] + FMi[offset[k]+j]);
            
#else
            
            if (i+2 <= j)
            {
                const RealT *p1 = &(FM1i[offset[i]+i+1]);
                const RealT *p2 = &(FMi[offset[i+1]+j]);
                for (register int k = i+1; k < j; k++)
                {
                    Fast_LogPlusEquals(FM2i, (*p1) + (*p2));
                    ++p1;
                    p2 += L-k;
                }
            }
            
#endif
            
#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
            
            // FN[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           and the next interaction is not a stacking pair
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                RealT outside = FNo[offset[i]+j] - Z;
                
                // compute ScoreHairpin(i,j)
                
                if (allow_unpaired[offset[i]+j] && j-i >= C_MIN_HAIRPIN_LENGTH)
                    CountHairpin(i,j,Fast_Exp(outside + ScoreHairpin(i,j)));
                
                // compute SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                {
                    if (p > i && !allow_unpaired_position[p]) break;
                    int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                    for (int q = j; q >= q_min; q--)
                    {
                        if (q < j && !allow_unpaired_position[q+1]) break;
                        if (!allow_paired[offset[p+1]+q]) continue;
                        if (i == p && j == q) continue;
                        
                        CountSingle(i,j,p,q,Fast_Exp(outside + ScoreSingle(i,j,p,q) + FCi[offset[p+1]+q-1]));
                    }
                }
                
#else
                
                {
                    RealT score_other = outside + ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        const RealT *FCptr = &(FCi[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            if (i == p && j == q) continue;
                            
                            RealT value = Fast_Exp(score_other + cache_score_single[p-i][j-q].first + FCptr[q] + ScoreBasePair(p+1,q) + ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                            cache_score_single[p-i][j-q].second += value;
                            CountBasePair(p+1,q,value);
                            CountJunctionB(i,j,value);
                            CountJunctionB(q,p,value);
                            CountSingleNucleotides(i,j,p,q,value);
                        }
                    }
                }
#endif
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c)
                
                {
                    RealT value = Fast_Exp(outside + FM2i + ScoreJunctionA(i,j) + ScoreMultiPaired() + ScoreMultiBase());
                    CountJunctionA(i,j,value);
                    CountMultiPaired(value);
                    CountMultiBase(value);
                }
            }
            
            // FE[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) ... (i-D+1,j+D) are 
            //           already base-paired
            //
            //         = SUM [ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]   if i+2<=j,
            //                FN(i,j)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                RealT outside = FEo[offset[i]+j] - Z;
                
                // compute ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]
                
                if (i+2 <= j && allow_paired[offset[i+1]+j])
                {
                    RealT value = Fast_Exp(outside + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) + FEi[offset[i+1]+j-1]);
                    CountBasePair(i+1,j,value);
                    CountHelixStacking(i,j+1,value);
                }
                
                // compute FN(i,j) -- do nothing
                
            }
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           but (i-1,j+2) are not
            //
            //         = SUM [ScoreIsolated() + FN(i,j),
            //                SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k)),
            //                FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT outside = FCo[offset[i]+j] - Z;
                
                // compute ScoreIsolated() + FN(i,j)
                
                CountIsolated(Fast_Exp(outside + ScoreIsolated() + FNi[offset[i]+j]));
                
                // compute SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k))
                
                bool allowed = true;
                for (int k = 2; k < D_MAX_HELIX_LENGTH; k++)
                {
                    if (i + 2*k - 2 > j) break;
                    if (!allow_paired[offset[i+k-1]+j-k+2]) { allowed = false; break; }
                    CountHelix(i-1,j+1,k,Fast_Exp(outside + ScoreHelix(i-1,j+1,k) + FNi[offset[i+k-1]+j-k+1]));
                }
                
                // compute FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
                
                if (i + 2*D_MAX_HELIX_LENGTH-2 <= j)
                {
                    if (allowed && allow_paired[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+2])
                        CountHelix(i-1,j+1,D_MAX_HELIX_LENGTH,
                                   Fast_Exp(outside + ScoreHelix(i-1,j+1,D_MAX_HELIX_LENGTH) + FEi[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+1]));
                }
            }

#else
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT outside = FCo[offset[i]+j] - Z;
                
                // compute ScoreHairpin(i,j)
                
                if (allow_unpaired[offset[i]+j] && j-i >= C_MIN_HAIRPIN_LENGTH)
                    CountHairpin(i,j,Fast_Exp(outside + ScoreHairpin(i,j)));
                
                // compute SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                {
                    if (p > i && !allow_unpaired_position[p]) break;
                    int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                    for (int q = j; q >= q_min; q--)
                    {
                        if (q < j && !allow_unpaired_position[q+1]) break;
                        if (!allow_paired[offset[p+1]+q]) continue;

                        if (p == i && q == j)
                        {
                            RealT value = Fast_Exp(outside + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) + FCi[offset[p+1]+q-1]);
                            CountBasePair(i+1,j,value);
                            CountHelixStacking(i,j+1,value);
                        }
                        else
                        {
                            CountSingle(i,j,p,q,Fast_Exp(outside + ScoreSingle(i,j,p,q) + FCi[offset[p+1]+q-1]));
                        }
                    }
                }
                
#else
                
                {
                    RealT score_helix = (i+2 <= j ? outside + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) : 0);
                    RealT score_other = outside + ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        const RealT *FCptr = &(FCi[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            
                            if (p == i && q == j)
                            {
                                RealT value = Fast_Exp(score_helix + FCptr[q]);
                                cache_score_single[0][0].second += value;
                                CountBasePair(i+1,j,value);
                                CountHelixStacking(i,j+1,value);
                            }
                            else
                            {
                                RealT value = Fast_Exp(score_other + cache_score_single[p-i][j-q].first + FCptr[q] + ScoreBasePair(p+1,q) + 
                                                       ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                                cache_score_single[p-i][j-q].second += value;
                                CountBasePair(p+1,q,value);
                                CountJunctionB(i,j,value);
                                CountJunctionB(q,p,value);
                                CountSingleNucleotides(i,j,p,q,value);
                            }
                        }
                    }
                }
#endif
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c)
                
                {
                    RealT value = Fast_Exp(outside + FM2i + ScoreJunctionA(i,j) + ScoreMultiPaired() + ScoreMultiBase());
                    CountJunctionA(i,j,value);
                    CountMultiPaired(value);
                    CountMultiBase(value);
                }
            }
            
#endif
            
            // FM1[i,j] = optimal energy for substructure belonging to a
            //            multibranch loop containing a (k+1,j) base pair
            //            preceded by 5' unpaired nucleotides from i to k
            //            for some i <= k <= j-2
            //
            //          = SUM [FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)  if i+2<=j,
            //                 FM1[i+1,j] + b                                          if i+2<=j]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                
                // compute FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)
                
                if (allow_paired[offset[i+1]+j])
                {
                    RealT value = Fast_Exp(FM1o[offset[i]+j] + FCi[offset[i+1]+j-1] + ScoreJunctionA(j,i) + ScoreMultiPaired() + ScoreBasePair(i+1,j) - Z);
                    CountJunctionA(j,i,value);
                    CountMultiPaired(value);
                    CountBasePair(i+1,j,value);
                }
                
                // compute FM1[i+1,j] + b
                
                if (allow_unpaired_position[i+1])
                {
                    CountMultiUnpaired(i+1,Fast_Exp(FM1o[offset[i]+j] + FM1i[offset[i+1]+j] + ScoreMultiUnpaired(i+1) - Z));
                }
            }
            
            // FM[i,j] = optimal energy for substructure belonging to a
            //           multibranch loop which contains at least one 
            //           helix
            //
            //         = SUM [SUM (i<k<j : FM1[i,k] + FM[k,j]),
            //                FM[i,j-1] + b,
            //                FM1[i,j]]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j]) -- do nothing
                
                // compute FM[i,j-1] + b
                
                if (allow_unpaired_position[j])
                    CountMultiUnpaired(j,Fast_Exp(FMo[offset[i]+j] + FMi[offset[i]+j-1] + ScoreMultiUnpaired(j) - Z));
                
                // compute FM1[i,j] -- do nothing
            }
        }
    }
    
    for (int j = 1; j <= L; j++)
    {
        
        // F5[j] = optimal energy for substructure between positions 0 and j
        //         (or 0 if j = 0)
        //
        //       = SUM [F5[j-1] + ScoreExternalUnpaired(),
        //              SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))]
        
        RealT outside = F5o[j] - Z;
        
        // compute F5[j-1] + ScoreExternalUnpaired()
        
        if (allow_unpaired_position[j])
            CountExternalUnpaired(j,Fast_Exp(outside + F5i[j-1] + ScoreExternalUnpaired(j)));
        
        // compute SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))
        
        for (int k = 0; k < j; k++)
        {
            if (allow_paired[offset[k+1]+j])
            {
                RealT value = Fast_Exp(outside + F5i[k] + FCi[offset[k+1]+j-1] + ScoreExternalPaired() + ScoreBasePair(k+1,j) + ScoreJunctionA(j,k));
                CountExternalPaired(value);
                CountBasePair(k+1,j,value);
                CountJunctionA(j,k,value);
            }      
        }
    }
    
#if SHOW_TIMINGS
    std::cerr << "Feature expectations (" << GetSystemTime() - starting_time << " seconds)" << std::endl;
#endif

    return GetCounts();
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::ComputePosterior()
// 
// Combine the results of the inside and outside algorithms
// in order to compute posterior probabilities of base pairing.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void InferenceEngine<RealT>::ComputePosterior()
{ 
    posterior.clear();
    posterior.resize(SIZE, RealT(0));
    
    //double starting_time = GetSystemTime();

    const RealT Z = ComputeLogPartitionCoefficient();
    
    for (int i = L; i >= 0; i--)
    {
        for (int j = i; j <= L; j++)
        {
            
            // FM2[i,j] = SUM (i<k<j : FM1[i,k] + FM[k,j])
            
            RealT FM2i = RealT(NEG_INF);
            
#if SIMPLE_FM2
      
            for (int k = i+1; k < j; k++)
                Fast_LogPlusEquals(FM2i, FM1i[offset[i]+k] + FMi[offset[k]+j]);
            
#else
            
            if (i+2 <= j)
            {
                const RealT *p1 = &(FM1i[offset[i]+i+1]);
                const RealT *p2 = &(FMi[offset[i+1]+j]);
                for (register int k = i+1; k < j; k++)
                {
                    Fast_LogPlusEquals(FM2i, (*p1) + (*p2));
                    ++p1;
                    p2 += L-k;
                }
            }
      
#endif

#if PARAMS_HELIX_LENGTH || PARAMS_ISOLATED_BASE_PAIR
            
            // FN[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           and the next interaction is not a stacking pair
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT outside = FNo[offset[i]+j] - Z;
                
                // compute ScoreHairpin(i,j)
                
                if (allow_unpaired[offset[i]+j] && j-i >= C_MIN_HAIRPIN_LENGTH)
                    CountHairpin(i,j,Fast_Exp(outside + ScoreHairpin(i,j)));
                
                // compute SUM (i<=p<p+2<=q<=j, p-i+j-q>0 : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                {
                    if (p > i && !allow_unpaired_position[p]) break;
                    int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                    for (int q = j; q >= q_min; q--)
                    {
                        if (q < j && !allow_unpaired_position[q+1]) break;
                        if (!allow_paired[offset[p+1]+q]) continue;
                        if (i == p && j == q) continue;
                        
                        posterior[offset[p+1]+q] += Fast_Exp(outside + ScoreSingle(i,j,p,q) + FCi[offset[p+1]+q-1]);
                    }
                }
                
#else
                
                {
                    RealT score_other = outside + ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        const RealT *FCptr = &(FCi[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            if (i == p && j == q) continue;
                            
                            posterior[offset[p+1]+q] += 
                                Fast_Exp(score_other + cache_score_single[p-i][j-q].first + FCptr[q] + ScoreBasePair(p+1,q) + ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                        }
                    }
                }
#endif

                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c) -- do nothing
                
            }
            
            // FE[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) ... (i-D+1,j+D) are 
            //           already base-paired
            //
            //         = SUM [ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]   if i+2<=j,
            //                FN(i,j)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                RealT outside = FEo[offset[i]+j] - Z;
                
                // compute ScoreBP(i+1,j) + ScoreHelixStacking(i,j+1) + FE[i+1,j-1]
                
                if (i+2 <= j && allow_paired[offset[i+1]+j])
                    posterior[offset[i]+j] += Fast_Exp(outside + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) + FEi[offset[i+1]+j-1]);
                
                // compute FN(i,j) -- do nothing
                
            }
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //           but (i-1,j+2) are not
            //
            //         = SUM [ScoreIsolated() + FN(i,j),
            //                SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k)),
            //                FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
            //
            //           (assuming 0 < i <= j < L)
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                RealT outside = FCo[offset[i]+j] - Z;
                
                // compute ScoreIsolated() + FN(i,j) -- do nothing
                
                CountIsolated(Fast_Exp(outside + ScoreIsolated() + FNi[offset[i]+j]));
                
                // compute SUM (2<=k<D : FN(i+k-1,j-k+1) + ScoreHelix(i-1,j+1,k))
                
                bool allowed = true;
                for (int k = 2; k < D_MAX_HELIX_LENGTH; k++)
                {
                    if (i + 2*k - 2 > j) break;
                    if (!allow_paired[offset[i+k-1]+j-k+2]) { allowed = false; break; }
                    RealT value = Fast_Exp(outside + ScoreHelix(i-1,j+1,k) + FNi[offset[i+k-1]+j-k+1]);
                    for (int p = 1; p < k; p++)
                        posterior[offset[i+p]+j-p+1] += value;
                }
                
                // compute FE(i+D-1,j-D+1) + ScoreHelix(i-1,j+1,D)]
                
                if (i + 2*D_MAX_HELIX_LENGTH-2 <= j)
                {
                    if (allowed && allow_paired[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+2]) {
                        RealT value = Fast_Exp(outside + ScoreHelix(i-1,j+1,D_MAX_HELIX_LENGTH) + FEi[offset[i+D_MAX_HELIX_LENGTH-1]+j-D_MAX_HELIX_LENGTH+1]);
                        
                        for (int k = 1; k < D_MAX_HELIX_LENGTH; k++)
                            posterior[offset[i+k]+j-k+1] += value;
                    }
                }
            }
            
#else
            
            // FC[i,j] = optimal energy for substructure between positions
            //           i and j such that letters (i,j+1) are base-paired
            //
            //         = SUM [ScoreHairpin(i,j),
            //                SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1]),
            //                ScoreJunctionA(i,j) + a + c + SUM (i<k<j : FM1[i,k] + FM[k,j])]
            //
            //           (assuming 0 < i <= j < L)
            //
            // Multi-branch loops are scored as [a + b * (# unpaired) + c * (# branches)]
            
            if (0 < i && j < L && allow_paired[offset[i]+j+1])
            {
                
                RealT outside = FCo[offset[i]+j] - Z;
                
                // compute ScoreHairpin(i,j)
                
                if (allow_unpaired[offset[i]+j] && j-i >= C_MIN_HAIRPIN_LENGTH)
                    CountHairpin(i,j,Fast_Exp(outside + ScoreHairpin(i,j)));
                
                // compute SUM (i<=p<p+2<=q<=j : ScoreSingle(i,j,p,q) + FC[p+1,q-1])
                
#if !FAST_SINGLE_BRANCH_LOOPS
                for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                {
                    if (p > i && !allow_unpaired_position[p]) break;
                    int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                    for (int q = j; q >= q_min; q--)
                    {
                        if (q < j && !allow_unpaired_position[q+1]) break;
                        if (!allow_paired[offset[p+1]+q]) continue;

                        if (p == i && q == j)
                        {
                            posterior[offset[p+1]+q] += Fast_Exp(outside + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) + FCi[offset[p+1]+q-1]);
                        }
                        else
                        {
                            posterior[offset[p+1]+q] += Fast_Exp(outside + ScoreSingle(i,j,p,q) + FCi[offset[p+1]+q-1]);
                        }
                    }
                }
                
#else
                
                {
                    RealT score_helix = (i+2 <= j ? outside + ScoreBasePair(i+1,j) + ScoreHelixStacking(i,j+1) : 0);
                    RealT score_other = outside + ScoreJunctionB(i,j);
                    
                    for (int p = i; p <= std::min(i+C_MAX_SINGLE_LENGTH,j); p++)
                    {
                        if (p > i && !allow_unpaired_position[p]) break;
                        int q_min = std::max(p+2,p-i+j-C_MAX_SINGLE_LENGTH);
                        const RealT *FCptr = &(FCi[offset[p+1]-1]);
                        for (int q = j; q >= q_min; q--)
                        {
                            if (q < j && !allow_unpaired_position[q+1]) break;
                            if (!allow_paired[offset[p+1]+q]) continue;
                            
                            posterior[offset[p+1]+q] +=
                                Fast_Exp(p == i && q == j ?
                                         score_helix + FCptr[q] :
                                         score_other + cache_score_single[p-i][j-q].first + FCptr[q] + ScoreBasePair(p+1,q) + 
                                         ScoreJunctionB(q,p) + ScoreSingleNucleotides(i,j,p,q));
                        }
                    }
                }
                
#endif
                
                // compute SUM (i<k<j : FM1[i,k] + FM[k,j] + ScoreJunctionA(i,j) + a + c) -- do nothing
                
            }
            
#endif
            
            // FM1[i,j] = optimal energy for substructure belonging to a
            //            multibranch loop containing a (k+1,j) base pair
            //            preceded by 5' unpaired nucleotides from i to k
            //            for some i <= k <= j-2
            //
            //          = SUM [FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)  if i+2<=j,
            //                 FM1[i+1,j] + b                                          if i+2<=j]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            if (0 < i && i+2 <= j && j < L)
            {
                
                // Compute FC[i+1,j-1] + ScoreJunctionA(j,i) + c + ScoreBP(i+1,j)
                
                if (allow_paired[offset[i+1]+j])
                    posterior[offset[i+1]+j] += Fast_Exp(FM1o[offset[i]+j] + FCi[offset[i+1]+j-1] + ScoreJunctionA(j,i) + ScoreMultiPaired() + ScoreBasePair(i+1,j) - Z);
                
                // Compute FM1[i+1,j] + b -- do nothing
                
            }
            
            // FM[i,j] = optimal energy for substructure belonging to a
            //           multibranch loop which contains at least one 
            //           helix
            //
            //         = SUM [SUM (i<k<j : FM1[i,k] + FM[k,j]),
            //                FM[i,j-1] + b,
            //                FM1[i,j]]
            //
            //            (assuming 0 < i < i+2 <= j < L)
            
            // Compute SUM (i<k<j : FM1[i,k] + FM[k,j]) -- do nothing
            
            // Compute FM[i,j-1] + b -- do nothing
            
            // Compute FM1[i,j] -- do nothing
        }
    }
    
    for (int j = 1; j <= L; j++)
    {
        
        // F5[j] = optimal energy for substructure between positions 0 and j
        //         (or 0 if j = 0)
        //
        //       = SUM [F5[j-1] + ScoreExternalUnpaired(),
        //              SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))]
        
        RealT outside = F5o[j] - Z;
        
        // compute F5[j-1] + ScoreExternalUnpaired() -- do nothing
        
        // compute SUM (0<=k<j : F5[k] + FC[k+1,j-1] + ScoreExternalPaired() + ScoreBP(k+1,j) + ScoreJunctionA(j,k))
        
        for (int k = 0; k < j; k++)
        {
            if (allow_paired[offset[k+1]+j])
                posterior[offset[k+1]+j] += Fast_Exp(outside + F5i[k] + FCi[offset[k+1]+j-1] + ScoreExternalPaired() + ScoreBasePair(k+1,j) + ScoreJunctionA(j,k));
        }
    }

    for (int i = 1; i <= L; i++)
    {
        for (int j = i+1; j <= L; j++)
        {
            posterior[offset[i]+j] = Clip(posterior[offset[i]+j], RealT(0), RealT(1));
        }
    }
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::PredictPairingsPosterior()
//
// Use posterior decoding to predict pairings.
//////////////////////////////////////////////////////////////////////

template<class RealT>
std::vector<int> InferenceEngine<RealT>::PredictPairingsPosterior(const RealT gamma) const
{
    Assert(gamma > 0, "Non-negative gamma expected.");
    
#if SHOW_TIMINGS
    double starting_time = GetSystemTime();
#endif
    RealT* unpaired_posterior  = new RealT[L+1];
    RealT* score               = new RealT[SIZE];
    int* traceback             = new int[SIZE];
    
    // compute the scores for unpaired nucleotides
    
    for (int i = 1; i <= L; i++)
    {
        unpaired_posterior[i] = RealT(1);
        for (int j = 1; j < i; j++) unpaired_posterior[i] -= posterior[offset[j]+i];
        for (int j = i+1; j <= L; j++) unpaired_posterior[i] -= posterior[offset[i]+j];
    }
    
    for (int i = 1; i <= L; i++) unpaired_posterior[i] /= 2 * gamma;
    
    // initialize matrices
    
    std::fill(score, score+SIZE, RealT(-1.0));
    std::fill(traceback, traceback+SIZE, -1);
    
    // dynamic programming
    
    for (int i = L; i >= 0; i--)
    {
        for (int j = i; j <= L; j++)
        {
            RealT &this_score = score[offset[i]+j];
            int &this_traceback = traceback[offset[i]+j];
            
            if (i == j)
            {
                UPDATE_MAX(this_score, this_traceback, RealT(0), 0);
            }
            else
            {
                if (allow_unpaired_position[i+1])
                    UPDATE_MAX(this_score, this_traceback, unpaired_posterior[i+1] + score[offset[i+1]+j], 1);
                if (allow_unpaired_position[j])
                    UPDATE_MAX(this_score, this_traceback, unpaired_posterior[j] + score[offset[i]+j-1], 2);
                if (i+2 <= j)
                { 
                    if (allow_paired[offset[i+1]+j])
                        UPDATE_MAX(this_score, this_traceback, posterior[offset[i+1]+j] + score[offset[i+1]+j-1], 3);
                    
#if SIMPLE_FM2
                    
                    for (int k = i+1; k < j; k++)
                        UPDATE_MAX(this_score, this_traceback, score[offset[i]+k] + score[offset[k]+j], k+4);	
                    
#else
                    
                    RealT *p1 = &(score[offset[i]+i+1]);
                    RealT *p2 = &(score[offset[i+1]+j]);
                    for (register int k = i+1; k < j; k++)
                    {
                        UPDATE_MAX(this_score, this_traceback, (*p1) + (*p2), k+4);
                        ++p1;
                        p2 += L-k;
                    }
                    
#endif
                }
            }
        }
    }

#if SHOW_TIMINGS
    std::cerr << "Time: " << GetSystemTime() - starting_time << std::endl;
#endif
    
    // perform traceback
    
    std::vector<int> solution(L+1,SStruct::UNPAIRED);
    solution[0] = SStruct::Sequence::UNKNOWN;
    
    std::queue<std::pair<int,int> > traceback_queue;
    traceback_queue.push(std::make_pair(0, L));
    
    while (!traceback_queue.empty())
    {
        std::pair<int,int> t = traceback_queue.front();
        traceback_queue.pop();
        const int i = t.first;
        const int j = t.second;
        
        switch (traceback[offset[i]+j])
        {
            case -1:
                Assert(false, "Should not get here.");
                break;
            case 0: 
                break;
            case 1: 
                traceback_queue.push(std::make_pair(i+1,j));
                break;
            case 2: 
                traceback_queue.push(std::make_pair(i,j-1));
                break;
            case 3:
                solution[i+1] = j;
                solution[j] = i+1;
                traceback_queue.push(std::make_pair(i+1,j-1));
                break;
            default:
            {
                const int k = traceback[offset[i]+j] - 4;
                traceback_queue.push(std::make_pair(i,k));
                traceback_queue.push(std::make_pair(k,j));
            }
            break;       
        }
    }
    
    return solution;
}

//////////////////////////////////////////////////////////////////////
// InferenceEngine::GetPosterior()
//
// Return posterior probability matrix, thresholded.
//////////////////////////////////////////////////////////////////////

template<class RealT>
RealT *InferenceEngine<RealT>::GetPosterior(const RealT posterior_cutoff) const
{
    RealT *ret = new RealT[SIZE];
    for (int i = 0; i < SIZE; i++)
        ret[i] = (posterior[i] >= posterior_cutoff ? posterior[i] : RealT(0));
    return ret;
}
*/
