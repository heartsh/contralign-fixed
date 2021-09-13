//////////////////////////////////////////////////////////////////////
// SpectralConsistency.ipp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// SpectralConsistency::SpectralConsistency()
//
// Constructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
SpectralConsistency<RealT>::SpectralConsistency(std::vector<SparseMatrix<RealT> *> &posteriors,
                                                const bool toggle_verbose,
                                                const int num_iterations) :
    orig_posteriors(posteriors.size(), static_cast<SparseMatrix<RealT> *>(NULL)),
    self_edge_posteriors(posteriors.size(), static_cast<SparseMatrix<RealT> *>(NULL)),
    curr_posteriors(posteriors),
    m(int(Sqrt(double(posteriors.size())) + 0.5)),
    toggle_verbose(toggle_verbose),
    num_iterations(num_iterations),
    alpha(RealT(2) / RealT(m))
{
    Assert(m*m == int(posteriors.size()), "Dimension mismatch.");

    // make deep copy of posterior probabilities
    // and prepare to compute self-edge weights (initialize to 1)

    for (int x = 0; x < m; x++)
    {
        for (int y = x+1; y < m; y++)
        {
            orig_posteriors[x*m+y] = new SparseMatrix<RealT>(*posteriors[x*m+y]);
            std::vector<RealT> unsparse_mask(posteriors[x*m+y]->GetUnsparseMask());
            self_edge_posteriors[x*m+y] = new SparseMatrix<RealT>(*posteriors[x*m+y], &unsparse_mask[0]);
        }
    }

    // compute self-edge weights

    Multiply(self_edge_posteriors, false);
    for (int x = 0; x < m; x++)
    {
        for (int y = x+1; y < m; y++)
        {
            std::vector<RealT> unsparse = self_edge_posteriors[x*m+y]->GetUnsparse();
            unsparse = (RealT(1) - unsparse);
            SparseMatrix<RealT> *new_self_edge = new SparseMatrix<RealT>(*self_edge_posteriors[x*m+y], &unsparse[0]);
            delete self_edge_posteriors[x*m+y];
            self_edge_posteriors[x*m+y] = new_self_edge;
        }
    }
}

//////////////////////////////////////////////////////////////////////
// SpectralConsistency::~SpectralConsistency()
//
// Destructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
SpectralConsistency<RealT>::~SpectralConsistency()
{
    for (size_t i = 0; i < orig_posteriors.size(); i++)
    {
        delete orig_posteriors[i];
        delete self_edge_posteriors[i];
    }
}

//////////////////////////////////////////////////////////////////////
// SpectralConsistency::AccumulateTransitive()
//
// Accumulate probabilistic consistency transformation through z.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void SpectralConsistency<RealT>::Accumulate(std::vector<RealT> &res, std::vector<SparseMatrix<RealT> *> &curr_posteriors, int x, int y, int z)
{
    Assert(x < y, "Unexpected ordering of arguments.");
    
    const SparseMatrix<RealT> orig_XZ = (x < z ? *orig_posteriors[x*m+z] : SparseMatrix<RealT>(*orig_posteriors[z*m+x], SparseMatrix<RealT>::TRANSPOSE));
    const SparseMatrix<RealT> orig_YZ = (y < z ? *orig_posteriors[y*m+z] : SparseMatrix<RealT>(*orig_posteriors[z*m+y], SparseMatrix<RealT>::TRANSPOSE));
    const SparseMatrix<RealT> curr_ZX = (z < x ? *curr_posteriors[z*m+x] : SparseMatrix<RealT>(*curr_posteriors[x*m+z], SparseMatrix<RealT>::TRANSPOSE));
    const SparseMatrix<RealT> curr_ZY = (z < y ? *curr_posteriors[z*m+y] : SparseMatrix<RealT>(*curr_posteriors[y*m+z], SparseMatrix<RealT>::TRANSPOSE));

    Assert(orig_XZ.GetNumRows() == curr_ZX.GetNumCols(), "Dimension mismatch.");
    Assert(orig_XZ.GetNumCols() == curr_ZX.GetNumRows(), "Dimension mismatch.");
    Assert(orig_YZ.GetNumRows() == curr_ZY.GetNumCols(), "Dimension mismatch.");
    Assert(orig_YZ.GetNumCols() == curr_ZY.GetNumRows(), "Dimension mismatch.");
    Assert(orig_XZ.GetNumCols() == orig_YZ.GetNumCols(), "Dimension mismatch.");
    Assert(orig_XZ.GetNumRows() * orig_YZ.GetNumRows() == int(res.size()), "Dimension mismatch.");

    std::vector<RealT> tmp_res(res.size(), RealT(0));
    const int row_size = orig_YZ.GetNumRows();

    for (int i = 1; i < orig_XZ.GetNumRows(); i++)
    {
        for (const SparseMatrixEntry<RealT> *orig_XZ_iter = orig_XZ.GetRowBegin(i); orig_XZ_iter != orig_XZ.GetRowEnd(i); ++orig_XZ_iter)
        {
            const int k = orig_XZ_iter->column;
            for (const SparseMatrixEntry<RealT> *curr_ZY_iter = curr_ZY.GetRowBegin(k); curr_ZY_iter != curr_ZY.GetRowEnd(k); ++curr_ZY_iter)
            {
                const int j = curr_ZY_iter->column;
                tmp_res[i * row_size + j] += orig_XZ_iter->value * curr_ZY_iter->value;
            }
        }
    }

    for (int j = 1; j < orig_YZ.GetNumRows(); j++)
    {
        for (const SparseMatrixEntry<RealT> *orig_YZ_iter = orig_YZ.GetRowBegin(j); orig_YZ_iter != orig_YZ.GetRowEnd(j); ++orig_YZ_iter)
        {
            const int k = orig_YZ_iter->column;
            for (const SparseMatrixEntry<RealT> *curr_ZX_iter = curr_ZX.GetRowBegin(k); curr_ZX_iter != curr_ZX.GetRowEnd(k); ++curr_ZX_iter)
            {
                const int i = curr_ZX_iter->column;
                tmp_res[i * row_size + j] += orig_YZ_iter->value * curr_ZX_iter->value;
            }
        }
    }

    // update accumulated variable

    res += ((RealT(1) - alpha) / (RealT(2) * RealT(m - 2))) * tmp_res;    
}

//////////////////////////////////////////////////////////////////////
// SpectralConsistency::Multiply()
//
// Multiply posteriors vector by weight matrix.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void SpectralConsistency<RealT>::Multiply(std::vector<SparseMatrix<RealT> *> &posteriors, bool use_self_edges)
{
    std::vector<SparseMatrix<RealT> *> new_posteriors(m*m, static_cast<SparseMatrix<RealT> *>(NULL));

    for (int x = 0; x < m; x++)
    {
        for (int y = x+1; y < m; y++)
        {
            if (toggle_verbose)
            {
                WriteProgressMessage(SPrintF("Reestimating pairwise posteriors: (%d) vs (%d)...", x+1, y+1));
            }

            std::vector<RealT> new_table =
                use_self_edges ?
                self_edge_posteriors[x*m+y]->GetUnsparse() * posteriors[x*m+y]->GetUnsparse() :
                std::vector<RealT>(self_edge_posteriors[x*m+y]->GetNumRows() * self_edge_posteriors[x*m+y]->GetNumCols(), RealT(0));
            
            for (int z = 0; z < m; z++)
            {
                if (z == x || z == y) continue;
                Accumulate(new_table, posteriors, x, y, z);
            }
            
            new_posteriors[x*m+y] = new SparseMatrix<RealT>(*posteriors[x*m+y], &new_table[0]);
        }
    }

    // replace old posteriors
    
    for (size_t i = 0; i < posteriors.size(); i++)
    {
        delete posteriors[i];
        posteriors[i] = new_posteriors[i];
    }

    if (toggle_verbose)
    {
        WriteProgressMessage("");
    }     
}

//////////////////////////////////////////////////////////////////////
// SpectralConsistency::Recenter()
//
// Recenter posterior matrices.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void SpectralConsistency<RealT>::Recenter(std::vector<SparseMatrix<RealT> *> &posteriors)
{
    // compute average

    RealT average = 0;
    int num_entries = 0;
    
    for (int x = 0; x < m; x++)
    {
        for (int y = x+1; y < m; y++)
        {
            average += posteriors[x*m+y]->GetSum();
            num_entries += posteriors[x*m+y]->GetNumEntries();
        }
    }

    average /= RealT(num_entries);

    RealT norm = 0;
    
    // recenter

    for (int x = 0; x < m; x++)
    {
        for (int y = x+1; y < m; y++)
        {
            RealT partial_norm2 = 0;
            SparseMatrix<RealT> &sparse = *posteriors[x*m+y];
            for (int i = 1; i < sparse.GetNumRows(); i++)
            {
                for (SparseMatrixEntry<RealT> *iter = sparse.GetRowBegin(i); iter != sparse.GetRowEnd(i); ++iter)
                {
                    iter->value -= average;
                    partial_norm2 += iter->value * iter->value;
                }
            }
            norm += partial_norm2;
        }
    }

    norm = Sqrt(norm);

    // renormalize

    for (int x = 0; x < m; x++)
    {
        for (int y = x+1; y < m; y++)
        {
            SparseMatrix<RealT> &sparse = *posteriors[x*m+y];
            for (int i = 1; i < sparse.GetNumRows(); i++)
            {
                for (SparseMatrixEntry<RealT> *iter = sparse.GetRowBegin(i); iter != sparse.GetRowEnd(i); ++iter)
                {
                    iter->value /= norm;
                }
            }
        }
    }

}

//////////////////////////////////////////////////////////////////////
// SpectralConsistency::Transform()
//
// Perform consistency transformation on posteriors.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void SpectralConsistency<RealT>::Transform()
{
    for (int iter = 0; iter < num_iterations; iter++)
    {
        Multiply(curr_posteriors, true);
        Recenter(curr_posteriors);
    }

    // curr_posteriors[0*m+1]->PrintSparse(std::cerr);
}
