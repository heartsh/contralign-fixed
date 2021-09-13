//////////////////////////////////////////////////////////////////////
// ProbabilisticConsistency.ipp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// ProbabilisticConsistency::ProbabilisticConsistency()
//
// Constructor.
//////////////////////////////////////////////////////////////////////

template<class RealT>
ProbabilisticConsistency<RealT>::ProbabilisticConsistency(std::vector<SparseMatrix<RealT> *> &posteriors,
                                                          const bool toggle_verbose,
                                                          const int num_iterations) :
    posteriors(posteriors),
    m(int(Sqrt(double(posteriors.size())) + 0.5)),
    toggle_verbose(toggle_verbose),
    num_iterations(num_iterations)
{
    Assert(m*m == int(posteriors.size()), "Dimension mismatch.");
}

//////////////////////////////////////////////////////////////////////
// ProbabilisticConsistency::Accumulate()
//
// Accumulate probabilistic consistency transformation through z.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ProbabilisticConsistency<RealT>::Accumulate(std::vector<RealT> &res, int x, int y, int z)
{
    const SparseMatrix<RealT> XZ = (x < z ? *posteriors[x*m+z] : SparseMatrix<RealT>(*posteriors[z*m+x], SparseMatrix<RealT>::TRANSPOSE));
    const SparseMatrix<RealT> ZY = (z < y ? *posteriors[z*m+y] : SparseMatrix<RealT>(*posteriors[y*m+z], SparseMatrix<RealT>::TRANSPOSE));

    Assert(XZ.GetNumCols() == ZY.GetNumRows(), "Dimension mismatch.");
    Assert(XZ.GetNumRows() * ZY.GetNumCols() == int(res.size()), "Dimension mismatch.");

    const int row_size = ZY.GetNumCols();

    for (int i = 1; i < XZ.GetNumRows(); i++)
    {
        for (const SparseMatrixEntry<RealT> *XZiter = XZ.GetRowBegin(i); XZiter != XZ.GetRowEnd(i); ++XZiter)
        {
            const int k = XZiter->column;
            for (const SparseMatrixEntry<RealT> *ZYiter = ZY.GetRowBegin(k); ZYiter != ZY.GetRowEnd(k); ++ZYiter)
            {
                const int j = ZYiter->column;
                res[i * row_size + j] += XZiter->value * ZYiter->value;
            }
        }
    }    
}

//////////////////////////////////////////////////////////////////////
// ProbabilisticConsistency::Transform()
//
// Perform consistency transformation on posteriors.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void ProbabilisticConsistency<RealT>::Transform()
{
    for (int iter = 0; iter < num_iterations; iter++)
    {
        // compute new posteriors
        
        std::vector<SparseMatrix<RealT> *> new_posteriors(m*m, static_cast<SparseMatrix<RealT> *>(NULL));
        
        for (int x = 0; x < m; x++)
        {
            for (int y = x+1; y < m; y++)
            {
                if (toggle_verbose)
                {
                    WriteProgressMessage(SPrintF("Reestimating pairwise posteriors: (%d) vs (%d)...", x+1, y+1));
                }
                
                std::vector<RealT> new_table = RealT(2) * posteriors[x*m+y]->GetUnsparse();
                
                for (int z = 0; z < m; z++)
                {
                    if (z == x || z == y) continue;
                    Accumulate(new_table, x, y, z);
                }

                new_table = new_table / RealT(m);                
                new_posteriors[x*m+y] = new SparseMatrix<RealT>(*posteriors[x*m+y], &new_table[0]);
            }
        }

        // replace old posteriors

        for (size_t i = 0; i < posteriors.size(); i++)
        {
            delete posteriors[i];
            posteriors[i] = new_posteriors[i];
        }
    }

    if (toggle_verbose)
    {
        WriteProgressMessage("");
    }     
}
