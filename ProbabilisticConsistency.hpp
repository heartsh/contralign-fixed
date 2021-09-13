//////////////////////////////////////////////////////////////////////
// ProbabilisticConsistency.hpp
//////////////////////////////////////////////////////////////////////

#ifndef PROBABILISTICCONSISTENCY_HPP
#define PROBABILISTICCONSISTENCY_HPP

#include "SparseMatrix.hpp"
#include "Utilities.hpp"

//////////////////////////////////////////////////////////////////////
// class ProbabilisticConsistency
//////////////////////////////////////////////////////////////////////

template<class RealT>
class ProbabilisticConsistency
{
    std::vector<SparseMatrix<RealT> *> &posteriors;
    const int m;
    const bool toggle_verbose;
    const int num_iterations;

    void Accumulate(std::vector<RealT> &res, int x, int y, int z);

public:
    ProbabilisticConsistency(std::vector<SparseMatrix<RealT> *> &posteriors,
                             const bool toggle_verbose,
                             const int num_iterations);
    
    void Transform();
};

#include "ProbabilisticConsistency.ipp"

#endif
