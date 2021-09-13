//////////////////////////////////////////////////////////////////////
// SpectralConsistency.hpp
//////////////////////////////////////////////////////////////////////

#ifndef SPECTRALCONSISTENCY_HPP
#define SPECTRALCONSISTENCY_HPP

#include "SparseMatrix.hpp"
#include "Utilities.hpp"

//////////////////////////////////////////////////////////////////////
// class SpectralConsistency
//////////////////////////////////////////////////////////////////////

template<class RealT>
class SpectralConsistency
{
    std::vector<SparseMatrix<RealT> *> orig_posteriors;
    std::vector<SparseMatrix<RealT> *> self_edge_posteriors;
    std::vector<SparseMatrix<RealT> *> &curr_posteriors;
    const int m;
    const bool toggle_verbose;
    const int num_iterations;
    const RealT alpha;

    void Accumulate(std::vector<RealT> &res, std::vector<SparseMatrix<RealT> *> &curr_posteriors, int x, int y, int z);
    void Multiply(std::vector<SparseMatrix<RealT> *> &posteriors, bool use_self_edges);
    void Recenter(std::vector<SparseMatrix<RealT> *> &posteriors);

public:
    SpectralConsistency(std::vector<SparseMatrix<RealT> *> &posteriors,
                        const bool toggle_verbose,
                        const int num_iterations);
    ~SpectralConsistency();
    
    void Transform();
};

#include "SpectralConsistency.ipp"

#endif
