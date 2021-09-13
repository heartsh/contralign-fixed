/////////////////////////////////////////////////////////////////
// Annealing.hpp
/////////////////////////////////////////////////////////////////

#ifndef ANNEALING_HPP
#define ANNEALING_HPP

#include "MultiSequence.hpp"
#include "SparseMatrix.hpp"
#include "Utilities.hpp"

#define INVALID_EDGE (RealT(1000000000))

/////////////////////////////////////////////////////////////////
// struct Vertex
/////////////////////////////////////////////////////////////////

struct Vertex
{
    int ord;         // label in topological sorting algo
    int next;        // order of next vertex in cluster
    int time_stamp;  // time most recently visited

    bool operator< (const Vertex &rhs) const { return ord < rhs.ord || (ord == rhs.ord && next < rhs.next); }
};

/////////////////////////////////////////////////////////////////
// struct Edge
/////////////////////////////////////////////////////////////////

template<class RealT>
struct Edge
{
    RealT weight;
    int id1, id2;

    bool operator< (const Edge &rhs) const { return weight < rhs.weight; }
};

//////////////////////////////////////////////////////////////////////
// class TopologicalOrderComparator
//////////////////////////////////////////////////////////////////////

class TopologicalOrderComparator
{
    const std::vector<Vertex> &vertices;
public:
    TopologicalOrderComparator(const std::vector<Vertex> &vertices) : vertices(vertices) {}
    bool operator()(const int &id1, const int &id2) const { return (vertices[id1].ord < vertices[id2].ord); }
};

/////////////////////////////////////////////////////////////////
// class Annealing
/////////////////////////////////////////////////////////////////

template<class RealT>
class Annealing
{
    const MultiSequence &multi_seqs;
    const std::vector<SparseMatrix<RealT> *> &posteriors;
    const bool toggle_verbose;
    MultiSequence *alignment;
    
    std::vector<Vertex> vertices;
    std::vector<Edge<RealT> > edges;
    std::vector<int> offsets;

    std::vector<int> forward;
    std::vector<int> backward;
    std::vector<int> L;
    std::vector<int> R;
    int time_stamp;
    bool cycle_found;
    
    int GetClusterRepresentative(int id);
    std::pair<int,int> ConvertVertexToLocation(int id);
    int ConvertLocationToVertex(int seq, int index);

    RealT ComputeWeight(const int id1, const int id2);
    void ForwardDFS(const int x, const int y);
    void BackwardDFS(const int x, const int y);
    void Reorder();
    void Merge(const int x, const int y);
    void UseEdge(const Edge<RealT> &edge);
    void InitializeVertices();
    void InitializeEdges();
    void FinalizeOrdering(std::vector<Vertex> &vertices);
    void FormAlignment(std::vector<Vertex> &vertices);
        
public:
    Annealing(const MultiSequence &multi_seqs,
              const std::vector<SparseMatrix<RealT> *> &posteriors,
              const bool toggle_verbose);
    ~Annealing();
    
    void DoAlignment();
    const MultiSequence &GetAlignment();
};

#include "Annealing.ipp"

#endif
