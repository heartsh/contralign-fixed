/////////////////////////////////////////////////////////////////
// Annealing.ipp
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
// Annealing::Annealing()
//
// Constructor.
/////////////////////////////////////////////////////////////////

template<class RealT>
Annealing<RealT>::Annealing(const MultiSequence &multi_seqs,
                            const std::vector<SparseMatrix<RealT> *> &posteriors,
                            const bool toggle_verbose) :
    multi_seqs(multi_seqs), posteriors(posteriors), toggle_verbose(toggle_verbose),
    alignment(NULL), time_stamp(0), cycle_found(false)
{}

/////////////////////////////////////////////////////////////////
// Annealing::~Annealing()
//
// Destructor.
/////////////////////////////////////////////////////////////////

template<class RealT>
Annealing<RealT>::~Annealing()
{
    delete alignment;
}

/////////////////////////////////////////////////////////////////
// Annealing::GetClusterRepresentative()
//
// Get representative for a cluster of vertices.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline int Annealing<RealT>::GetClusterRepresentative(int id)
{
    int representative = id;
    int id_iter = id;
    
    while (true)
    {
        id_iter = vertices[id_iter].next;
        if (id_iter == id) break;
        representative = std::min(representative, id_iter);
    }
    
    return representative;    
}

/////////////////////////////////////////////////////////////////
// Annealing::ConvertVertexToLocation()
//
// Convert a vertex to a (sequence,index) pair.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline std::pair<int,int> Annealing<RealT>::ConvertVertexToLocation(int id)
{
    const int seq = std::upper_bound(offsets.begin(), offsets.end(), id) - offsets.begin() - 1;
    const int index = id - offsets[seq] + 1;
    return std::make_pair(seq, index);
}

/////////////////////////////////////////////////////////////////
// Annealing::ConvertLocationToVertex()
//
// Convert a (sequence,index) pair to a vertex.
/////////////////////////////////////////////////////////////////

template<class RealT>
inline int Annealing<RealT>::ConvertLocationToVertex(int seq, int index)
{
    return offsets[seq] + index - 1;
}

/////////////////////////////////////////////////////////////////
// Annealing::ComputeWeight()
//
// Compute weight for an edge.  The weight of an edge is
// defined as the average score increase among all newly created
// letter pairs for merging a pair of vertex clusters.  Only
// edges between the lowest index pairs in two clusters are valid;
// all other edges are given high weight and subsequently
// discarded.
/////////////////////////////////////////////////////////////////

template<class RealT>
RealT Annealing<RealT>::ComputeWeight(const int id1, const int id2)
{
    const int m = multi_seqs.GetNumSequences();
    RealT weight = RealT(0);
    int num_pairs = 0;

    if (vertices[id1].ord == vertices[id2].ord) return INVALID_EDGE;    

    // loop through nucleotides in cluster 1
    
    int id1_iter = id1;
    do
    {
        // loop through nucleotides in cluster 2
        
        int id2_iter = id2;
        do
        {
            // add contribution

            const std::pair<int,int> loc1 = ConvertVertexToLocation(id1_iter);
            const std::pair<int,int> loc2 = ConvertVertexToLocation(id2_iter);
            if (loc1.first == loc2.first) return INVALID_EDGE;

            weight += (loc1.first < loc2.first ?
                       ((*posteriors[loc1.first * m + loc2.first])(loc1.second, loc2.second)) :
                       ((*posteriors[loc2.first * m + loc1.first])(loc2.second, loc1.second)));

            num_pairs++;
            
            id2_iter = vertices[id2_iter].next;
        }
        while (id2_iter != id2);
            
        id1_iter = vertices[id1_iter].next;
    }
    while (id1_iter != id1);

    // report average

    return weight / RealT(num_pairs);
}

/////////////////////////////////////////////////////////////////
// Annealing::ForwardDFS()
//
// Forward DFS in the Pearce and Kelly dynamic topological
// sorting algorithm.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::ForwardDFS(const int x, const int y)
{
    Assert(y == GetClusterRepresentative(y), "Can only operator on cluster representatives.");

    vertices[y].time_stamp = time_stamp;
    forward.push_back(y);

    // loop through all outgoing edges
    
    int id = y;
    do
    {
        const std::pair<int,int> loc = ConvertVertexToLocation(id);
        if (loc.second < multi_seqs.GetSequence(loc.first).GetLength())
        {
            int successor = GetClusterRepresentative(id+1);

            // check for cycle
            
            if (vertices[successor].ord == vertices[x].ord)
            {
                cycle_found = true;
                return;
            }

            // continue to search affected region

            if (vertices[successor].time_stamp != time_stamp && vertices[successor].ord < vertices[x].ord)
            {
                ForwardDFS(x, successor);
                if (cycle_found) return;
            }
        }
        id = vertices[id].next;
    }
    while (id != y);
}

/////////////////////////////////////////////////////////////////
// Annealing::BackwardDFS()
//
// Backward DFS in the Pearce and Kelly dynamic topological
// sorting algorithm.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::BackwardDFS(const int x, const int y)
{
    Assert(x == GetClusterRepresentative(x), "Can only operator on cluster representatives.");

    vertices[x].time_stamp = time_stamp;
    backward.push_back(x);

    // loop through all incoming edges
    
    int id = x;
    do
    {
        const std::pair<int,int> loc = ConvertVertexToLocation(id);
        if (loc.second > 1)
        {
            int predecessor = GetClusterRepresentative(id-1);

            // continue to search affected region

            if (vertices[predecessor].time_stamp != time_stamp && vertices[predecessor].ord > vertices[y].ord)
            {
                BackwardDFS(predecessor, y);
            }
        }
        id = vertices[id].next;
    }
    while (id != x);
}

//////////////////////////////////////////////////////////////////////
// Annealing::Reorder()
//
// Reorder elements for topological ordering.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::Reorder()
{
    TopologicalOrderComparator comparator(vertices);    
    std::sort(forward.begin(), forward.end(), comparator);
    std::sort(backward.begin(), backward.end(), comparator);
    L.clear();
    R.resize(backward.size() + forward.size());

    for (size_t i = 0; i < backward.size(); i++)
    {
        L.push_back(backward[i]);
        backward[i] = vertices[L.back()].ord;
    }
    for (size_t i = 0; i < forward.size(); i++)
    {
        L.push_back(forward[i]);
        forward[i] = vertices[L.back()].ord;
    }
    std::merge(backward.begin(), backward.end(), forward.begin(), forward.end(), R.begin());
    for (size_t i = 0; i < L.size(); i++)
    {
        vertices[L[i]].ord = R[i];
    }
}

//////////////////////////////////////////////////////////////////////
// Annealing::Merge()
//
// Merge vertices.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::Merge(const int x, const int y)
{
    Assert(vertices[x].ord < vertices[y].ord, "Vertex x should come before vertex y.");
    
    // merge cycles

    const int x_next = vertices[x].next;
    const int y_next = vertices[y].next;

    vertices[x].next = y_next;
    vertices[y].next = x_next;    
}


//////////////////////////////////////////////////////////////////////
// Annealing::UseEdge()
//
// Use edge to merge vertices if no cycle results.  In particular,
// suppose without loss of generality that ord[y] < ord[x].  To merge
// x and y, it suffices to ensure that introducing a (x,y) edge does
// not generate cycles.
//////////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::UseEdge(const Edge<RealT> &edge)
{
    // ensure ord[y] < ord[x]
    
    int x = edge.id1;
    int y = edge.id2;
    if (vertices[x].ord < vertices[y].ord) std::swap(x, y);

    Assert(x == GetClusterRepresentative(x), "Can only operator on cluster representatives.");
    Assert(y == GetClusterRepresentative(y), "Can only operator on cluster representatives.");
    Assert(vertices[edge.id1].ord != vertices[edge.id2].ord, "Should not be equal.");

    // initialization

    ++time_stamp;
    cycle_found = false;
    forward.clear();
    backward.clear();

    // perform forward DFS from y
    
    ForwardDFS(x, y);
    if (cycle_found) return;

    // perform backward DFS from x

    BackwardDFS(x, y);

    // reorder affected region

    const std::pair<int,int> loc1 = ConvertVertexToLocation(x);
    const std::pair<int,int> loc2 = ConvertVertexToLocation(y);
    Reorder();

    // merge x and y

    Merge(x,y);
}

/////////////////////////////////////////////////////////////////
// Annealing::DoAlignment()
//
// Perform sequence annealing.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::DoAlignment()
{
    if (toggle_verbose)
    {
        WriteProgressMessage("Running annealing algorithm...");
    }
    
    InitializeVertices();
    InitializeEdges();

    int num_edges = int(edges.size());

    // repeat until no more candidate edges

    while (edges.size() > 0)
    {
        // update progress
        
        if (toggle_verbose)
        {
            static int old_percentage = -1;
            int new_percentage = 100 * (num_edges - edges.size()) / num_edges;
            if (new_percentage != old_percentage)
            {
                WriteProgressMessage(SPrintF("Annealing %d percent complete.", new_percentage));
                old_percentage = new_percentage;
            }
        }
        
        // recompute weight for current edge

        std::pop_heap(edges.begin(), edges.end());
        edges.rbegin()->id1 = GetClusterRepresentative(edges.rbegin()->id1);
        edges.rbegin()->id2 = GetClusterRepresentative(edges.rbegin()->id2);
        edges.rbegin()->weight = ComputeWeight(edges.rbegin()->id1, edges.rbegin()->id2);

        // reinsert edge if not highest weight
        
        if (edges.rbegin()->weight < edges[0].weight)
        {
            std::push_heap(edges.begin(), edges.end());
            continue;
        }

        // check if edge is valid
        
        if (edges.rbegin()->weight != INVALID_EDGE)
        {
            UseEdge(*edges.rbegin());

            /*
            // show step-by-step
            
            if (!cycle_found)
            {
                std::vector<Vertex> vertices_copy(vertices);
                FinalizeOrdering(vertices_copy);
                FormAlignment(vertices_copy);
                
                std::cerr << "Added edge (" << edges.rbegin()->id1 << "," << edges.rbegin()->id2 << "): " << edges.rbegin()->weight << std::endl;
                alignment->WriteMFA(std::cerr);
                std::cerr << std::endl;
            }
            else
            {
                std::cerr << "Cycle found." << std::endl;
            }
            */
                
        }
        /*
        else
        {
            std::cerr << "Invalid edge." << std::endl;
        }
        */
        edges.pop_back();
    }

    if (toggle_verbose)
    {
        WriteProgressMessage("Finishing annealing algorithm...");
    }

    FinalizeOrdering(vertices);
    FormAlignment(vertices);

    if (toggle_verbose)
    {
        WriteProgressMessage("");
    }
}

/////////////////////////////////////////////////////////////////
// Annealing::InitializeVertices()
//
// Generate separate vertex for each nucleotide.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::InitializeVertices()
{
    const int m = multi_seqs.GetNumSequences();

    // count number of vertices
    
    int num_vertices = 0;
    for (int i = 0; i < m; i++)
        num_vertices += multi_seqs.GetSequence(i).GetLength();

    // create vertices

    vertices.resize(num_vertices);

    int c = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 1; j <= multi_seqs.GetSequence(i).GetLength(); j++)
        {
            vertices[c].ord = c;
            vertices[c].next = c;
            vertices[c].time_stamp = 0;
            ++c;
        }
    }

    Assert(c == num_vertices, "Should be equal.");
}

/////////////////////////////////////////////////////////////////
// Annealing::InitializeEdges()
//
// Generate edges for each pair of nucleotides.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::InitializeEdges()
{
    const int m = multi_seqs.GetNumSequences();

    // count number of edges

    int num_edges = 0;
    for (int s = 0; s < m; s++)
        for (int t = s+1; t < m; t++)
            num_edges += posteriors[s*m+t]->GetNumEntries();

    // compute sequence offsets

    offsets.resize(m);
    
    offsets[0] = 0;
    for (int s = 0; s < m-1; s++)
        offsets[s+1] = offsets[s] + multi_seqs.GetSequence(s).GetLength();
    
    // create edges

    edges.resize(num_edges);
    
    int c = 0;
    for (int s = 0; s < m; s++)
    {
        for (int t = s+1; t < m; t++)
        {
            const SparseMatrix<RealT> &sparse = *posteriors[s*m+t];
            for (int i = 1; i < sparse.GetNumRows(); i++)
            {
                for (const SparseMatrixEntry<RealT> *iter = sparse.GetRowBegin(i); iter != sparse.GetRowEnd(i); ++iter)
                {
                    edges[c].id1 = offsets[s] + i - 1;
                    edges[c].id2 = offsets[t] + iter->column - 1;
                    edges[c].weight = iter->value;
                    ++c;
                }
            }
        }
    }

    Assert(c == num_edges, "Should be equal.");

    std::make_heap(edges.begin(), edges.end());
}

/////////////////////////////////////////////////////////////////
// Annealing::FinalizeOrdering()
//
// Transfer ordering from cluster representatives to all
// vertices.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::FinalizeOrdering(std::vector<Vertex> &vertices)
{
    ++time_stamp;
    for (int i = 0; i < int(vertices.size()); i++)
    {
        if (vertices[i].time_stamp == time_stamp) continue;
        
        const int new_ord = vertices[GetClusterRepresentative(i)].ord;
        
        int id = i;
        do
        {
            vertices[id].ord = new_ord;
            vertices[id].time_stamp = time_stamp;
            id = vertices[id].next;
        }
        while (id != i);
    }
}

/////////////////////////////////////////////////////////////////
// Annealing::FormAlignment()
//
// Convert graph representation into alignment.
/////////////////////////////////////////////////////////////////

template<class RealT>
void Annealing<RealT>::FormAlignment(std::vector<Vertex> &vertices)
{
    std::sort(vertices.begin(), vertices.end());

    int curr_ord = -1;
    std::vector<std::string> res(multi_seqs.GetNumSequences(), "@");

    for (size_t i = 0; i < vertices.size(); i++)
    {
        if (vertices[i].ord != curr_ord)
        {
            for (size_t j = 0; j < res.size(); j++)
                res[j].push_back('-');
            curr_ord = vertices[i].ord;
        }

        std::pair<int,int> loc = ConvertVertexToLocation(vertices[i].next);
        res[loc.first][res[loc.first].length()-1] = multi_seqs.GetSequence(loc.first).GetData()[loc.second];
    }

    delete alignment;
    alignment = new MultiSequence();
    for (size_t i = 0; i < res.size(); i++)
        alignment->AddSequence(new Sequence(res[i], multi_seqs.GetSequence(i).GetName(), multi_seqs.GetSequence(i).GetID()));
}

/////////////////////////////////////////////////////////////////
// Annealing::GetAlignment()
//
// Get final alignment.
/////////////////////////////////////////////////////////////////

template<class RealT>
const MultiSequence &Annealing<RealT>::GetAlignment()
{
    return *alignment;
}
