//////////////////////////////////////////////////////////////////////
// FileDescription.cpp
//////////////////////////////////////////////////////////////////////

#include "FileDescription.hpp"

//////////////////////////////////////////////////////////////////////
// FileDescription::FileDescription()
//
// Default constructor.
//////////////////////////////////////////////////////////////////////

FileDescription::FileDescription() :
    seqs(),
    input_filename(""),
    size(0),
    weight(0.0)
{}
    
//////////////////////////////////////////////////////////////////////
// FileDescription::~FileDescription()
//
// Destructor.
//////////////////////////////////////////////////////////////////////

FileDescription::~FileDescription()
{}

//////////////////////////////////////////////////////////////////////
// FileDescription::FileDescription()
//
// Constructor.
//////////////////////////////////////////////////////////////////////

FileDescription::FileDescription(const std::string &input_filename) :
    seqs(input_filename, false),
    input_filename(input_filename),
    weight(1.0)
{
    size = 0;
    for (int i = 0; i < seqs.GetNumSequences(); i++)
    {
        size = std::max(size, seqs.GetSequence(i).GetLength());
    }
    size *= size;
    
    if (seqs.GetNumSequences() < 2)
        Warning("File contains fewer than two sequences.");
}

//////////////////////////////////////////////////////////////////////
// FileDescription::FileDescription()
//
// Copy constructor.
//////////////////////////////////////////////////////////////////////

FileDescription::FileDescription(const FileDescription &rhs) :
    seqs(rhs.seqs),
    input_filename(rhs.input_filename), 
    size(rhs.size),
    weight(rhs.weight)
{}

//////////////////////////////////////////////////////////////////////
// FileDescription::operator=()
//
// Assignment operator.
//////////////////////////////////////////////////////////////////////

FileDescription &FileDescription::operator=(const FileDescription &rhs)
{
    if (this != &rhs)
    {
        seqs = rhs.seqs;
        input_filename = rhs.input_filename;
        size = rhs.size;
        weight = rhs.weight;
    }
    return *this;
}

//////////////////////////////////////////////////////////////////////
// FileDescription::operator<()
//
// Comparator used to sort by decreasing size.
//////////////////////////////////////////////////////////////////////

bool FileDescription::operator<(const FileDescription &rhs) const 
{
    return size > rhs.size;
}

