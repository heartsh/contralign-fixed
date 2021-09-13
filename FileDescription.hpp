//////////////////////////////////////////////////////////////////////
// FileDescription.hpp
//
// Contains a description of a file for processing.
//////////////////////////////////////////////////////////////////////

#ifndef FILEDESCRIPTION_HPP
#define FILEDESCRIPTION_HPP

#include <string>
#include "Config.hpp"
#include "MultiSequence.hpp"

//////////////////////////////////////////////////////////////////////
// struct FileDescription
//////////////////////////////////////////////////////////////////////

struct FileDescription
{
    MultiSequence seqs;
    std::string input_filename;
    int size;
    double weight;
    
    // constructors, assignment operator, destructor
    FileDescription();
    FileDescription(const std::string &input_filename); 
    FileDescription(const FileDescription &rhs);
    FileDescription &operator=(const FileDescription &rhs);
    ~FileDescription();

    // comparator for sorting by decreasing size
    bool operator<(const FileDescription &rhs) const;
};

#endif
