///////////////////////////////////////////////////////////////////////////////
// NABLA - a Numerical Analysis Based LAnguage                               //
//                                                                           //
// Copyright (C) 2014~2015 CEA/DAM/DIF                                       //
// IDDN.FR.001.520002.000.S.P.2014.000.10500                                 //
//                                                                           //
// Contributor(s): CAMIER Jean-Sylvain - Jean-Sylvain.Camier@cea.fr          //
//                                                                           //
// This software is a computer program whose purpose is to translate         //
// numerical-analysis specific sources and to generate optimized code        //
// for different targets and architectures.                                  //
//                                                                           //
// This software is governed by the CeCILL license under French law and      //
// abiding by the rules of distribution of free software. You can  use,      //
// modify and/or redistribute the software under the terms of the CeCILL     //
// license as circulated by CEA, CNRS and INRIA at the following URL:        //
// "http://www.cecill.info".                                                 //
//                                                                           //
// The CeCILL is a free software license, explicitly compatible with         //
// the GNU GPL.                                                              //
//                                                                           //
// As a counterpart to the access to the source code and rights to copy,     //
// modify and redistribute granted by the license, users are provided only   //
// with a limited warranty and the software's author, the holder of the      //
// economic rights, and the successive licensors have only limited liability.//
//                                                                           //
// In this respect, the user's attention is drawn to the risks associated    //
// with loading, using, modifying and/or developing or reproducing the       //
// software by the user in light of its specific status of free software,    //
// that may mean that it is complicated to manipulate, and that also         //
// therefore means that it is reserved for developers and experienced        //
// professionals having in-depth computer knowledge. Users are therefore     //
// encouraged to load and test the software's suitability as regards their   //
// requirements in conditions enabling the security of their systems and/or  //
// data to be ensured and, more generally, to use and operate it in the      //
// same conditions as regards security.                                      //
//                                                                           //
// The fact that you are presently reading this means that you have had      //
// knowledge of the CeCILL license and that you accept its terms.            //
//                                                                           //
// See the LICENSE file for details.                                         //
///////////////////////////////////////////////////////////////////////////////
#ifndef _ALEPH_STD_CPP_
#define _ALEPH_STD_CPP_

#include <string>
#include <vector>
#include <assert.h>
#include <iostream>
#include <exception>
#include <stdexcept>

#define A_FUNCINFO __PRETTY_FUNCTION__

using std::ostream;
using std::string;
using std::vector; 


// ****************************************************************************
// * Trace Stuff
// ****************************************************************************
template<typename T> inline ostream& operator<<(ostream& o, const vector<T>& val){
  for(int i=0, is=val.size(); i<is; ++i ){
    if (i!=0) o << ' ';
    o << '"' << val[i] << '"';
  }
  return o;
}

class ITraceMng{
public:
  virtual void flush() =0;
  virtual ostream info() =0;
  virtual ostream debug() =0;
};

class TraceAccessor{
public:
  TraceAccessor(ITraceMng* m);
  virtual ~TraceAccessor();
public:
  ITraceMng* traceMng() const;
  ostream info() const;
  ostream debug() const;
  ostream warning() const;
};


// ****************************************************************************
// * Items & Mesh Stuff
// ****************************************************************************
enum eItemKind{
  IK_Node     = 0,
  IK_Edge     = 1,
  IK_Face     = 2,
  IK_Cell     = 3,
  IK_DualNode = 4,
  IK_Link     = 5,
  IK_Particle = 6,
  IK_Unknown  = 7
};

class item{
public:
  bool isOwn() const { return true; }
  int uniqueId(){ return uid;}
private:
  int uid;
};

typedef vector<item> items;

#define ENUMERATE_GROUP(name,itms) \
  for(items::iterator name(itms.begin()); name!=itms.end(); ++name )

class IMesh{
 public:
  virtual items ownCells() =0;
  virtual items ownFaces() =0;
  virtual items ownNodes() =0;
};


// ****************************************************************************
// * Variables Stuff
// ****************************************************************************
class IVariable{
public:
  virtual int dimension() const =0;
  virtual items itemGroup() const =0;
  virtual const string& name() const =0;
  virtual eItemKind itemKind() const =0;
};

class Variable{
public:
  virtual IVariable* variable() const =0;
};

template <typename T>
class VariableItemT:public vector<T>{
public:
  VariableItemT(eItemKind ik);
  string name() { return "name";}
  T& operator[](items::iterator itm) { return this->at(itm->uniqueId());}
  T& operator[](const item itm) { return this->at(0);}
  virtual void synchronize();
};

typedef VariableItemT<int> VariableItemInt;


// ****************************************************************************
// * Parallel Stuff
// ****************************************************************************
namespace Parallel{
  enum eReduceType{ReduceMin, ReduceMax, ReduceSum };
  class Request{
    union _Request{
      int i;
      long l;
      void* v;
      const void* cv;
    };
    enum Type {
      T_Int,
      T_Long,
      T_Ptr,
      T_Null
    };
  private:
    _Request m_request;
  };
}

class IParallelMng{
 public: 
  virtual bool isParallel() const =0;
  virtual int commRank() const =0;
  virtual int commSize() const =0;
  virtual ITraceMng* traceMng() const =0;

  virtual IParallelMng* worldParallelMng() const =0;
  virtual IParallelMng* createSubParallelMng(vector<int> kept_ranks) =0;

  virtual void broadcast(vector<int> send_buf,int rank) =0;
  virtual void broadcast(vector<double> send_buf,int rank) =0;
  virtual void broadcast(vector<long unsigned int> send_buf,int rank) =0;
  
  virtual Parallel::Request recv(vector<int> values,int rank,bool is_blocking) =0;
  virtual Parallel::Request recv(double* values,int rank,bool is_blocking) =0;
  virtual Parallel::Request recv(vector<double> values,int rank,bool is_blocking) =0;
  
  virtual Parallel::Request send(vector<int> values,int rank,bool is_blocking) =0;
  virtual Parallel::Request send(double* values,int rank,bool is_blocking) =0;
  virtual Parallel::Request send(vector<double> values,int rank,bool is_blocking) =0;
  
  virtual void waitAllRequests(vector<Parallel::Request> rvalues) =0;
  
  virtual void allGather(vector<int> send_buf,vector<int> recv_buf) =0;
  virtual void allGatherVariable(vector<int> send_buf, vector<int>& recv_buf) =0;
  
  virtual int reduce(Parallel::eReduceType rt,int v) =0;

};

class ISubDomain{
 public:
  virtual IMesh* defaultMesh() =0;
  virtual IParallelMng* parallelMng() =0;
};

#endif  
