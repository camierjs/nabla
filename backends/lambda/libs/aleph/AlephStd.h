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
#include <sstream>
#include <fstream>

#define A_FUNCINFO __PRETTY_FUNCTION__

using std::ostream;
using std::ofstream;
using std::string;
using std::vector;

extern std::ofstream devNull;
extern std::ostream& debug();
extern std::ostream& info();


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

class ITraceMng{ public: void flush(){std::cout.flush();}};

class TraceAccessor{
public:
  TraceAccessor(ITraceMng* m):m_trace(m){}
public:
  ITraceMng* traceMng() const { return m_trace;}
  ostream& warning() const { return info();}
 private:
  ITraceMng* m_trace;
};


// ****************************************************************************
// * Item, Item's' & Mesh Stuff
// ****************************************************************************
enum eItemKind{
  IK_Node     = 0,
  IK_Edge     = 1,
  IK_Face     = 2,
  IK_Cell     = 3,
  IK_Particle = 4,
  IK_Unknown  = 5
};

class item{
public:
  item():uid(0){}
  item(int id):uid(id){}
public:
  bool isOwn() const { return true; }
  int uniqueId(){ return uid;}
private:
  int uid;
};

typedef vector<item> items;

#define ENUMERATE_GROUP(name,itms) \
  for(items::iterator name(itms.begin());name!=itms.end();++name)


// ****************************************************************************
// * Mesh class
// ****************************************************************************
class IMesh{
public:
  IMesh(int x, int y, int z):size_x(x),
                             size_y(y),
                             size_z(z),
                             uid_idx(0)
  {
    cells.resize(x*y*z);
    faces.resize(x*y*z);
    nodes.resize((x+1)*(y+1)*(z+1));
  }
public:
  int size(){ return size_x*size_y*size_z; }
  items ownCells() { return cells;}
  items ownFaces() { return faces;}
  items ownNodes() { return nodes;}
  void checkValidMeshFull(){}
public:
  int size_x;
  int size_y;
  int size_z;
private:
  int uid_idx;
  items cells;
  items faces;
  items nodes;
};


// ****************************************************************************
// * Variables Stuff
// ****************************************************************************
/*class IVariable{
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
  VariableItemT(eItemKind ik){}
  string name() { return "name";}
  T& operator[](items::iterator itm) { return this->at(itm->uniqueId());}
  T& operator[](const item itm) { return this->at(0);}
  void synchronize(){}
};

typedef VariableItemT<int> VariableItemInt;
*/

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

  virtual IParallelMng* worldParallelMng() =0;
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
  virtual double reduce(Parallel::eReduceType rt,double v) =0;
};

class ISubDomain{
public:
  ISubDomain(IMesh *m,
             IParallelMng *p):m_mesh(m),
                              m_parallel_mg(p){}
 public:
  IMesh* defaultMesh(){return m_mesh;}
  IParallelMng* parallelMng(){ return m_parallel_mg;}
private:
  IMesh *m_mesh;
  IParallelMng *m_parallel_mg;
};

// ****************************************************************************
// * IParallelMng Séquentiel
// ****************************************************************************
class SequentialMng:public IParallelMng{
public:
  SequentialMng(ITraceMng *t):m_trace_mng(t){}
public:
  bool isParallel() const { return false;}
  int commRank() const {debug()<<"\n\t\33[1;36m SequentialMng::commRank \33[0m"; return 0;}
  int commSize() const {debug()<<"\n\t\33[1;36m SequentialMng::commSize \33[0m"; return 1;}
  ITraceMng* traceMng() const { return m_trace_mng;}
  IParallelMng* worldParallelMng() {
    //throw std::logic_error("SequentialMng::worldParallelMng");
    return this;
  }
  IParallelMng* createSubParallelMng(vector<int> kept_ranks){
    //throw std::logic_error("SequentialMng::createSubParallelMng");
    return this;
  }

  void broadcast(vector<int> send_buf,int rank) {throw std::logic_error("SequentialMng::");}
  void broadcast(vector<double> send_buf,int rank) {throw std::logic_error("SequentialMng::");}
  void broadcast(vector<long unsigned int> send_buf,int rank) {throw std::logic_error("SequentialMng::");}
  
  Parallel::Request recv(vector<int> values,int rank,bool is_blocking) {
    throw std::logic_error("SequentialMng::");
    return Parallel::Request();
  }
  Parallel::Request recv(double* values,int rank,bool is_blocking) {
    throw std::logic_error("SequentialMng::");
    return Parallel::Request();
  }
  Parallel::Request recv(vector<double> values,int rank,bool is_blocking) {
    throw std::logic_error("SequentialMng::");
    return Parallel::Request();
  }
  
  Parallel::Request send(vector<int> values,int rank,bool is_blocking) {
    throw std::logic_error("SequentialMng::");
    return Parallel::Request();
  }
  Parallel::Request send(double* values,int rank,bool is_blocking) {
    throw std::logic_error("SequentialMng::");
    return Parallel::Request();
  }
  Parallel::Request send(vector<double> values,int rank,bool is_blocking) {
    throw std::logic_error("SequentialMng::");
    return Parallel::Request();
  }
  
  void waitAllRequests(vector<Parallel::Request> rvalues) {throw std::logic_error("SequentialMng::");}
  
  void allGather(vector<int> send_buf,vector<int> recv_buf) {throw std::logic_error("SequentialMng::");}
  void allGatherVariable(vector<int> send_buf, vector<int>& recv_buf) {throw std::logic_error("SequentialMng::");}
  
  int reduce(Parallel::eReduceType rt,int v){
    debug()<<"\t[SequentialMng::reduce] (int)v="<<v;
    //throw std::logic_error("SequentialMng::");
    return v;
  }
  double reduce(Parallel::eReduceType rt,double v){
    debug()<<"\t[SequentialMng::reduce] (double) v="<<v;
    //throw std::logic_error("SequentialMng::");
    return v;
  }
private:
  ITraceMng *m_trace_mng;
};

#endif  
