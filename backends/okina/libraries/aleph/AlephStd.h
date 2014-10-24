#ifndef _ALEPH_STD_CPP_
#define _ALEPH_STD_CPP_

#include <assert.h>
#include <string>
#include <iostream>
#include <vector>

using std::istream;
using std::ostream;

typedef int Int32;
typedef long Int64;
typedef unsigned int UInt32;
typedef unsigned long UInt64;
typedef bool Bool;
typedef double Real;
typedef Int32 Integer;

class ITraceMng;
class IApplication;
class CommonVariables;


class UniqueId{
public:
  UniqueId(int){}
  UniqueId(long){}
  const Int32 asInt32()const { return 0;}
  Int32 asInt32(){ return 0;}
  Int64 asInt64(){ return 0;}
};

class Item{
public:
  bool isOwn() const { return true; }
  UniqueId uniqueId(){return 0;}
  const UniqueId uniqueId()const {return 0l;}
};


class String:public std::string{
public:
  String(const char *cstr){}
public:
  const char *localstr() const { return this->c_str(); }
};


// *****************************************************************************
// * Std Array
// *****************************************************************************
template <typename T>
class Array:public std::vector<T>{
public:
  Array(){}
  Array(int){}
  Array(int,long int){}
public:
  void add(T elem){this->push_back(elem);}
  void setAt(int i, T elem){this->setAt(i,elem);}
  Array<T> view(){return *this;}
  Array<T> constView(){return *this;}
  void fill(T elem){this->assign(this->size(),elem);}
  void resize(Integer size){}
  void resize(std::vector<int>& sizes){m_sizes=sizes;}
  const T* unguardedBasePointer() const { return this->const_pointer; }
  T* begin() { return unguardedBasePointer();}
  T* unguardedBasePointer() { return &this->at(0); }
  std::vector<int>* unguardedBasePointers() { return &m_sizes; }
  void copy(Array<T> elems){}
  Array<T> subView(Integer begin,Integer size){return *this;}
  T& operator[](const Item itm) { return this->at(itm.uniqueId().asInt32());}
  T& operator[](int i) { return this->at(i);}
  const T& operator[](int i) const { return this->at(i);}
private:
  std::vector<int> m_sizes;
};

template<typename T> inline ostream&
operator<<(ostream& o, const Array<T>& val){
  for(Integer i=0, is=val.size(); i<is; ++i ){
    if (i!=0) o << ' ';
    o << '"' << val[i] << '"';
  }
  return o;
}


typedef Item* ItemEnumerator;

#define ArrayView Array
#define ConstArrayView Array
#define Int32ConstArrayView Array<int>
#define IntegerConstArrayView Array<int>
#define MultiArray2Int32 Array<Array<int> >
#define MultiArray2Real Array<Array<double> >

typedef Array<Item> group;

typedef group ItemGroup;
typedef group CellGroup;
typedef group FaceGroup;
typedef group NodeGroup;



//! Enumérateur générique d'un groupe de noeuds
#define ENUMERATE_NODE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

//! Enumérateur générique d'un groupe d'arêtes
#define ENUMERATE_EDGE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

//! Enumérateur générique d'un groupe de faces
#define ENUMERATE_FACE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

//! Enumérateur générique d'un groupe de mailles
#define ENUMERATE_CELL(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

#define ENUMERATE_ITEM(name,group)\
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

#define ENUMERATE_PARTICLE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )



class TraceMessage{
public:
TraceMessage(ostream *ostr, ITraceMng*m):
  m_stream(ostr), m_parent(m){}
 public:
  ostream* m_stream; 
  ITraceMng* m_parent;
};
template<class T> inline const TraceMessage& operator<<(const TraceMessage& o,const T& v){
  *o.m_stream << v;
  return o;
}


class ITraceMng{
public:
  virtual void flush() =0;
  virtual TraceMessage info() =0;
  virtual TraceMessage debug() =0;
  virtual TraceMessage error() =0;
  virtual TraceMessage fatal() =0;
  virtual TraceMessage warning() =0;
};


class TraceAccessor{
 public:
  TraceAccessor(ITraceMng* m);
  virtual ~TraceAccessor();
public:
  ITraceMng* traceMng() const;
  TraceMessage info() const;
  TraceMessage debug() const;
  TraceMessage warning() const;
 private:
  ITraceMng* m_trace;
};



#define A_FUNCINFO __PRETTY_FUNCTION__
class Exception: public std::exception{
 public:
  Exception(const char *name,const char *where){
    std::cerr << "** Exception: Debug mode activated. Execution paused.\n";
  }
  Exception(const char *where){
    std::cerr << "** Exception: Debug mode activated. Execution paused.\n";
  }
};
class ArgumentException: public Exception{
 public:
  ArgumentException(const char* where,const char* message): Exception(where,message){}
};
class FatalErrorException: public Exception{
 public:
  FatalErrorException(const char* where,const char* message): Exception(where,message){}
};
class NotImplementedException: public Exception{
 public:
  NotImplementedException(const char* where,const char* message): Exception(where,message){}
  NotImplementedException(const char* where): Exception(where){}
};


typedef unsigned char Byte;
typedef signed char SByte;
typedef unsigned short UChar;
typedef unsigned short UInt16;
typedef short Int16;
typedef float Single;

enum eDataInitialisationPolicy{
  DIP_None =0,
  DIP_InitWithDefault =1,
  DIP_InitWithNan
};

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



class IMesh{
 public:
  virtual CellGroup ownCells() =0;
  virtual FaceGroup ownFaces() =0;
  virtual NodeGroup ownNodes() =0;
};


namespace Parallel{
  enum eReduceType{
    ReduceMin,
    ReduceMax,
    ReduceSum
  };
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
  public:
    template<typename T> operator T*() const { return (T*)m_request.v; }
    void print(ostream& o) const;
  private:
    _Request m_request;
  };
}

class IVariable{
public:
  virtual const String& name() const =0;
  virtual IMesh* mesh() const =0;
  virtual eItemKind itemKind() const =0;
  virtual Integer dimension() const =0;
  virtual ItemGroup itemGroup() const =0;
};

class VariableRef{
public:
  virtual IVariable* variable() const =0;
};

class VariableBuildInfo{
 public:
  VariableBuildInfo(IMesh* mesh,const String& name,int property=0);
};


template <typename T>
class VariableItemT:public Array<T>{
public:
  VariableItemT(const VariableBuildInfo& b,eItemKind ik);
  String name() { return "name";}
  T& operator[](Array<Item>::iterator itm) { return this->at(0);}//itm.uniqueId().asInt32());}
  T& operator[](const Item itm) { return this->at(0);}
  virtual void synchronize();
};

typedef VariableItemT<int> VariableItemInt32;



class IParallelMng{
public:
  typedef Parallel::eReduceType eReduceType;
 public:
  typedef Parallel::Request Request;
 public:
  virtual void build() =0;

 public:
  virtual bool isParallel() const =0;
  virtual Int32 commRank() const =0;
  virtual Int32 commSize() const =0;
  virtual ITraceMng* traceMng() const =0;
  virtual void* getMPICommunicator() =0;
  virtual IParallelMng* worldParallelMng() const =0;
  virtual IParallelMng* createSubParallelMng(Int32ConstArrayView kept_ranks) =0;

  virtual void broadcast(Array<int> send_buf,Integer rank) =0;
  virtual void broadcast(Array<double> send_buf,Integer rank) =0;
  virtual void broadcast(Array<long unsigned int> send_buf,Integer rank) =0;
  
  virtual Request recv(Array<int> values,Integer rank,bool is_blocking) =0;
  virtual Request recv(Array<double> values,Integer rank,bool is_blocking) =0;
  
  virtual Request send(ConstArrayView<int> values,Integer rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<double> values,Integer rank,bool is_blocking) =0;
  
  virtual void waitAllRequests(ArrayView<Request> rvalues) =0;
  
  virtual void allGather(ConstArrayView<int> send_buf,ArrayView<int> recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<int> send_buf, Array<int>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<long int> send_buf, Array<long int>& recv_buf) =0;
  
  virtual int reduce(eReduceType rt,int v) =0;

};



class ISubDomain{
 public:
  virtual IMesh* defaultMesh() =0;
 public:
  virtual IParallelMng* parallelMng() =0;
  virtual IApplication* application() =0;
  virtual const CommonVariables& commonVariables() const =0;
};

class CommonVariables{
 public:
  CommonVariables(ISubDomain* sd);
  virtual ~CommonVariables() {}
 public:
  Int32 globalIteration() const;
 public:
  Int32 m_global_iteration; 
};


#define ALEPH_ASSERT(a,b) assert(a)

#endif  
