#ifndef _OKINA_FAKE_ARCANE_
#define _OKINA_FAKE_ARCANE_

#include <assert.h>
#include <cstddef>
#include <iosfwd>
#include <limits.h>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <exception>
#include <iterator>

#define ARCANE_BEGIN_NAMESPACE
#define ARCANE_END_NAMESPACE

#define ARCANE_UTILS_EXPORT

using std::istream;
using std::ostream;
using std::ios;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::istringstream;
using std::string;
using std::iterator;

typedef int Int32;
typedef unsigned int UInt32;

typedef long Int64;
typedef unsigned long UInt64;

typedef void* Pointer;

typedef bool Bool;
typedef double Real;
typedef Int32 Integer;

typedef Int32 LocalIdType;
typedef Int64 UniqueIdType;

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


#include "arcane/utils/String.h"
//#define String string

#include "arcane/utils/Array.h"
//#define Array std::vector

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
#include "arcane/ItemEnumerator.h"

#include "arcane/utils/TraceMessage.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/utils/Exceptions.h"

#include "arcane/ArcaneTypes.h"

#include "arcane/IMesh.h"
#include "arcane/Parallel.h"


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

#include "arcane/IParallelMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/CommonVariables.h"

#define ARCANE_ASSERT(a,b) assert(a)

#endif  
