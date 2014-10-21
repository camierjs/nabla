#ifndef ALEPH_IALEPHFACTORY_H
#define ALEPH_IALEPHFACTORY_H


#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/TraceInfo.h"
//#include "arcane/utils/String.h"
//#include "arcane/NotSupportedException.h"
//#include "arcane/NotImplementedException.h"

#include "AlephGlobal.h"
#include "AlephInterface.h"

#include <map>
#include <string>

class IApplication;

class AlephFactory: public IAlephFactory{
 private:
  struct FactoryImpl{
   public:
    FactoryImpl(const String& name) : m_factory(0),
                                      m_name(name),
                                      m_initialized(false){}
   public:
    IAlephFactoryImpl* m_factory;
    String m_name;
    Bool m_initialized;
  };
 public:
  AlephFactory(IApplication* app,ITraceMng *tm);
  ~AlephFactory();
 public:
  IAlephTopology* GetTopology(AlephKernel *kernel, Integer index, Integer nb_row_size);
  IAlephVector* GetVector(AlephKernel *kernel, Integer index);
  IAlephMatrix* GetMatrix(AlephKernel *kernel, Integer index);
 private:
  typedef std::map<Integer,FactoryImpl*> FactoryImplMap;
  FactoryImplMap m_impl_map;
  IAlephFactoryImpl* _getFactory(Integer solver_index);
};

#endif 
