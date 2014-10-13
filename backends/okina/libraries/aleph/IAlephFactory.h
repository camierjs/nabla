/*---------------------------------------------------------------------------*/
/* IAlephFactory.h                                             (C) 2000-2013 */
/*                                                                           */
/* Interface des fabriques pour Aleph.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ALEPH_IALEPHFACTORY_H
#define ARCANE_ALEPH_IALEPHFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/aleph/AlephGlobal.h"
#include "arcane/aleph/AlephInterface.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/******************************************************************************
 * IAlephFactory::IAlephFactory
 *****************************************************************************/
class ARCANE_ALEPH_EXPORT AlephFactory
: public IAlephFactory
{
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


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_IALEPH_FACTORY_H
