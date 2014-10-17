/*---------------------------------------------------------------------------*/
/* AlephFactory.cc                                             (C) 2010-2013 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/aleph/IAlephFactory.h"
#include "arcane/ServiceBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/******************************************************************************
 * IAlephFactory::IAlephFactory
 *****************************************************************************/
AlephFactory::AlephFactory(IApplication* app,
                           ITraceMng *tm): IAlephFactory(tm){
  // Liste des impl�mentations possibles.
  // 0 est le choix automatique qui doit aller vers une des biblioth�ques suivantes:
  m_impl_map.insert(std::make_pair(1,new FactoryImpl("Sloop")));
  m_impl_map.insert(std::make_pair(2,new FactoryImpl("Hypre")));
  m_impl_map.insert(std::make_pair(3,new FactoryImpl("Trilinos")));
  m_impl_map.insert(std::make_pair(4,new FactoryImpl("Cuda")));
  m_impl_map.insert(std::make_pair(5,new FactoryImpl("PETSc")));
  ServiceBuilder<IAlephFactoryImpl> sb(app);
  // Pour chaque impl�mentation possible,
  // cr�� la fabrique correspondante si elle est disponible.
  for(FactoryImplMap::iterator i = m_impl_map.begin(); i!=m_impl_map.end(); ++i ){
    FactoryImpl *implementation=i->second;
    const String& name = implementation->m_name;
    debug()<<"\33[1;34m\t[AlephFactory] Adding "<<name<<" library..."<<"\33[0m";
    IAlephFactoryImpl *factory = sb.createInstance(name+"AlephFactory",SB_AllowNull);
    implementation->m_factory = factory;
  }
  debug()<<"\33[1;34m\t[AlephFactory] done"<<"\33[0m";
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
AlephFactory::~AlephFactory(){
  debug()<<"\33[1;34m\t[~AlephFactory] Destruction des fabriques"<<"\33[0m";
  FactoryImplMap::iterator i = m_impl_map.begin();
  for(;i!=m_impl_map.end();++i)
    delete i->second->m_factory;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
IAlephFactoryImpl* AlephFactory::_getFactory(Integer solver_index){
  FactoryImplMap::const_iterator ci = m_impl_map.find(solver_index);
  if (ci==m_impl_map.end())
    throw FatalErrorException(A_FUNCINFO,
                              String::format("Invalid solver index '{0}' for aleph factory",
                                             solver_index));
  FactoryImpl *implementation=ci->second;
  IAlephFactoryImpl* factory = implementation->m_factory;
  if (!factory)
    throw NotSupportedException(A_FUNCINFO,
                                String::format("Implementation for '{0}' not available",
                                               implementation->m_name));
  // Si la fabrique de l'impl�mentation consid�r�e n'a pas
  // �t� initialis�e, on le fait maintenant
  if (!implementation->m_initialized){
    debug()<< "\33[1;34m\t\t[_getFactory] initializing solver_index="
          << solver_index << " ..."<<"\33[0m";
    implementation->m_initialized=true;
    factory->initialize();
  }
  return factory;
}


/******************************************************************************
 * AlephFactory::GetTopology
 *****************************************************************************/
IAlephTopology* AlephFactory::GetTopology(AlephKernel *kernel,
                                          Integer index,
                                          Integer nb_row_size){
  debug()<<"\33[1;34m\t\t[IAlephFactory::GetTopology] Switch="<<kernel->underlyingSolver()<<"\33[0m";
  return _getFactory(kernel->underlyingSolver())->createTopology(traceMng(),
                                                                 kernel,
                                                                 index,
                                                                 nb_row_size);
}


/******************************************************************************
 * AlephFactory::GetVector
 *****************************************************************************/
IAlephVector* AlephFactory::GetVector(AlephKernel *kernel,
                                      Integer index){
  debug()<<"\33[1;34m\t\t[AlephFactory::GetVector] Switch="<<kernel->underlyingSolver()<<"\33[0m";
  return _getFactory(kernel->underlyingSolver())->createVector(traceMng(),
                                                               kernel,
                                                               index);
}

/******************************************************************************
 * AlephFactory::GetMatrix
 *****************************************************************************/
IAlephMatrix* AlephFactory::GetMatrix(AlephKernel *kernel,
                                      Integer index){
  debug()<<"\33[1;34m\t\t[AlephFactory::GetMatrix] Switch="<<kernel->underlyingSolver()<<"\33[0m";
  return _getFactory(kernel->underlyingSolver())->createMatrix(traceMng(),
                                                               kernel,
                                                               index);
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/