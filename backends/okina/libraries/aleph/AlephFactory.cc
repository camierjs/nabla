#include "IAlephFactory.h"



/******************************************************************************
 * IAlephFactory::IAlephFactory
 *****************************************************************************/
AlephFactory::AlephFactory(ITraceMng *tm): IAlephFactory(tm){
  // Liste des implémentations possibles.
  // 0 est le choix automatique qui doit aller vers une des bibliothèques suivantes:
  m_impl_map.insert(std::make_pair(1,new FactoryImpl("Sloop")));
  m_impl_map.insert(std::make_pair(2,new FactoryImpl("Hypre")));
  m_impl_map.insert(std::make_pair(3,new FactoryImpl("Trilinos")));
  m_impl_map.insert(std::make_pair(4,new FactoryImpl("Cuda")));
  m_impl_map.insert(std::make_pair(5,new FactoryImpl("PETSc")));
  //ServiceBuilder<IAlephFactoryImpl> sb(app);
  // Pour chaque implémentation possible,
  // créé la fabrique correspondante si elle est disponible.
  for(FactoryImplMap::iterator i = m_impl_map.begin(); i!=m_impl_map.end(); ++i ){
    FactoryImpl *implementation=i->second;
    const String& name = implementation->m_name;
    debug()<<"\33[1;34m\t[AlephFactory] Adding "<<name<<" library..."<<"\33[0m";
    //IAlephFactoryImpl *factory = sb.createInstance(name+"AlephFactory",SB_AllowNull);
    //implementation->m_factory = factory;
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
    throw FatalErrorException("AlephFactory::_getFactory",
                              "Invalid solver index for aleph factory");
  FactoryImpl *implementation=ci->second;
  IAlephFactoryImpl* factory = implementation->m_factory;
  if (!factory)
    throw FatalErrorException("AlephFactory::_getFactory",
                              "Implementation not available");
  // Si la fabrique de l'implémentation considérée n'a pas
  // été initialisée, on le fait maintenant
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

