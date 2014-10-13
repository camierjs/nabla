/*---------------------------------------------------------------------------*/
/* AlephKappa.h                                                      (C) 2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_KAPPA_H
#define ALEPH_KAPPA_H 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class AlephKernel;
class AlephFactory;

class AlephKappaService
: public AbstractService
, public IDirectExecution
{
 public:
  AlephKappaService(const ServiceBuildInfo& sbi);
  ~AlephKappaService();
  virtual void build(void){}
 public:
  //! Ex�cute l'op�ration du service
  virtual void execute(void);
  //! Vrai si le service est actif
  virtual bool isActive(void) const { return true;}
  virtual void setParallelMng(IParallelMng *wpm){ m_world_parallel = wpm; }

 private:

  AlephKernel *m_kernel;
  IApplication* m_application;
  IParallelMng* m_world_parallel;
  Integer m_world_rank;
  Integer m_size;
  Integer m_world_size;
  AlephFactory *m_factory;
  Integer m_underlying_solver;
  Integer m_solver_size;
  bool m_reorder;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

