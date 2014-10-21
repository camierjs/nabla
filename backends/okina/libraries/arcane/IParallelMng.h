/*---------------------------------------------------------------------------*/
/* IParallelMng.h                                              (C) 2000-2011 */
/*                                                                           */
/* Interface du gestionnaire du parallélisme sur un sous-domaine.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPARALLELMNG_H
#define ARCANE_IPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/utils/UtilsTypes.h"

//#include "arcane/Parallel.h"
//#include "arcane/VariableTypedef.h"

class IParallelMng{
public:
  typedef Parallel::eReduceType eReduceType;
public:
  virtual ~IParallelMng() {} //!< Libère les ressources.
 public:
  typedef Parallel::Request Request;
  //typedef Parallel::eReduceType eReduceType;
  //typedef Parallel::IStat IStat;
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

#endif  

