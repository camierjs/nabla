/*---------------------------------------------------------------------------*/
/* AlephOrdering.h                                                  (C) 2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_ORDERING_H
#define ALEPH_ORDERING_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*!
 * \brief Gestionaire de reordering
 */
class AlephOrdering: public TraceAccessor{
public:
  AlephOrdering(AlephKernel*);
  AlephOrdering(AlephKernel*,Integer,Integer,bool=false);
  ~AlephOrdering();
public:
  inline Integer swap(Integer i){
    if (m_do_swap) return m_swap.at(i);
    return i;
  }
private:
  void initCellOrder(void);
  void initTwiceCellOrder(void);
  void initFaceOrder(void);
  void initCellFaceOrder(void);
  void initCellNodeOrder(void);
  void initTwiceCellNodeOrder(void);
private:
  bool m_do_swap;
  AlephKernel* m_kernel;
private:
  Array<Int64> m_swap;
};
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

