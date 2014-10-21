#ifndef ALEPH_INDEXING_H
#define ALEPH_INDEXING_H

class AlephIndexing: public TraceAccessor{
 public:
  AlephIndexing(AlephKernel*);
  ~AlephIndexing();
 public:
  Int32 updateKnownItems(VariableItemInt32*,const Item &);
  Int32 findWhichLidFromMapMap(IVariable*,const Item &);
  Integer get(const VariableRef&, const ItemEnumerator&);
  Integer get(const VariableRef&, const Item&);
  void buildIndexesFromAddress(void);
  void nowYouCanBuildTheTopology(AlephMatrix*,AlephVector*,AlephVector*);
 private:
  Integer localKnownItems(void);
 private:
  AlephKernel *m_kernel;
  ISubDomain *m_sub_domain;
  Integer m_current_idx;
  Int32 m_known_items_own;
  Array<Int32*> m_known_items_all_address;
  typedef std::map<IVariable*,VariableItemInt32*> VarMapIdx;
  VarMapIdx m_var_map_idx;
};

#endif  

