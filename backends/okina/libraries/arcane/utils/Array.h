#ifndef _OKINA_ARCANE_ARRAY_
#define _OKINA_ARCANE_ARRAY_


#include <memory>


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
  T* unguardedBasePointer() { return &this->at(0); }
  std::vector<int>* unguardedBasePointers() { return &m_sizes; }

  void copy(Array<T> elems){}
  
  Array<T> subView(Integer begin,Integer size){return *this;}
  T& operator[](const Item itm) { return this->at(itm.uniqueId().asInt32());}
  //T& operator[](Array<Item>::iterator itm_iterator) { return this->at(itm.uniqueId().asInt32());}
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


//template class Array<int>;
//template class Array<double>;

//typedef ArrayT<int> Array<int>;
//typedef Array<double> ArrayReal;

#endif
