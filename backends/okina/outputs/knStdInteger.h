#ifndef _KN_STD_INTEGER_H_
#define _KN_STD_INTEGER_H_

// ****************************************************************************
// * Standard integer
// ****************************************************************************
class integer {
public:
  int vec;
public:
  // Constructors
  inline integer():vec(0){}
  inline integer(int i):vec(i){}
  //inline integer(int i3, int i2, int i1, int i0){vec=_mm_set_epi32(i3, i2, i1, i0);}
  
  // Convertors
  inline operator int() const { return vec; }

  //inline integer operator&(const integer &b) { return (vec&b); }
  //inline integer operator|(const integer &b) { return (vec|b); }
  //inline integer operator^(const integer &b) { return (vec^b); }
      
  inline integer& operator&=(const integer &a) { return *this = (integer) (vec&a); }
  inline integer& operator|=(const integer &a) { return *this = (integer) (vec|a); }
  inline integer& operator^=(const integer &a) { return *this = (integer) (vec^a); }

  inline integer& operator+=(const integer &a) { return *this = (integer)(vec+a); }
  inline integer& operator-=(const integer &a) { return *this = (integer)(vec-a); }   

  //friend inline bool operator==(const integer &a, const int i){ return a==i; }
};

//inline integer operator&(const integer &a, const integer &b) { return (a&b); }
//inline integer operator|(const integer &a, const integer &b) { return (a|b); }
//inline integer operator^(const integer &a, const integer &b) { return (a^b); }

//inline integer operator&(const integer &a, const int &b) { return integer(a&b); }
//inline integer operator|(const integer &a, const int &b) { return integer(a|b); }
//inline integer operator^(const integer &a, const int &b) { return integer(a^b); }

#endif //  _KN_STD_INTEGER_H_
