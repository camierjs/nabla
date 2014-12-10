// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
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
