#ifndef ARCANE_ITEMENUMERATOR_H
#define ARCANE_ITEMENUMERATOR_H

//! Enum�rateur g�n�rique d'un groupe de noeuds
#define ENUMERATE_NODE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

//! Enum�rateur g�n�rique d'un groupe d'ar�tes
#define ENUMERATE_EDGE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

//! Enum�rateur g�n�rique d'un groupe de faces
#define ENUMERATE_FACE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

//! Enum�rateur g�n�rique d'un groupe de mailles
#define ENUMERATE_CELL(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

#define ENUMERATE_ITEM(name,group)\
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

#define ENUMERATE_PARTICLE(name,group) \
  for(Array<Item>::iterator name((group).begin()); name!=group.end(); ++name )

#endif  
