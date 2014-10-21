/*---------------------------------------------------------------------------*/
/* IMesh.h                                                     (C) 2000-2013 */
/*                                                                           */
/* Interface d'un maillage.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESH_H
#define ARCANE_IMESH_H

class IMesh{
 public:
  virtual ~IMesh() {} //<! Libère les ressources
 public:
  virtual void build() =0;
 public:

  //! Nom du maillage
  virtual const String& name() =0;

  //! Nombre de noeuds du maillage
  virtual Integer nbNode() =0;

  //! Nombre d'arêtes du maillage
  virtual Integer nbEdge() =0;

  //! Nombre de faces du maillage
  virtual Integer nbFace() =0;

  //! Nombre de mailles du maillage
  virtual Integer nbCell() =0;

  //! Nombre d'éléments du genre \a ik
  virtual Integer nbItem(eItemKind ik) =0;
  
  virtual CellGroup ownCells() =0;
  virtual FaceGroup ownFaces() =0;
  virtual NodeGroup ownNodes() =0;

};

#endif  
