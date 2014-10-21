/*---------------------------------------------------------------------------*/
/* ArcaneTypes.h                                               (C) 2000-2013 */
/*                                                                           */
/* D�finition des types g�n�raux de Arcane.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ARCANETYPES_H
#define ARCANE_ARCANETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/datatype/DataTypes.h"

//! Genre d'entit� de maillage
enum eItemKind{
  IK_Node     = 0, //!< Entit� de maillage de genre noeud
  IK_Edge     = 1, //!< Entit� de maillage de genre ar�te
  IK_Face     = 2, //!< Entit� de maillage de genre face
  IK_Cell     = 3, //!< Entit� de maillage de genre maille
  IK_DualNode = 4, //!< Entit� de maillage de genre noeuds dual d'un graphe
  IK_Link     = 5, //!< Entit� de maillage de genre lien d'un graphe
  IK_Particle = 6, //!< Entit� de maillage de genre particule
  IK_Unknown  = 7  //!< Entit� de maillage de genre inconnu ou non initialis�
};

//! Nombre de genre d'entit�s de maillage.
static const Integer NB_ITEM_KIND = 6;

//! Nom du genre d'entit�.
extern "C++" const char* itemKindName(eItemKind kind);

//! Op�rateur de sortie sur un flot
extern "C++" ostream& operator<< (ostream& ostr,eItemKind item_kind);

//! Op�rateur d'entr�e depuis un flot
extern "C++" istream& operator>> (istream& istr,eItemKind& item_kind);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Num�ro correspondant � une entit� nulle.
  \deprecated.
*/
static const Integer NULL_ITEM_ID = static_cast<Integer>(-1);

//! Num�ro correspondant � une entit� nulle
static const Integer NULL_ITEM_LOCAL_ID = static_cast<Integer>(-1);

//! Num�ro correspondant � une entit� nulle
static const Int64 NULL_ITEM_UNIQUE_ID = static_cast<Int64>(-1);

//! Num�ro correspondant � un sous-domaine nul
static const Integer NULL_SUB_DOMAIN_ID = static_cast<Integer>(-1);

//! Num�ro correspondant � un rang nul
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

//! Num�ro du type d'entit� inconnu ou null
static const Integer IT_NullType = 0;
//! Num�ro du type d'entit� Noeud (1 sommet 1D, 2D et 3D)
static const Integer IT_Vertex = 1;
//! Num�ro du type d'entit� Ar�te (2 sommets, 1D, 2D et 3D)
static const Integer IT_Line2 = 2;
//! Num�ro du type d'entit� Triangle (3 sommets, 2D)
static const Integer IT_Triangle3 = 3;
//! Num�ro du type d'entit� Quadrilat�re (4 sommets, 2D)
static const Integer IT_Quad4 = 4;
//! Num�ro du type d'entit� Pentagone (5 sommets, 2D)
static const Integer IT_Pentagon5 = 5;
//! Num�ro du type d'entit� Hexagone (6 sommets, 2D)
static const Integer IT_Hexagon6 = 6;
//! Num�ro du type d'entit� Tetra�dre (4 sommets, 3D)
static const Integer IT_Tetraedron4 = 7;
//! Num�ro du type d'entit� Pyramide (5 sommets, 3D)
static const Integer IT_Pyramid5 = 8;
//! Num�ro du type d'entit� Prisme (6 sommets, 3D)
static const Integer IT_Pentaedron6 = 9;
//! Num�ro du type d'entit� Hexa�dre (8 sommets, 3D)
static const Integer IT_Hexaedron8 = 10;
//! Num�ro du type d'entit� Hepta�dre (prisme � base pentagonale)
static const Integer IT_Heptaedron10 = 11;
//! Num�ro du type d'entit� Octa�dre (prisme � base hexagonale)
static const Integer IT_Octaedron12 = 12;
//! Num�ro du type d'entit� HemiHexa7 (h�xah�dre � 1 d�g�n�rescence)
static const Integer IT_HemiHexa7 = 13;
//! Num�ro du type d'entit� HemiHexa6 (h�xah�dre � 2 d�g�n�rescences non contigues)
static const Integer IT_HemiHexa6 = 14;
//! Num�ro du type d'entit� HemiHexa5 (h�xah�dres � 3 d�g�n�rescences non contigues)
static const Integer IT_HemiHexa5 = 15;
//! Num�ro du type d'entit� AntiWedgeLeft6 (h�xah�dre � 2 d�g�n�rescences contigues)
static const Integer IT_AntiWedgeLeft6 = 16;
//! Num�ro du type d'entit� AntiWedgeRight6 (h�xah�dre � 2 d�g�n�rescences contigues (seconde forme))
static const Integer IT_AntiWedgeRight6 = 17;
//! Num�ro du type d'entit� DiTetra5 (h�xah�dre � 3 d�g�n�rescences orthogonales)
static const Integer IT_DiTetra5 = 18;
//! Numero du type d'entite noeud dual d'un sommet
static const Integer IT_DualNode = 19;
//! Numero du type d'entite noeud dual d'une ar�te
static const Integer IT_DualEdge = 20;
//! Numero du type d'entite noeud dual d'une face
static const Integer IT_DualFace = 21;
//! Numero du type d'entite noeud dual d'une cellule
static const Integer IT_DualCell = 22;
//! Num�ro du type d'entit� liaison
static const Integer IT_Link = 23;
//! Num�ro du type d'entit� Face pour les maillages 1D.
static const Integer IT_FaceVertex = 24;
//! Num�ro du type d'entit� Cell pour les maillages 1D.
static const Integer IT_CellLine2 = 25;
//! Numero du type d'entite noeud dual d'une particule
static const Integer IT_DualParticle = 26;

//! Nombre de types d'entit�s disponible par d�faut
static const Integer NB_BASIC_ITEM_TYPE = 27;

extern "C++" eItemKind dualItemKind(Integer type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Phase d'une action temporelle.
 */
enum eTimePhase{
  TP_Computation = 0,
  TP_Communication,
  TP_InputOutput
};
static const Integer NB_TIME_PHASE = 3;

//! Op�rateur de sortie sur un flot
extern "C++" ostream& operator<< (ostream& ostr,eTimePhase time_phase);

//! Op�rateur d'entr�e depuis un flot
extern "C++" istream& operator>> (istream& istr,eTimePhase& time_phase);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type de la direction pour un maillage structur�
enum eMeshDirection{
  //! Direction X
  MD_DirX = 0,
  //! Direction Y
  MD_DirY = 1,
  //! Direction Z
  MD_DirZ = 2,
  //! Direction invalide ou non initialis�e
  MD_DirInvalid = (-1)
};

//! Op�rateur de sortie sur un flot
extern "C++" ostream& operator<<(ostream& o,eMeshDirection md);

#endif
