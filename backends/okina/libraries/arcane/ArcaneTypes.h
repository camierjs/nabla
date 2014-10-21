/*---------------------------------------------------------------------------*/
/* ArcaneTypes.h                                               (C) 2000-2013 */
/*                                                                           */
/* Définition des types généraux de Arcane.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ARCANETYPES_H
#define ARCANE_ARCANETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/datatype/DataTypes.h"

//! Genre d'entité de maillage
enum eItemKind{
  IK_Node     = 0, //!< Entité de maillage de genre noeud
  IK_Edge     = 1, //!< Entité de maillage de genre arête
  IK_Face     = 2, //!< Entité de maillage de genre face
  IK_Cell     = 3, //!< Entité de maillage de genre maille
  IK_DualNode = 4, //!< Entité de maillage de genre noeuds dual d'un graphe
  IK_Link     = 5, //!< Entité de maillage de genre lien d'un graphe
  IK_Particle = 6, //!< Entité de maillage de genre particule
  IK_Unknown  = 7  //!< Entité de maillage de genre inconnu ou non initialisé
};

//! Nombre de genre d'entités de maillage.
static const Integer NB_ITEM_KIND = 6;

//! Nom du genre d'entité.
extern "C++" const char* itemKindName(eItemKind kind);

//! Opérateur de sortie sur un flot
extern "C++" ostream& operator<< (ostream& ostr,eItemKind item_kind);

//! Opérateur d'entrée depuis un flot
extern "C++" istream& operator>> (istream& istr,eItemKind& item_kind);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Numéro correspondant à une entité nulle.
  \deprecated.
*/
static const Integer NULL_ITEM_ID = static_cast<Integer>(-1);

//! Numéro correspondant à une entité nulle
static const Integer NULL_ITEM_LOCAL_ID = static_cast<Integer>(-1);

//! Numéro correspondant à une entité nulle
static const Int64 NULL_ITEM_UNIQUE_ID = static_cast<Int64>(-1);

//! Numéro correspondant à un sous-domaine nul
static const Integer NULL_SUB_DOMAIN_ID = static_cast<Integer>(-1);

//! Numéro correspondant à un rang nul
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

//! Numéro du type d'entité inconnu ou null
static const Integer IT_NullType = 0;
//! Numéro du type d'entité Noeud (1 sommet 1D, 2D et 3D)
static const Integer IT_Vertex = 1;
//! Numéro du type d'entité Arête (2 sommets, 1D, 2D et 3D)
static const Integer IT_Line2 = 2;
//! Numéro du type d'entité Triangle (3 sommets, 2D)
static const Integer IT_Triangle3 = 3;
//! Numéro du type d'entité Quadrilatère (4 sommets, 2D)
static const Integer IT_Quad4 = 4;
//! Numéro du type d'entité Pentagone (5 sommets, 2D)
static const Integer IT_Pentagon5 = 5;
//! Numéro du type d'entité Hexagone (6 sommets, 2D)
static const Integer IT_Hexagon6 = 6;
//! Numéro du type d'entité Tetraèdre (4 sommets, 3D)
static const Integer IT_Tetraedron4 = 7;
//! Numéro du type d'entité Pyramide (5 sommets, 3D)
static const Integer IT_Pyramid5 = 8;
//! Numéro du type d'entité Prisme (6 sommets, 3D)
static const Integer IT_Pentaedron6 = 9;
//! Numéro du type d'entité Hexaèdre (8 sommets, 3D)
static const Integer IT_Hexaedron8 = 10;
//! Numéro du type d'entité Heptaèdre (prisme à base pentagonale)
static const Integer IT_Heptaedron10 = 11;
//! Numéro du type d'entité Octaèdre (prisme à base hexagonale)
static const Integer IT_Octaedron12 = 12;
//! Numéro du type d'entité HemiHexa7 (héxahèdre à 1 dégénérescence)
static const Integer IT_HemiHexa7 = 13;
//! Numéro du type d'entité HemiHexa6 (héxahèdre à 2 dégénérescences non contigues)
static const Integer IT_HemiHexa6 = 14;
//! Numéro du type d'entité HemiHexa5 (héxahèdres à 3 dégénérescences non contigues)
static const Integer IT_HemiHexa5 = 15;
//! Numéro du type d'entité AntiWedgeLeft6 (héxahèdre à 2 dégénérescences contigues)
static const Integer IT_AntiWedgeLeft6 = 16;
//! Numéro du type d'entité AntiWedgeRight6 (héxahèdre à 2 dégénérescences contigues (seconde forme))
static const Integer IT_AntiWedgeRight6 = 17;
//! Numéro du type d'entité DiTetra5 (héxahèdre à 3 dégénérescences orthogonales)
static const Integer IT_DiTetra5 = 18;
//! Numero du type d'entite noeud dual d'un sommet
static const Integer IT_DualNode = 19;
//! Numero du type d'entite noeud dual d'une arête
static const Integer IT_DualEdge = 20;
//! Numero du type d'entite noeud dual d'une face
static const Integer IT_DualFace = 21;
//! Numero du type d'entite noeud dual d'une cellule
static const Integer IT_DualCell = 22;
//! Numéro du type d'entité liaison
static const Integer IT_Link = 23;
//! Numéro du type d'entité Face pour les maillages 1D.
static const Integer IT_FaceVertex = 24;
//! Numéro du type d'entité Cell pour les maillages 1D.
static const Integer IT_CellLine2 = 25;
//! Numero du type d'entite noeud dual d'une particule
static const Integer IT_DualParticle = 26;

//! Nombre de types d'entités disponible par défaut
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

//! Opérateur de sortie sur un flot
extern "C++" ostream& operator<< (ostream& ostr,eTimePhase time_phase);

//! Opérateur d'entrée depuis un flot
extern "C++" istream& operator>> (istream& istr,eTimePhase& time_phase);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type de la direction pour un maillage structuré
enum eMeshDirection{
  //! Direction X
  MD_DirX = 0,
  //! Direction Y
  MD_DirY = 1,
  //! Direction Z
  MD_DirZ = 2,
  //! Direction invalide ou non initialisée
  MD_DirInvalid = (-1)
};

//! Opérateur de sortie sur un flot
extern "C++" ostream& operator<<(ostream& o,eMeshDirection md);

#endif
