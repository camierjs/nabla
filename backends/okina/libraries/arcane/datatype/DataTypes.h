/*---------------------------------------------------------------------------*/
/* DataTypes.h                                                 (C) 2000-2005 */
/*                                                                           */
/* Définition des types liées aux données.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPES_DATATYPES_H
#define ARCANE_DATATYPES_DATATYPES_H


//! Type d'une donnée
enum eDataType
{
  DT_Byte = 0, //!< Donnée de type octet
  DT_Real, //!< Donnée de type réel
  DT_Int32, //!< Donnée de type entier 32 bits
  DT_Int64, //!< Donnée de type entier 64 bits
  DT_String, //!< Donnée de type chaîne de caractère unicode
  DT_Real2, //!< Donnée de type vecteur 2
  DT_Real3, //!< Donnée de type vecteur 3
  DT_Real2x2, //!< Donnée de type tenseur 3x3
  DT_Real3x3, //!< Donnée de type tenseur 3x3
  DT_Unknown  //!< Donnée de type inconnu ou non initilialisé
};

//! Nom du type de donnée.
extern "C++" const char* dataTypeName(eDataType type);

//! Trouve le type associé à \a name
extern "C++" eDataType dataTypeFromName(const char* name,bool& has_error);

//! Trouve le type associé à \a name. Envoie une exception en cas d'erreur
extern "C++" eDataType dataTypeFromName(const char* name);

//! Taille du type de donnée \a type (qui doit être différent de \a DT_String)
extern "C++" Integer dataTypeSize(eDataType type);

//! Opérateur de sortie sur un flot
extern "C++" ostream& operator<< (ostream& ostr,eDataType data_type);

//! Opérateur d'entrée depuis un flot
extern "C++" istream& operator>> (istream& istr,eDataType& data_type);


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type de politique d'initialisation possible pour une donnée.
 *
 * Par défaut, il s'agit de DIP_NoInit.
 */
enum eDataInitialisationPolicy{
  //! Pas d'initialisation forcée
  DIP_None =0,
  //! Initialisation avec le constructeur par défaut
  DIP_InitWithDefault =1,
  //! Initialisation avec des Not A Number (uniquement pour les Real*)
  DIP_InitWithNan
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Positionne la politique d'initialisation des variables.
extern "C++" void  setGlobalDataInitialisationPolicy(eDataInitialisationPolicy init_policy);

//! Récupère la politique d'initialisation des variables.
extern "C++" eDataInitialisationPolicy getGlobalDataInitialisationPolicy();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Type de trace possible
enum eTraceType{
  TT_None = 0,
  TT_Read = 1,
  TT_Write = 2
};

#endif  
