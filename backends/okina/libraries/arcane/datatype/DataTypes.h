/*---------------------------------------------------------------------------*/
/* DataTypes.h                                                 (C) 2000-2005 */
/*                                                                           */
/* D�finition des types li�es aux donn�es.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPES_DATATYPES_H
#define ARCANE_DATATYPES_DATATYPES_H


//! Type d'une donn�e
enum eDataType
{
  DT_Byte = 0, //!< Donn�e de type octet
  DT_Real, //!< Donn�e de type r�el
  DT_Int32, //!< Donn�e de type entier 32 bits
  DT_Int64, //!< Donn�e de type entier 64 bits
  DT_String, //!< Donn�e de type cha�ne de caract�re unicode
  DT_Real2, //!< Donn�e de type vecteur 2
  DT_Real3, //!< Donn�e de type vecteur 3
  DT_Real2x2, //!< Donn�e de type tenseur 3x3
  DT_Real3x3, //!< Donn�e de type tenseur 3x3
  DT_Unknown  //!< Donn�e de type inconnu ou non initilialis�
};

//! Nom du type de donn�e.
extern "C++" const char* dataTypeName(eDataType type);

//! Trouve le type associ� � \a name
extern "C++" eDataType dataTypeFromName(const char* name,bool& has_error);

//! Trouve le type associ� � \a name. Envoie une exception en cas d'erreur
extern "C++" eDataType dataTypeFromName(const char* name);

//! Taille du type de donn�e \a type (qui doit �tre diff�rent de \a DT_String)
extern "C++" Integer dataTypeSize(eDataType type);

//! Op�rateur de sortie sur un flot
extern "C++" ostream& operator<< (ostream& ostr,eDataType data_type);

//! Op�rateur d'entr�e depuis un flot
extern "C++" istream& operator>> (istream& istr,eDataType& data_type);


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type de politique d'initialisation possible pour une donn�e.
 *
 * Par d�faut, il s'agit de DIP_NoInit.
 */
enum eDataInitialisationPolicy{
  //! Pas d'initialisation forc�e
  DIP_None =0,
  //! Initialisation avec le constructeur par d�faut
  DIP_InitWithDefault =1,
  //! Initialisation avec des Not A Number (uniquement pour les Real*)
  DIP_InitWithNan
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Positionne la politique d'initialisation des variables.
extern "C++" void  setGlobalDataInitialisationPolicy(eDataInitialisationPolicy init_policy);

//! R�cup�re la politique d'initialisation des variables.
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
