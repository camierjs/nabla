/*---------------------------------------------------------------------------*/
/* ISubDomain.h                                                (C) 2000-2014 */
/*                                                                           */
/* Interface d'un sous-domaine.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ISUBDOMAIN_H
#define ARCANE_ISUBDOMAIN_H

class IApplication;
class CommonVariables;

class ISubDomain{
 protected:
  virtual ~ISubDomain() {} //!< Libère les ressources.

 public:
  //virtual void destroy() =0;
  virtual IMesh* defaultMesh() =0;

 public:

  //! Retourne le gestionnaire de parallélisme
  virtual IParallelMng* parallelMng() =0;
  virtual IApplication* application() =0;
  virtual const CommonVariables& commonVariables() const =0;

};

#endif  

