#ifndef ARCANE_COMMONVARIABLES_H
#define ARCANE_COMMONVARIABLES_H

//#include "arcane/utils/String.h"

//#include "arcane/VariableTypes.h"

//class ModuleMaster;

class CommonVariables{
 public:
  //! Construit les références des variables communes pour le module \a c
  //CommonVariables(IModule* c);
  //! Construit les références des variables communes pour le sous-domaine \a sd
  CommonVariables(ISubDomain* sd);
  virtual ~CommonVariables() {} //!< Libère les ressources.
 public:
	
  //! Numéro de l'itération courante
  Int32 globalIteration() const;
  //! Temps courant
  Real globalTime() const;
  //! Temps courant précédent.
  Real globalOldTime() const;
  //! Temps final de la simulation
  Real globalFinalTime() const;
  //! Delta T courant.
  Real globalDeltaT() const;
  //! Temps CPU utilisé (en seconde)
  Real globalCPUTime() const;
  //! Temps CPU utilisé précédent (en seconde)
  Real globalOldCPUTime() const;
  //! Temps horloge (elapsed) utilisé (en seconde)
  Real globalElapsedTime() const;
  //! Temps horloge (elapsed) utilisé précédent (en seconde)
  Real globalOldElapsedTime() const;
 public:
  Int32 m_global_iteration; //!< Itération courante
  Real m_global_time; //!< Temps actuel
  Real m_global_deltat; //!< Delta T global
  Real m_global_old_time; //!< Temps précédent le temps actuel
  Real m_global_old_deltat; //!< Delta T au temps précédent le temps global
  Real m_global_final_time; //!< Temps final du cas
  Real m_global_old_cpu_time; //!< Temps précédent CPU utilisé (en seconde)
  Real m_global_cpu_time; //!< Temps CPU utilisé (en seconde)
  Real m_global_old_elapsed_time; //!< Temps précédent horloge utilisé (en seconde)
  Real m_global_elapsed_time; //!< Temps horloge utilisé (en seconde)
};

#endif  

