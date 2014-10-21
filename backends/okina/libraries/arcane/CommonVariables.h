#ifndef ARCANE_COMMONVARIABLES_H
#define ARCANE_COMMONVARIABLES_H

//#include "arcane/utils/String.h"

//#include "arcane/VariableTypes.h"

//class ModuleMaster;

class CommonVariables{
 public:
  //! Construit les r�f�rences des variables communes pour le module \a c
  //CommonVariables(IModule* c);
  //! Construit les r�f�rences des variables communes pour le sous-domaine \a sd
  CommonVariables(ISubDomain* sd);
  virtual ~CommonVariables() {} //!< Lib�re les ressources.
 public:
	
  //! Num�ro de l'it�ration courante
  Int32 globalIteration() const;
  //! Temps courant
  Real globalTime() const;
  //! Temps courant pr�c�dent.
  Real globalOldTime() const;
  //! Temps final de la simulation
  Real globalFinalTime() const;
  //! Delta T courant.
  Real globalDeltaT() const;
  //! Temps CPU utilis� (en seconde)
  Real globalCPUTime() const;
  //! Temps CPU utilis� pr�c�dent (en seconde)
  Real globalOldCPUTime() const;
  //! Temps horloge (elapsed) utilis� (en seconde)
  Real globalElapsedTime() const;
  //! Temps horloge (elapsed) utilis� pr�c�dent (en seconde)
  Real globalOldElapsedTime() const;
 public:
  Int32 m_global_iteration; //!< It�ration courante
  Real m_global_time; //!< Temps actuel
  Real m_global_deltat; //!< Delta T global
  Real m_global_old_time; //!< Temps pr�c�dent le temps actuel
  Real m_global_old_deltat; //!< Delta T au temps pr�c�dent le temps global
  Real m_global_final_time; //!< Temps final du cas
  Real m_global_old_cpu_time; //!< Temps pr�c�dent CPU utilis� (en seconde)
  Real m_global_cpu_time; //!< Temps CPU utilis� (en seconde)
  Real m_global_old_elapsed_time; //!< Temps pr�c�dent horloge utilis� (en seconde)
  Real m_global_elapsed_time; //!< Temps horloge utilis� (en seconde)
};

#endif  

