#ifndef ARCANE_TIMER_H
#define ARCANE_TIMER_H


#include "arcane/ArcaneTypes.h"
#include "arcane/utils/ArcaneGlobal.h"


class ITimerMng;
//class String;
class ITimeStats;
class ISubDomain;

class Timer{
 public:

  //! Type du timer
  enum eTimerType  {
    TimerVirtual, //!< Timer utilisant le temps CPU
    TimerReal //!< Timer utilisant le temps r�el
  };

 public:
  
  /*!
   * \brief Sentinelle pour le timer.
   * La sentinelle associ�e � un timer permet de d�clancher celui-ci
   * au moment de sa construction et de l'arr�ter au moment de sa
   * destruction. Cela assure que le timer sera bien arr�t� en cas
   * d'exception par exemple.
   */
  class Sentry{
   public:
    //! Associe le timer \a t et le d�marre
    Sentry(Timer* t) : m_timer(t)
      { m_timer->start(); }
    //! Stoppe le timer associ�
    ~Sentry()
      { m_timer->stop(); }
   private:
    Timer* m_timer; //!< Timer associ�
  };

  /*!
   * \brief Postionne le nom de l'action en cours d'ex�cution.
   *
   Le nom d'une action peut-�tre n'importe quoi. Il est
   juste utilis� pour diff�rencier les diff�rentes partie d'une
   ex�cution et conna�tre le temps de chacune d'elle.
   Les actions doivent s'imbriquent les unes dans les autres
  */
  class Action  {
   public:
    Action(ISubDomain* sub_domain,const String& action_name,bool print_time=false);
    Action(ITimeStats* stats,const String& action_name,bool print_time=false);
    ~Action();
   public:
   private:
    ITimeStats* m_stats;
    String m_action_name;
    bool m_print_time;
   private:
    void _init();
  };

  /*!
   * \brief Positionne la phase de l'action en cours d'ex�cution.
   */
  class Phase  {
   public:
   public:
    Phase(ISubDomain* sub_domain,eTimePhase pt);
    Phase(ITimeStats* stats,eTimePhase pt);
    ~Phase();
   public:
   private:
    ITimeStats* m_stats; //!< Gestionnaire de sous-domaine
    eTimePhase m_phase_type;
   private:
    void _init();
  };

  /*!
   * \brief Affiche le temps pass� entre l'appel au constructeur et le destructeur.
   *
   * Cette classe permet de simplement afficher au moment du destructeur,
   * le temps r�el �coul� depuis l'appel au constructeur. L'affichage se fait
   * via la m�thode info() du ITraceMng.
   \code
   * {
   *   Timer::SimplePrinter sp(traceMng(),"myFunction");
   *   myFunction();
   * }
   */
  class SimplePrinter
  {
   public:
    SimplePrinter(ITraceMng* tm,const String& msg);
    SimplePrinter(ITraceMng* tm,const String& msg,bool is_active);
    ~SimplePrinter();
   private:
    ITraceMng* m_trace_mng;
    Real m_begin_time;
    bool m_is_active;
    String m_message;
   private:
    void _init();
  };

 public:

  /*!
   * \brief Construit un timer.
   * Construit un timer li� au sous-domaine \a sd, de nom \a name et de
   * type \a type.
   */
  Timer(ISubDomain* sd,const String& name,eTimerType type);

  /*!
   * \brief Construit un timer.
   *
   * Construit un timer li� au gestionnaire \a tm, de nom \a name et de
   * type \a type.
   */
  Timer(ITimerMng* tm,const String& name,eTimerType type);

  ~Timer(); //!< Lib�re les ressources

 public:
	
  /*!
   * \brief Active le timer.
   * Si le timer est d�j� actif, cette m�thode ne fait rien.
   */
  void start();

  /*!
   * \brief D�sactive le timer.
   * Si le timer n'est pas actif au moment de l'appel, cette m�thode ne
   * fait rien.
   * \return le temps �coul� (en secondes) depuis la derni�re activation.
   */
  Real stop();

  //! Retourne l'�tat d'activation du timer
  bool isActivated() const { return m_is_activated; }

  //! Retourne le nom du timer
  const String& name() const { return m_name; }

  //! Retourne le temps total (en secondes) pass� dans le timer
  Real totalTime() const { return m_total_time; }

  //! Retourne le temps (en secondes) pass� lors de la derni�re activation du timer
  Real lastActivationTime() const { return m_activation_time; }

  //! Retourne le nombre de fois que le timer a �t� activ�
  Integer nbActivated() const { return m_nb_activated; }

  //! Retourne le type du temps utilis�
  eTimerType type() const { return m_type; }

  //! Remet � z�ro les compteur de temps
  void reset();

 private:

  ITimerMng* m_timer_mng; //!< Gestionnaire de timer
  eTimerType m_type; //!< Type du timer
  Integer m_nb_activated; //!< Nombre de fois que le timer a �t� activ�
  bool m_is_activated; //!< \a true si le timer est actif
  Real m_activation_time; //!< Temps pass� lors de la derni�re activation
  Real m_total_time; //!< Temps total pass� dans le timer
  String m_name; //!< Nom du timer
};


#endif  

