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
    TimerReal //!< Timer utilisant le temps réel
  };

 public:
  
  /*!
   * \brief Sentinelle pour le timer.
   * La sentinelle associée à un timer permet de déclancher celui-ci
   * au moment de sa construction et de l'arrêter au moment de sa
   * destruction. Cela assure que le timer sera bien arrêté en cas
   * d'exception par exemple.
   */
  class Sentry{
   public:
    //! Associe le timer \a t et le démarre
    Sentry(Timer* t) : m_timer(t)
      { m_timer->start(); }
    //! Stoppe le timer associé
    ~Sentry()
      { m_timer->stop(); }
   private:
    Timer* m_timer; //!< Timer associé
  };

  /*!
   * \brief Postionne le nom de l'action en cours d'exécution.
   *
   Le nom d'une action peut-être n'importe quoi. Il est
   juste utilisé pour différencier les différentes partie d'une
   exécution et connaître le temps de chacune d'elle.
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
   * \brief Positionne la phase de l'action en cours d'exécution.
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
   * \brief Affiche le temps passé entre l'appel au constructeur et le destructeur.
   *
   * Cette classe permet de simplement afficher au moment du destructeur,
   * le temps réel écoulé depuis l'appel au constructeur. L'affichage se fait
   * via la méthode info() du ITraceMng.
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
   * Construit un timer lié au sous-domaine \a sd, de nom \a name et de
   * type \a type.
   */
  Timer(ISubDomain* sd,const String& name,eTimerType type);

  /*!
   * \brief Construit un timer.
   *
   * Construit un timer lié au gestionnaire \a tm, de nom \a name et de
   * type \a type.
   */
  Timer(ITimerMng* tm,const String& name,eTimerType type);

  ~Timer(); //!< Libère les ressources

 public:
	
  /*!
   * \brief Active le timer.
   * Si le timer est déjà actif, cette méthode ne fait rien.
   */
  void start();

  /*!
   * \brief Désactive le timer.
   * Si le timer n'est pas actif au moment de l'appel, cette méthode ne
   * fait rien.
   * \return le temps écoulé (en secondes) depuis la dernière activation.
   */
  Real stop();

  //! Retourne l'état d'activation du timer
  bool isActivated() const { return m_is_activated; }

  //! Retourne le nom du timer
  const String& name() const { return m_name; }

  //! Retourne le temps total (en secondes) passé dans le timer
  Real totalTime() const { return m_total_time; }

  //! Retourne le temps (en secondes) passé lors de la dernière activation du timer
  Real lastActivationTime() const { return m_activation_time; }

  //! Retourne le nombre de fois que le timer a été activé
  Integer nbActivated() const { return m_nb_activated; }

  //! Retourne le type du temps utilisé
  eTimerType type() const { return m_type; }

  //! Remet à zéro les compteur de temps
  void reset();

 private:

  ITimerMng* m_timer_mng; //!< Gestionnaire de timer
  eTimerType m_type; //!< Type du timer
  Integer m_nb_activated; //!< Nombre de fois que le timer a été activé
  bool m_is_activated; //!< \a true si le timer est actif
  Real m_activation_time; //!< Temps passé lors de la dernière activation
  Real m_total_time; //!< Temps total passé dans le timer
  String m_name; //!< Nom du timer
};


#endif  

