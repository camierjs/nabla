/*---------------------------------------------------------------------------*/
/* ITraceMng.h                                                 (C) 2000-2012 */
/*                                                                           */
/* Gestionnaire des traces.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ITRACEMNG_H
#define ARCANE_UTILS_ITRACEMNG_H

class ITraceMng{
public:
  virtual ~ITraceMng(){}
 public:
  //! Flot pour un message d'erreur
  virtual TraceMessage error() =0;
  //! Flot pour un message d'erreur fatale
  virtual TraceMessage fatal() =0;
  //! Flot pour un message d'avertissement
  virtual TraceMessage warning() =0;
  //! Flot pour un message d'information
  virtual TraceMessage info() =0;
  //! Flot pour un message de debug.
  virtual TraceMessage debug() =0;
  //! Flush tous les flots.
  virtual void flush() =0;
};

#endif
