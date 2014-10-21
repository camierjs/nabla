/*---------------------------------------------------------------------------*/
/* TraceMessage.h                                              (C) 2000-2013 */
/*                                                                           */
/* Message de trace.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_TRACEMESSAGE_H
#define ARCANE_UTILS_TRACEMESSAGE_H

class ITraceMng;

class TraceMessage{
public:
TraceMessage(ostream *ostr, ITraceMng*m):
  m_stream(ostr), m_parent(m){}
  ~TraceMessage(){}
 public:
  ostream* m_stream; //!< Flot sur lequel le message est envoyé
  ITraceMng* m_parent; //!< Gestionnaire de message parent
};


template<class T> inline const TraceMessage& operator<<(const TraceMessage& o,const T& v){
  *o.m_stream << v;
  return o;
}

#endif

