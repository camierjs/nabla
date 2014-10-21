/*---------------------------------------------------------------------------*/
/* TraceAccessor.h                                             (C) 2000-2009 */
/*                                                                           */
/* Accès aux traces.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_TRACEACCESSOR_H
#define ARCANE_UTILS_TRACEACCESSOR_H

class ITraceMng;

class TraceAccessor{
 public:
  TraceAccessor(ITraceMng* m);
  virtual ~TraceAccessor();
public:
  ITraceMng* traceMng() const;
  TraceMessage info() const;
  TraceMessage debug() const;
  TraceMessage warning() const;
 private:
  ITraceMng* m_trace;
};

#endif  

