/*---------------------------------------------------------------------------*/
/* TraceInfo.h                                                 (C) 2000-2010 */
/*                                                                           */
/* Informations de trace.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_TRACEINFO_H
#define ARCANE_UTILS_TRACEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

class TraceInfo
{
 public:
  TraceInfo()
  : m_file("(None)"), m_name("(None)"), m_line(-1) {}
  TraceInfo(const char* file,const char* func_name,int line)
  : m_file(file), m_name(func_name), m_line(line), m_print_signature(true) {}
  TraceInfo(const char* file,const char* func_name,int line,bool print_signature)
  : m_file(file), m_name(func_name), m_line(line), m_print_signature(print_signature) {}
 public:
  const char* file() const { return m_file; }
  int line() const { return m_line; }
  const char* name() const { return m_name; }
  bool printSignature() const { return m_print_signature; }
 private:
  const char* m_file;
  const char* m_name;
  int m_line;
  bool m_print_signature;
};

#endif  

