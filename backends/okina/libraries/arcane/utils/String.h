#ifndef _OKINA_ARCANE_STRING_
#define _OKINA_ARCANE_STRING_

class String:public std::string{
public:
  String(const char *cstr){}
public:
  //char *localstr() { return this->data(); }
  const char *localstr() const { return this->c_str(); }
};

#endif
