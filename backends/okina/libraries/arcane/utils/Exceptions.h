#ifndef ARCANE_UTILS_EXCEPTION_H
#define ARCANE_UTILS_EXCEPTION_H

#define A_FUNCINFO __PRETTY_FUNCTION__

class Exception: public std::exception{
 public:
  Exception(const char *name,const char *where){
    std::cerr << "** Exception: Debug mode activated. Execution paused.\n";
  }
  Exception(const char *where){
    std::cerr << "** Exception: Debug mode activated. Execution paused.\n";
  }
};



class ArgumentException: public Exception{
 public:
  ArgumentException(const char* where,const char* message): Exception(where,message){}
};

class FatalErrorException: public Exception{
 public:
  FatalErrorException(const char* where,const char* message): Exception(where,message){}
};

class NotImplementedException: public Exception{
 public:
  NotImplementedException(const char* where,const char* message): Exception(where,message){}
  NotImplementedException(const char* where): Exception(where){}
};


#endif  

