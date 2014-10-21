/*---------------------------------------------------------------------------*/
/* Parallel.h                                                  (C) 2000-2005 */
/*                                                                           */
/* Espace de nom des types gérant le parallélisme.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_H
#define ARCANE_PARALLEL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Parallel
{
  /*!
   * \brief Types des réductions supportées.
   */
  enum eReduceType
  {
    ReduceMin, //!< Minimum des valeurs
    ReduceMax, //!< Maximum des valeurs
    ReduceSum  //!< Somme des valeurs
  };

  /*!
   * \brief Type d'attente.
   */
  enum eWaitType
  {
    WaitAll, //! Attend que tous les messages de la liste soient traités
    WaitSome,//! Attend que au moins un message de la liste soit traité
    WaitSomeNonBlocking //! Traite uniquement les messages qui peuvent l'être sans attendre.
  };

  /*!
   * \internal
   * \brief Informations de retour d'un message.
   * Ces informations sont utilisées pour les messages non bloquants.
   */
  class Request {
    union _Request
    {
      int i;
      long l;
      void* v;
      const void* cv;
    };

    enum Type {
      T_Int,
      T_Long,
      T_Ptr,
      T_Null
    };

   public:

    Request()
    : m_return_value(0)
    {
      m_type = T_Null;
      m_request = null_request;
    }

    Request(int return_value,void* request)
    : m_return_value(return_value)
    {
      m_type = T_Ptr;
      m_request.v = request;
    }

    Request(int return_value,const void* request)
    : m_return_value(return_value)
    {
      m_type = T_Ptr;
      m_request.cv = request;
    }

    Request(int return_value,int request)
    : m_return_value(return_value)
    {
      m_type = T_Int;
      m_request.i = request;
    }

    Request(int return_value,long request)
    : m_return_value(return_value)
    {
      m_type = T_Long;
      m_request.l = request;
    }

    Request(const Request& rhs)
    : m_return_value(rhs.m_return_value), m_type(rhs.m_type)
    {
      m_request.cv = rhs.m_request.cv;
    }
   public:

    template<typename T>
    operator const T*() const { return (const T*)m_request.cv; }
    template<typename T>
    operator T*() const { return (T*)m_request.v; }
    operator int() const { return m_request.i; }
    operator long() const { return m_request.l; }

   public:

    int returnValue() const { return m_return_value; }
    bool isValid() const
    {
      if (m_type==T_Null)
        return false;
      if (m_type==T_Int)
        return m_request.i!=null_request.i;
      if (m_type==T_Long)
        return m_request.l!=null_request.l;
      return m_request.cv!=null_request.cv;
    }
    void* requestAsVoidPtr() const { return m_request.v; }

    static void setNullRequest(Request r) { null_request = r.m_request; }

    void reset() {
    	m_request = null_request;
    }

    void print(ostream& o) const;

   private:

    int m_return_value;
    int m_type;
    _Request m_request;

    static _Request null_request;
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ostream& operator<<(ostream& o,const Parallel::Request prequest)
{
  prequest.print(o);
  return o;
}

#endif  

