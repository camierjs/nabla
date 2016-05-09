
extern "C"
{
  typedef struct
  {
    __off_t __pos;
    __mbstate_t __state;
  } _G_fpos_t;
  typedef struct
  {
    __off64_t __pos;
    __mbstate_t __state;
  } _G_fpos64_t;
  struct _IO_jump_t;
  struct _IO_FILE;
  typedef void _IO_lock_t;
  struct _IO_marker
  {
    struct _IO_marker *_next;
    struct _IO_FILE *_sbuf;
    int _pos;
  };
  enum __codecvt_result
  {
    __codecvt_ok,
    __codecvt_partial,
    __codecvt_error,
    __codecvt_noconv
  };
  struct _IO_FILE
  {
    int _flags;
    char *_IO_read_ptr;
    char *_IO_read_end;
    char *_IO_read_base;
    char *_IO_write_base;
    char *_IO_write_ptr;
    char *_IO_write_end;
    char *_IO_buf_base;
    char *_IO_buf_end;
    char *_IO_save_base;
    char *_IO_backup_base;
    char *_IO_save_end;
    struct _IO_marker *_markers;
    struct _IO_FILE *_chain;
    int _fileno;
    int _flags2;
    __off_t _old_offset;
    unsigned short _cur_column;
    signed char _vtable_offset;
    char _shortbuf[1];
    _IO_lock_t *_lock;
    __off64_t _offset;
    void *__pad1;
    void *__pad2;
    void *__pad3;
    void *__pad4;
    size_t __pad5;
    int _mode;
    char _unused2[15 * sizeof (int) - 4 * sizeof (void *) - sizeof (size_t)];
  };
  struct _IO_FILE_plus;
  extern struct _IO_FILE_plus _IO_2_1_stdin_;
  extern struct _IO_FILE_plus _IO_2_1_stdout_;
  extern struct _IO_FILE_plus _IO_2_1_stderr_;
  typedef __ssize_t __io_read_fn (void *__cookie, char *__buf,
				  size_t __nbytes);
  typedef __ssize_t __io_write_fn (void *__cookie, const char *__buf,
				   size_t __n);
  typedef int __io_seek_fn (void *__cookie, __off64_t * __pos, int __w);
  typedef int __io_close_fn (void *__cookie);
  typedef __io_read_fn cookie_read_function_t;
  typedef __io_write_fn cookie_write_function_t;
  typedef __io_seek_fn cookie_seek_function_t;
  typedef __io_close_fn cookie_close_function_t;
  typedef struct
  {
    __io_read_fn *read;
    __io_write_fn *write;
    __io_seek_fn *seek;
    __io_close_fn *close;
  } _IO_cookie_io_functions_t;
  typedef _IO_cookie_io_functions_t cookie_io_functions_t;
  struct _IO_cookie_file;
  extern void _IO_cookie_init (struct _IO_cookie_file *__cfile,
			       int __read_write, void *__cookie,
			       _IO_cookie_io_functions_t __fns);
  extern "C"
  {
    extern int __underflow (_IO_FILE *);
    extern int __uflow (_IO_FILE *);
    extern int __overflow (_IO_FILE *, int);
    extern int _IO_getc (_IO_FILE * __fp);
    extern int _IO_putc (int __c, _IO_FILE * __fp);
    extern int _IO_feof (_IO_FILE * __fp) throw ();
    extern int _IO_ferror (_IO_FILE * __fp) throw ();
    extern int _IO_peekc_locked (_IO_FILE * __fp);
    extern void _IO_flockfile (_IO_FILE *) throw ();
    extern void _IO_funlockfile (_IO_FILE *) throw ();
    extern int _IO_ftrylockfile (_IO_FILE *) throw ();
    extern int _IO_vfscanf (_IO_FILE * __restrict, const char *__restrict,
			    __gnuc_va_list, int *__restrict);
    extern int _IO_vfprintf (_IO_FILE * __restrict, const char *__restrict,
			     __gnuc_va_list);
    extern __ssize_t _IO_padn (_IO_FILE *, int, __ssize_t);
    extern size_t _IO_sgetn (_IO_FILE *, void *, size_t);
    extern __off64_t _IO_seekoff (_IO_FILE *, __off64_t, int, int);
    extern __off64_t _IO_seekpos (_IO_FILE *, __off64_t, int);
    extern void _IO_free_backup_area (_IO_FILE *) throw ();
  }
  typedef __gnuc_va_list va_list;

  typedef _G_fpos_t fpos_t;

  typedef _G_fpos64_t fpos64_t;
  extern struct _IO_FILE *stdin;
  extern struct _IO_FILE *stdout;
  extern struct _IO_FILE *stderr;

  extern int remove (const char *__filename) throw ();
  extern int rename (const char *__old, const char *__new) throw ();

  extern int renameat (int __oldfd, const char *__old, int __newfd,
		       const char *__new) throw ();

  extern FILE *tmpfile (void);
  extern FILE *tmpfile64 (void);
  extern char *tmpnam (char *__s) throw ();

  extern char *tmpnam_r (char *__s) throw ();
  extern char *tempnam (const char *__dir, const char *__pfx)
    throw () __attribute__ ((__malloc__));

  extern int fclose (FILE * __stream);
  extern int fflush (FILE * __stream);

  extern int fflush_unlocked (FILE * __stream);
  extern int fcloseall (void);

  extern FILE *fopen (const char *__restrict __filename,
		      const char *__restrict __modes);
  extern FILE *freopen (const char *__restrict __filename,
			const char *__restrict __modes,
			FILE * __restrict __stream);

  extern FILE *fopen64 (const char *__restrict __filename,
			const char *__restrict __modes);
  extern FILE *freopen64 (const char *__restrict __filename,
			  const char *__restrict __modes,
			  FILE * __restrict __stream);
  extern FILE *fdopen (int __fd, const char *__modes) throw ();
  extern FILE *fopencookie (void *__restrict __magic_cookie,
			    const char *__restrict __modes,
			    _IO_cookie_io_functions_t __io_funcs) throw ();
  extern FILE *fmemopen (void *__s, size_t __len, const char *__modes)
    throw ();
  extern FILE *open_memstream (char **__bufloc, size_t * __sizeloc) throw ();

  extern void setbuf (FILE * __restrict __stream,
		      char *__restrict __buf) throw ();
  extern int setvbuf (FILE * __restrict __stream, char *__restrict __buf,
		      int __modes, size_t __n) throw ();

  extern void setbuffer (FILE * __restrict __stream, char *__restrict __buf,
			 size_t __size) throw ();
  extern void setlinebuf (FILE * __stream) throw ();

  extern int fprintf (FILE * __restrict __stream,
		      const char *__restrict __format, ...);
  extern int printf (const char *__restrict __format, ...);
  extern int sprintf (char *__restrict __s,
		      const char *__restrict __format, ...) throw ();
  extern int vfprintf (FILE * __restrict __s, const char *__restrict __format,
		       __gnuc_va_list __arg);
  extern int vprintf (const char *__restrict __format, __gnuc_va_list __arg);
  extern int vsprintf (char *__restrict __s, const char *__restrict __format,
		       __gnuc_va_list __arg) throw ();


  extern int snprintf (char *__restrict __s, size_t __maxlen,
		       const char *__restrict __format, ...)
    throw () __attribute__ ((__format__ (__printf__, 3, 4)));
  extern int vsnprintf (char *__restrict __s, size_t __maxlen,
			const char *__restrict __format, __gnuc_va_list __arg)
    throw () __attribute__ ((__format__ (__printf__, 3, 0)));

  extern int vasprintf (char **__restrict __ptr, const char *__restrict __f,
			__gnuc_va_list __arg)
    throw () __attribute__ ((__format__ (__printf__, 2, 0)));
  extern int __asprintf (char **__restrict __ptr,
			 const char *__restrict __fmt, ...)
    throw () __attribute__ ((__format__ (__printf__, 2, 3)));
  extern int asprintf (char **__restrict __ptr,
		       const char *__restrict __fmt, ...)
    throw () __attribute__ ((__format__ (__printf__, 2, 3)));
  extern int vdprintf (int __fd, const char *__restrict __fmt,
		       __gnuc_va_list __arg)
    __attribute__ ((__format__ (__printf__, 2, 0)));
  extern int dprintf (int __fd, const char *__restrict __fmt, ...)
    __attribute__ ((__format__ (__printf__, 2, 3)));

  extern int fscanf (FILE * __restrict __stream,
		     const char *__restrict __format, ...);
  extern int scanf (const char *__restrict __format, ...);
  extern int sscanf (const char *__restrict __s,
		     const char *__restrict __format, ...) throw ();


  extern int vfscanf (FILE * __restrict __s, const char *__restrict __format,
		      __gnuc_va_list __arg)
    __attribute__ ((__format__ (__scanf__, 2, 0)));
  extern int vscanf (const char *__restrict __format, __gnuc_va_list __arg)
    __attribute__ ((__format__ (__scanf__, 1, 0)));
  extern int vsscanf (const char *__restrict __s,
		      const char *__restrict __format, __gnuc_va_list __arg)
    throw () __attribute__ ((__format__ (__scanf__, 2, 0)));


  extern int fgetc (FILE * __stream);
  extern int getc (FILE * __stream);
  extern int getchar (void);

  extern int getc_unlocked (FILE * __stream);
  extern int getchar_unlocked (void);
  extern int fgetc_unlocked (FILE * __stream);

  extern int fputc (int __c, FILE * __stream);
  extern int putc (int __c, FILE * __stream);
  extern int putchar (int __c);

  extern int fputc_unlocked (int __c, FILE * __stream);
  extern int putc_unlocked (int __c, FILE * __stream);
  extern int putchar_unlocked (int __c);
  extern int getw (FILE * __stream);
  extern int putw (int __w, FILE * __stream);

  extern char *fgets (char *__restrict __s, int __n,
		      FILE * __restrict __stream);
  extern char *gets (char *__s) __attribute__ ((__deprecated__));

  extern char *fgets_unlocked (char *__restrict __s, int __n,
			       FILE * __restrict __stream);
  extern __ssize_t __getdelim (char **__restrict __lineptr,
			       size_t * __restrict __n, int __delimiter,
			       FILE * __restrict __stream);
  extern __ssize_t getdelim (char **__restrict __lineptr,
			     size_t * __restrict __n, int __delimiter,
			     FILE * __restrict __stream);
  extern __ssize_t getline (char **__restrict __lineptr,
			    size_t * __restrict __n,
			    FILE * __restrict __stream);

  extern int fputs (const char *__restrict __s, FILE * __restrict __stream);
  extern int puts (const char *__s);
  extern int ungetc (int __c, FILE * __stream);
  extern size_t fread (void *__restrict __ptr, size_t __size,
		       size_t __n, FILE * __restrict __stream);
  extern size_t fwrite (const void *__restrict __ptr, size_t __size,
			size_t __n, FILE * __restrict __s);

  extern int fputs_unlocked (const char *__restrict __s,
			     FILE * __restrict __stream);
  extern size_t fread_unlocked (void *__restrict __ptr, size_t __size,
				size_t __n, FILE * __restrict __stream);
  extern size_t fwrite_unlocked (const void *__restrict __ptr, size_t __size,
				 size_t __n, FILE * __restrict __stream);

  extern int fseek (FILE * __stream, long int __off, int __whence);
  extern long int ftell (FILE * __stream);
  extern void rewind (FILE * __stream);

  extern int fseeko (FILE * __stream, __off_t __off, int __whence);
  extern __off_t ftello (FILE * __stream);

  extern int fgetpos (FILE * __restrict __stream, fpos_t * __restrict __pos);
  extern int fsetpos (FILE * __stream, const fpos_t * __pos);

  extern int fseeko64 (FILE * __stream, __off64_t __off, int __whence);
  extern __off64_t ftello64 (FILE * __stream);
  extern int fgetpos64 (FILE * __restrict __stream,
			fpos64_t * __restrict __pos);
  extern int fsetpos64 (FILE * __stream, const fpos64_t * __pos);

  extern void clearerr (FILE * __stream) throw ();
  extern int feof (FILE * __stream) throw ();
  extern int ferror (FILE * __stream) throw ();

  extern void clearerr_unlocked (FILE * __stream) throw ();
  extern int feof_unlocked (FILE * __stream) throw ();
  extern int ferror_unlocked (FILE * __stream) throw ();

  extern void perror (const char *__s);

  extern int sys_nerr;
  extern const char *const sys_errlist[];
  extern int _sys_nerr;
  extern const char *const _sys_errlist[];
  extern int fileno (FILE * __stream) throw ();
  extern int fileno_unlocked (FILE * __stream) throw ();
  extern FILE *popen (const char *__command, const char *__modes);
  extern int pclose (FILE * __stream);
  extern char *ctermid (char *__s) throw ();
  extern char *cuserid (char *__s);
  struct obstack;
  extern int obstack_printf (struct obstack *__restrict __obstack,
			     const char *__restrict __format, ...)
    throw () __attribute__ ((__format__ (__printf__, 2, 3)));
  extern int obstack_vprintf (struct obstack *__restrict __obstack,
			      const char *__restrict __format,
			      __gnuc_va_list __args)
    throw () __attribute__ ((__format__ (__printf__, 2, 0)));
  extern void flockfile (FILE * __stream) throw ();
  extern int ftrylockfile (FILE * __stream) throw ();
  extern void funlockfile (FILE * __stream) throw ();
  extern __inline __attribute__ ((__gnu_inline__)) int
    vprintf (const char *__restrict __fmt, __gnuc_va_list __arg)
  {
    return vfprintf (stdout, __fmt, __arg);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int getchar (void)
  {
    return _IO_getc (stdin);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    fgetc_unlocked (FILE * __fp)
  {
    return (__builtin_expect
	    (((__fp)->_IO_read_ptr >= (__fp)->_IO_read_end),
	     0) ? __uflow (__fp) : *(unsigned char *) (__fp)->_IO_read_ptr++);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    getc_unlocked (FILE * __fp)
  {
    return (__builtin_expect
	    (((__fp)->_IO_read_ptr >= (__fp)->_IO_read_end),
	     0) ? __uflow (__fp) : *(unsigned char *) (__fp)->_IO_read_ptr++);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int getchar_unlocked (void)
  {
    return (__builtin_expect
	    (((stdin)->_IO_read_ptr >= (stdin)->_IO_read_end),
	     0) ? __uflow (stdin) : *(unsigned char *) (stdin)->
	    _IO_read_ptr++);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int putchar (int __c)
  {
    return _IO_putc (__c, stdout);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    fputc_unlocked (int __c, FILE * __stream)
  {
    return (__builtin_expect
	    (((__stream)->_IO_write_ptr >= (__stream)->_IO_write_end),
	     0) ? __overflow (__stream,
			      (unsigned char) (__c)) : (unsigned
							char) (*(__stream)->
							       _IO_write_ptr++
							       = (__c)));
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    putc_unlocked (int __c, FILE * __stream)
  {
    return (__builtin_expect
	    (((__stream)->_IO_write_ptr >= (__stream)->_IO_write_end),
	     0) ? __overflow (__stream,
			      (unsigned char) (__c)) : (unsigned
							char) (*(__stream)->
							       _IO_write_ptr++
							       = (__c)));
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    putchar_unlocked (int __c)
  {
    return (__builtin_expect
	    (((stdout)->_IO_write_ptr >= (stdout)->_IO_write_end),
	     0) ? __overflow (stdout,
			      (unsigned char) (__c)) : (unsigned
							char) (*(stdout)->
							       _IO_write_ptr++
							       = (__c)));
  }
  extern __inline __attribute__ ((__gnu_inline__)) __ssize_t
    getline (char **__lineptr, size_t * __n, FILE * __stream)
  {
    return __getdelim (__lineptr, __n, '\n', __stream);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    __attribute__ ((__leaf__)) feof_unlocked (FILE * __stream) throw ()
  {
    return (((__stream)->_flags & 0x10) != 0);
  }
  extern __inline __attribute__ ((__gnu_inline__)) int
    __attribute__ ((__leaf__)) ferror_unlocked (FILE * __stream) throw ()
  {
    return (((__stream)->_flags & 0x20) != 0);
  }
}
