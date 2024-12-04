
* https://github.com/python/cpython/blob/main/Include/cpython/code.h

```c

// To avoid repeating ourselves in deepfreeze.py, all PyCodeObject members are
// defined in this macro:
#define _PyCode_DEF(SIZE) {                                                    \
    PyObject_VAR_HEAD                                                          \
                                                                               \
    /* Note only the following fields are used in hash and/or comparisons      \
     *                                                                         \
     * - co_name                                                               \
     * - co_argcount                                                           \
     * - co_posonlyargcount                                                    \
     * - co_kwonlyargcount                                                     \
     * - co_nlocals                                                            \
     * - co_stacksize                                                          \
     * - co_flags                                                              \
     * - co_firstlineno                                                        \
     * - co_consts                                                             \
     * - co_names                                                              \
     * - co_localsplusnames                                                    \
     * This is done to preserve the name and line number for tracebacks        \
     * and debuggers; otherwise, constant de-duplication would collapse        \
     * identical functions/lambdas defined on different lines.                 \
     */                                                                        \
                                                                               \
    /* These fields are set with provided values on new code objects. */       \
                                                                               \
    /* The hottest fields (in the eval loop) are grouped here at the top. */   \
    PyObject *co_consts;           /* list (constants used) */                 \
    PyObject *co_names;            /* list of strings (names used) */          \
    PyObject *co_exceptiontable;   /* Byte string encoding exception handling  \
                                      table */                                 \
    int co_flags;                  /* CO_..., see below */                     \
                                                                               \
    /* The rest are not so impactful on performance. */                        \
    int co_argcount;              /* #arguments, except *args */               \
    int co_posonlyargcount;       /* #positional only arguments */             \
    int co_kwonlyargcount;        /* #keyword only arguments */                \
    int co_stacksize;             /* #entries needed for evaluation stack */   \
    int co_firstlineno;           /* first source line number */               \
                                                                               \
    /* redundant values (derived from co_localsplusnames and                   \
       co_localspluskinds) */                                                  \
    int co_nlocalsplus;           /* number of local + cell + free variables */ \
    int co_framesize;             /* Size of frame in words */                 \
    int co_nlocals;               /* number of local variables */              \
    int co_ncellvars;             /* total number of cell variables */         \
    int co_nfreevars;             /* number of free variables */               \
    uint32_t co_version;          /* version number */                         \
                                                                               \
    PyObject *co_localsplusnames; /* tuple mapping offsets to names */         \
    PyObject *co_localspluskinds; /* Bytes mapping to local kinds (one byte    \
                                     per variable) */                          \
    PyObject *co_filename;        /* unicode (where it was loaded from) */     \
    PyObject *co_name;            /* unicode (name, for reference) */          \
    PyObject *co_qualname;        /* unicode (qualname, for reference) */      \
    PyObject *co_linetable;       /* bytes object that holds location info */  \
    PyObject *co_weakreflist;     /* to support weakrefs to code objects */    \
    _PyExecutorArray *co_executors;      /* executors from optimizer */        \
    _PyCoCached *_co_cached;      /* cached co_* attributes */                 \
    uintptr_t _co_instrumentation_version; /* current instrumentation version */ \
    _PyCoMonitoringData *_co_monitoring; /* Monitoring data */                 \
    Py_ssize_t _co_unique_id;     /* ID used for per-thread refcounting */   \
    int _co_firsttraceable;       /* index of first traceable instruction */   \
    /* Scratch space for extra data relating to the code object.               \
       Type is a void* to keep the format private in codeobject.c to force     \
       people to go through the proper APIs. */                                \
    void *co_extra;                                                            \
    char co_code_adaptive[(SIZE)];                                             \
}

/* Bytecode object */
struct PyCodeObject _PyCode_DEF(1);


```