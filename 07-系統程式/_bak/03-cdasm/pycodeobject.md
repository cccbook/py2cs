
https://github.com/python/cpython/blob/main/Include/object.h


* [為你自己學 PYTHON](https://pythonbook.cc/chapters/basic/introduction)
    * [來讀 CPython 原始碼](https://pythonbook.cc/chapters/cpython)
    * [參觀 Bytecode 工廠](https://pythonbook.cc/chapters/cpython/from-py-to-pyc)
    * [虛擬機器五部曲（一）](https://pythonbook.cc/chapters/cpython/pvm-code-objects)

* [深度剖析CPython解释器](https://www.cnblogs.com/traditional/tag/%E6%B7%B1%E5%BA%A6%E5%89%96%E6%9E%90CPython%E8%A7%A3%E9%87%8A%E5%99%A8/)
    * [《深度剖析CPython解释器》10. Python中的PyCodeObject对象与pyc文件 - 古明地盆 - 博客园](https://www.cnblogs.com/traditional/p/13507329.html)


```c
typedef struct {
    PyObject_HEAD		/* 头部信息, 我们看到真的一切皆对象, 字节码也是个对象 */	
    int co_argcount;            /* 可以通过位置参数传递的参数个数 */
    int co_posonlyargcount;     /* 只能通过位置参数传递的参数个数,  Python3.8新增 */
    int co_kwonlyargcount;      /* 只能通过关键字参数传递的参数个数 */
    int co_nlocals;             /* 代码块中局部变量的个数，也包括参数 */
    int co_stacksize;           /* 执行该段代码块需要的栈空间 */
    int co_flags;               /* 参数类型标识 */
    int co_firstlineno;         /* 代码块在对应文件的行号 */
    PyObject *co_code;          /* 指令集, 也就是字节码, 它是一个bytes对象 */
    PyObject *co_consts;        /* 常量池, 一个元组，保存代码块中的所有常量。 */
    PyObject *co_names;         /* 一个元组,保存代码块中引用的其它作用域的变量 */
    PyObject *co_varnames;      /* 一个元组,保存当前作用域中的变量 */
    PyObject *co_freevars;      /* 内层函数引用的外层函数的作用域中的变量 */
    PyObject *co_cellvars;      /* 外层函数中作用域中被内层函数引用的变量，本质上和co_freevars是一样的 */

    Py_ssize_t *co_cell2arg;    /* 无需关注 */
    PyObject *co_filename;      /* 代码块所在的文件名 */
    PyObject *co_name;          /* 代码块的名字，通常是函数名或者类名 */
    PyObject *co_lnotab;        /* 字节码指令与python源代码的行号之间的对应关系，以PyByteObject的形式存在 */
    
    //剩下的无需关注了
    void *co_zombieframe;       /* for optimization only (see frameobject.c) */
    PyObject *co_weakreflist;   /* to support weakrefs to code objects */
    void *co_extra;
    unsigned char *co_opcache_map;
    _PyOpcache *co_opcache;
    int co_opcache_flag; 
    unsigned char co_opcache_size; 
} PyCodeObject;


//位置：Python/marshal.c

//FILE是一个文件句柄，可以把WFILE看成是FILE的包装
typedef struct {
    FILE *fp;  //文件句柄
    //下面的字段在写入信息的时候会看到
    int error;  
    int depth;
    PyObject *str;
    char *ptr;
    char *end;
    char *buf;
    _Py_hashtable_t *hashtable;
    int version;
} WFILE;

```

PyCodeObject里面的co_code指向了这个字节码

```c
//源代码很长, 具体逻辑就不贴了
//我们后面会单独截取一部分进行分析

static void
w_complex_object(PyObject *v, char flag, WFILE *p)
{
    Py_ssize_t i, n;

    if (PyLong_CheckExact(v)) {
        //......
    }
    else if (PyFloat_CheckExact(v)) {
        if (p->version > 1) {
            //......
        }
        else {
            //......
        }
    }
    else if (PyComplex_CheckExact(v)) {
        if (p->version > 1) {
            //......
        }
        else {
            //......
        }
    }
    else if (PyBytes_CheckExact(v)) {
        //......
    }
    else if (PyUnicode_CheckExact(v)) {
        if (p->version >= 4 && PyUnicode_IS_ASCII(v)) {
            	//......
            }
            else {
                //......
            }
        }
        else {
            //......
        }
    }
    else if (PyTuple_CheckExact(v)) {
       //......
    }
    else if (PyList_CheckExact(v)) {
        //......
    }
    else if (PyDict_CheckExact(v)) {
        //......
    }
    else if (PyAnySet_CheckExact(v)) {
        //......
    }
    else if (PyCode_Check(v)) {
        PyCodeObject *co = (PyCodeObject *)v;
        W_TYPE(TYPE_CODE, p);
        w_long(co->co_argcount, p);
        w_long(co->co_kwonlyargcount, p);
        w_long(co->co_nlocals, p);
        w_long(co->co_stacksize, p);
        w_long(co->co_flags, p);
        w_object(co->co_code, p);
        w_object(co->co_consts, p);
        w_object(co->co_names, p);
        w_object(co->co_varnames, p);
        w_object(co->co_freevars, p);
        w_object(co->co_cellvars, p);
        w_object(co->co_filename, p);
        w_object(co->co_name, p);
        w_long(co->co_firstlineno, p);
        w_object(co->co_lnotab, p);
    }
    else if (PyObject_CheckBuffer(v)) {
        //......
    }
    else {
        W_TYPE(TYPE_UNKNOWN, p);
        p->error = WFERR_UNMARSHALLABLE;
    }
}

PyCodeObject *co = (PyCodeObject *)v;
        W_TYPE(TYPE_CODE, p);
        w_long(co->co_argcount, p);
        w_long(co->co_kwonlyargcount, p);
        w_long(co->co_nlocals, p);
        w_long(co->co_stacksize, p);
        w_long(co->co_flags, p);
        w_object(co->co_code, p);
        w_object(co->co_consts, p);
        w_object(co->co_names, p);
        w_object(co->co_varnames, p);
        w_object(co->co_freevars, p);
        w_object(co->co_cellvars, p);
        w_object(co->co_filename, p);
        w_object(co->co_name, p);
        w_long(co->co_firstlineno, p);
        w_object(co->co_lnotab, p);

```

不管什么对象，最后都为归结为两种简单的形式，一种是数值写入，一种是字符串写入

```c
typedef struct _frame {
    PyObject_VAR_HEAD  		/* 可变对象的头部信息 */
    struct _frame *f_back;      /* 上一级栈帧, 也就是调用者的栈帧 */
    PyCodeObject *f_code;       /* PyCodeObject对象, 通过栈帧对象的f_code可以获取对应的PyCodeObject对象 */
    PyObject *f_builtins;       /* builtin命名空间，一个PyDictObject对象 */
    PyObject *f_globals;        /* global命名空间，一个PyDictObject对象 */
    PyObject *f_locals;         /* local命名空间，一个PyDictObject对象  */
    PyObject **f_valuestack;    /* 运行时的栈底位置 */

    PyObject **f_stacktop;      /* 运行时的栈顶位置 */
    PyObject *f_trace;          /* 回溯函数，打印异常栈 */
    char f_trace_lines;         /* 是否触发每一行的回溯事件 */
    char f_trace_opcodes;       /* 是否触发每一个操作码的回溯事件 */

    PyObject *f_gen;            /* 是否是生成器 */

    int f_lasti;                /* 上一条指令在f_code中的偏移量 */

    int f_lineno;               /* 当前字节码对应的源代码行 */
    int f_iblock;               /* 当前指令在栈f_blockstack中的索引 */
    char f_executing;           /* 当前栈帧是否仍在执行 */
    PyTryBlock f_blockstack[CO_MAXBLOCKS]; /* 用于try和loop代码块 */
    PyObject *f_localsplus[1];  /* 动态内存，维护局部变量+cell对象集合+free对象集合+运行时栈所需要的空间 */
} PyFrameObject;

```

不少初学者对 Python 存在误解，以为它是类似 Shell 的解释性脚本语言，其实并不是。虽然执行 Python 程序的 称为 Python 解释器，但它其实包含一个 "编译器" 和一个 "虚拟机"。

Python 程序执行时需要先由 编译器 编译成 PyCodeObject 对象，然后再交由 虚拟机 来执行。不管程序执行多少次，只要源码没有变化，编译后得到的 PyCodeObject 对象就肯定是一样的。因此，Python 将 PyCodeObject 对象序列化并保存到 pyc 文件中。当程序再次执行时，Python 直接从 pyc 文件中加载代码对象，省去编译环节。当然了，当 py 源码文件改动后，pyc 文件便失效了，这时 Python 必须重新编译 py 文件。


```c
static PyObject *
cmp_outcome(int op, PyObject *v, PyObject *w)
{	
    //我们说Python中的变量在C的层面上是一个指针, 因此Python中两个变量是否指向同一个对象 等价于 在C中两个指针是否相等
    //而Python中的==, 则需要调用PyObject_RichCompare(指针1, 指针2, 操作符)来看它们指向的对象所维护的值是否相等
    int res = 0;
    switch (op) {
    case PyCmp_IS:
        //is操作符的话, 在C的层面直接一个==判断即可
        res = (v == w);
        break;
    // ...
    default:
        //而PyObject_RichCompare是一个函数调用, 将进一步调用对象的魔法方法进行判断。
        return PyObject_RichCompare(v, w, op);
    }
    v = res ? Py_True : Py_False;
    Py_INCREF(v);
    return v;
}

```