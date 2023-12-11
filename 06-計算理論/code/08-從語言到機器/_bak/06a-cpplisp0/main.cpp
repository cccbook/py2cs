#include "lisp.cpp"

int main ()
{
    environment global_env; add_globals(global_env);
    repl("90> ", &global_env);
}

