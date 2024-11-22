// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VADDER__SYMS_H_
#define VERILATED_VADDER__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vadder.h"

// INCLUDE MODULE CLASSES
#include "Vadder___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vadder__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vadder* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vadder___024root               TOP;

    // CONSTRUCTORS
    Vadder__Syms(VerilatedContext* contextp, const char* namep, Vadder* modelp);
    ~Vadder__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
