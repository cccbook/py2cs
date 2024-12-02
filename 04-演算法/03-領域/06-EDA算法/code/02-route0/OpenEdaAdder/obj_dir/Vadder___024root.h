// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vadder.h for the primary calling header

#ifndef VERILATED_VADDER___024ROOT_H_
#define VERILATED_VADDER___024ROOT_H_  // guard

#include "verilated.h"


class Vadder__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vadder___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(a,3,0);
    VL_IN8(b,3,0);
    VL_OUT8(sum,4,0);
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __VactContinue;
    IData/*31:0*/ __VactIterCount;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<0> __VactTriggered;
    VlTriggerVec<0> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vadder__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vadder___024root(Vadder__Syms* symsp, const char* v__name);
    ~Vadder___024root();
    VL_UNCOPYABLE(Vadder___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
