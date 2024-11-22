// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vadder.h for the primary calling header

#include "Vadder__pch.h"
#include "Vadder__Syms.h"
#include "Vadder___024root.h"

void Vadder___024root___ctor_var_reset(Vadder___024root* vlSelf);

Vadder___024root::Vadder___024root(Vadder__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vadder___024root___ctor_var_reset(this);
}

void Vadder___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vadder___024root::~Vadder___024root() {
}
