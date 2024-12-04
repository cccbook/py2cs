// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vadder__pch.h"

//============================================================
// Constructors

Vadder::Vadder(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vadder__Syms(contextp(), _vcname__, this)}
    , a{vlSymsp->TOP.a}
    , b{vlSymsp->TOP.b}
    , sum{vlSymsp->TOP.sum}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vadder::Vadder(const char* _vcname__)
    : Vadder(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vadder::~Vadder() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vadder___024root___eval_debug_assertions(Vadder___024root* vlSelf);
#endif  // VL_DEBUG
void Vadder___024root___eval_static(Vadder___024root* vlSelf);
void Vadder___024root___eval_initial(Vadder___024root* vlSelf);
void Vadder___024root___eval_settle(Vadder___024root* vlSelf);
void Vadder___024root___eval(Vadder___024root* vlSelf);

void Vadder::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vadder::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vadder___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vadder___024root___eval_static(&(vlSymsp->TOP));
        Vadder___024root___eval_initial(&(vlSymsp->TOP));
        Vadder___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vadder___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vadder::eventsPending() { return false; }

uint64_t Vadder::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vadder::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vadder___024root___eval_final(Vadder___024root* vlSelf);

VL_ATTR_COLD void Vadder::final() {
    Vadder___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vadder::hierName() const { return vlSymsp->name(); }
const char* Vadder::modelName() const { return "Vadder"; }
unsigned Vadder::threads() const { return 1; }
void Vadder::prepareClone() const { contextp()->prepareClone(); }
void Vadder::atClone() const {
    contextp()->threadPoolpOnClone();
}
