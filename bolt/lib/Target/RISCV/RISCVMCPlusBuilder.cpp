//===- bolt/Target/RISCV/RISCVMCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace {

class RISCVMCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  bool equals(const MCTargetExpr &A, const MCTargetExpr &B,
              CompFuncTy Comp) const override {
    const auto &RISCVExprA = cast<RISCVMCExpr>(A);
    const auto &RISCVExprB = cast<RISCVMCExpr>(B);
    if (RISCVExprA.getSpecifier() != RISCVExprB.getSpecifier())
      return false;

    return MCPlusBuilder::equals(*RISCVExprA.getSubExpr(),
                                 *RISCVExprB.getSubExpr(), Comp);
  }

  void getCalleeSavedRegs(BitVector &Regs) const override {
    Regs |= getAliases(RISCV::X2);
    Regs |= getAliases(RISCV::X8);
    Regs |= getAliases(RISCV::X9);
    Regs |= getAliases(RISCV::X18);
    Regs |= getAliases(RISCV::X19);
    Regs |= getAliases(RISCV::X20);
    Regs |= getAliases(RISCV::X21);
    Regs |= getAliases(RISCV::X22);
    Regs |= getAliases(RISCV::X23);
    Regs |= getAliases(RISCV::X24);
    Regs |= getAliases(RISCV::X25);
    Regs |= getAliases(RISCV::X26);
    Regs |= getAliases(RISCV::X27);
  }

  bool shouldRecordCodeRelocation(uint32_t RelType) const override {
    switch (RelType) {
    case ELF::R_RISCV_JAL:
    case ELF::R_RISCV_CALL:
    case ELF::R_RISCV_CALL_PLT:
    case ELF::R_RISCV_BRANCH:
    case ELF::R_RISCV_RVC_BRANCH:
    case ELF::R_RISCV_RVC_JUMP:
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_PCREL_HI20:
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
    case ELF::R_RISCV_HI20:
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
    case ELF::R_RISCV_TLS_GOT_HI20:
      return true;
    default:
      llvm_unreachable("Unexpected RISCV relocation type in code");
    }
  }

  bool isNop(const MCInst &Inst) const {
    return Inst.getOpcode() == RISCV::ADDI &&
           Inst.getOperand(0).getReg() == RISCV::X0 &&
           Inst.getOperand(1).getReg() == RISCV::X0 &&
           Inst.getOperand(2).getImm() == 0;
  }

  bool isCNop(const MCInst &Inst) const {
    return Inst.getOpcode() == RISCV::C_NOP;
  }

  bool isNoop(const MCInst &Inst) const override {
    return isNop(Inst) || isCNop(Inst);
  }

  bool isPseudo(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return MCPlusBuilder::isPseudo(Inst);
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
      return false;
    }
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    if (!isCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JALR:
    case RISCV::C_JALR:
    case RISCV::C_JR:
      return true;
    }
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JAL:
    case RISCV::AUIPC:
      return true;
    }
  }

  unsigned getInvertedBranchOpcode(unsigned Opcode) const {
    switch (Opcode) {
    default:
      llvm_unreachable("Failed to invert branch opcode");
      return Opcode;
    case RISCV::BEQ:
      return RISCV::BNE;
    case RISCV::BNE:
      return RISCV::BEQ;
    case RISCV::BLT:
      return RISCV::BGE;
    case RISCV::BGE:
      return RISCV::BLT;
    case RISCV::BLTU:
      return RISCV::BGEU;
    case RISCV::BGEU:
      return RISCV::BLTU;
    case RISCV::C_BEQZ:
      return RISCV::C_BNEZ;
    case RISCV::C_BNEZ:
      return RISCV::C_BEQZ;
    }
  }

  void reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    auto Opcode = getInvertedBranchOpcode(Inst.getOpcode());
    Inst.setOpcode(Opcode);
    replaceBranchTarget(Inst, TBB, Ctx);
  }

  void replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");

    unsigned SymOpIndex;
    auto Result = getSymbolRefOperandNum(Inst, SymOpIndex);
    (void)Result;
    assert(Result && "unimplemented branch");

    Inst.getOperand(SymOpIndex) = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstr, unsigned &BaseRegNum,
      unsigned &IndexRegNum, int64_t &DispValue, const MCExpr *&DispExpr,
      MCInst *&PCRelBaseOut, MCInst *&FixedEntryLoadInst) const override {
    MemLocInstr = nullptr;
    BaseRegNum = 0;
    IndexRegNum = 0;
    DispValue = 0;
    DispExpr = nullptr;
    PCRelBaseOut = nullptr;
    FixedEntryLoadInst = nullptr;

    // Check for the following long tail call sequence:
    // 1: auipc xi, %pcrel_hi(sym)
    // jalr zero, %pcrel_lo(1b)(xi)
    if (Instruction.getOpcode() == RISCV::JALR && Begin != End) {
      MCInst &PrevInst = *std::prev(End);
      if (isRISCVCall(PrevInst, Instruction) &&
          Instruction.getOperand(0).getReg() == RISCV::X0)
        return IndirectBranchType::POSSIBLE_TAIL_CALL;
    }

    return IndirectBranchType::UNKNOWN;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    if (isTailCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unsupported tail call opcode");
    case RISCV::JAL:
    case RISCV::JALR:
    case RISCV::C_J:
    case RISCV::C_JR:
      break;
    }

    setTailCall(Inst);
    return true;
  }

  void createDirectCall(MCInst &Inst, const MCSymbol *Target, MCContext *Ctx,
    bool IsTailCall) override {
    unsigned Opcode = RISCV::JAL;
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(IsTailCall ? RISCV::X0 : RISCV::X1));
    Inst.addOperand(MCOperand::createExpr(getTargetExprFor(
        Inst,MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        *Ctx, 0)));
    if (IsTailCall)
      convertJmpToTailCall(Inst);
  }

  void createReturn(MCInst &Inst) const override {
    // TODO "c.jr ra" when RVC is enabled
    Inst.setOpcode(RISCV::JALR);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createReg(RISCV::X1));
    Inst.addOperand(MCOperand::createImm(0));
  }

  void createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.setOpcode(RISCV::JAL);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx)));
  }

  StringRef getTrapFillValue() const override {
    return StringRef("\0\0\0\0", 4);
  }

  void createCall(unsigned Opcode, MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) {
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(RISCVMCExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        RISCVMCExpr::VK_CALL, *Ctx)));
  }

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    return createCall(RISCV::PseudoCALL, Inst, Target, Ctx);
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createCall(RISCV::PseudoTAIL, Inst, Target, Ctx);
  }

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (isPseudo(*I) || isNoop(*I))
        continue;

      // Stop when we find the first non-terminator
      if (!isTerminator(*I) || isTailCall(*I) || !isBranch(*I))
        break;

      // Handle unconditional branches.
      if (isUnconditionalBranch(*I)) {
        // If any code was seen after this unconditional branch, we've seen
        // unreachable code. Ignore them.
        CondBranch = nullptr;
        UncondBranch = &*I;
        const MCSymbol *Sym = getTargetSymbol(*I);
        assert(Sym != nullptr &&
               "Couldn't extract BB symbol from jump operand");
        TBB = Sym;
        continue;
      }

      // Handle conditional branches and ignore indirect branches
      if (isIndirectBranch(*I))
        return false;

      if (CondBranch == nullptr) {
        const MCSymbol *TargetBB = getTargetSymbol(*I);
        if (TargetBB == nullptr) {
          // Unrecognized branch target
          return false;
        }
        FBB = TBB;
        TBB = TargetBB;
        CondBranch = &*I;
        continue;
      }

      llvm_unreachable("multiple conditional branches in one BB");
    }

    return true;
  }

  bool getSymbolRefOperandNum(const MCInst &Inst, unsigned &OpNum) const {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::C_J:
      OpNum = 0;
      return true;
    case RISCV::AUIPC:
    case RISCV::JAL:
    case RISCV::C_BEQZ:
    case RISCV::C_BNEZ:
      OpNum = 1;
      return true;
    case RISCV::BEQ:
    case RISCV::BGE:
    case RISCV::BGEU:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BLTU:
      OpNum = 2;
      return true;
    }
  }

  void createIndirectBranch(MCInst &Inst, MCPhysReg MemBaseReg,
    int64_t Disp) const {
    Inst.setOpcode(RISCV::JALR);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createReg(MemBaseReg));
    Inst.addOperand(MCOperand::createImm(Disp));
  }

  void createLD(MCInst &Inst, unsigned Dest, unsigned Base, int64_t Offset) const{
    Inst.setOpcode(RISCV::LD);
    Inst.addOperand(MCOperand::createReg(Dest));     
    Inst.addOperand(MCOperand::createReg(Base));     
    Inst.addOperand(MCOperand::createImm(Offset));   
  }

  InstructionListType createCmpJE(MCPhysReg RegNo, int64_t Imm,
    const MCSymbol *Target,
    MCContext *Ctx) const override {
      InstructionListType Code;
      Code.emplace_back(MCInstBuilder(RISCV::ADDI)
          .addReg(RegNo) 
          .addReg(RegNo)
          .addImm(-Imm));
      
      Code.emplace_back(MCInstBuilder(RISCV::BEQ)
          .addReg(RegNo)
          .addReg(RISCV::X0)   
          .addExpr(MCSymbolRefExpr::create(
              Target, MCSymbolRefExpr::VK_None, *Ctx)));
      
      return Code;
  }

  void createIndirectCallInst(MCInst &Inst, bool IsTailCall, MCPhysReg Reg) const {
    Inst.clear();
    Inst = MCInstBuilder(RISCV::JALR) 
          .addReg(IsTailCall ? RISCV::X0 : RISCV::X1)
          .addReg(Reg)
          .addImm(0); 
  }

  InstructionListType createLoadImmediate(const MCPhysReg Dest,
    uint64_t Imm) const override {
    InstructionListType Insts;
    MCInst lui= MCInstBuilder(RISCV::LUI) 
    .addReg(Dest)
    .addImm((Imm >> 12) & 0xFFFFF); 
    Insts.push_back(lui);

    MCInst addi= MCInstBuilder(RISCV::ADDI) 
    .addReg(Dest)
    .addReg(Dest)
    .addImm(Imm & 0xFFF);
    Insts.push_back(addi);

    return Insts;
  }

  void convertIndirectCallToLoad(MCInst &Inst, MCPhysReg Reg) override {
    bool IsTailCall = isTailCall(Inst);
    if (IsTailCall)
      removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
  
    if (Inst.getOpcode() == RISCV::JALR || Inst.getOpcode() == RISCV::C_JR || Inst.getOpcode() == RISCV::C_JALR) {
      Inst.setOpcode(RISCV::ADD);
      Inst.insert(Inst.begin(), MCOperand::createReg(Reg));
      Inst.insert(Inst.begin() + 1, MCOperand::createReg(RISCV::X0));
      return;
    }
    llvm_unreachable("Unsupported indirect call opcode");
  }

  InstructionListType
  createInstrIncMemory(const MCSymbol *Target, MCContext *Ctx, bool IsLeaf,
                       unsigned CodePointerSize) const override {
    // We need 2 scratch registers: one for the target address (t0/x5), and one
    // for the increment value (t1/x6).
    // addi sp, sp, -16
    // sd t0, 0(sp)
    // sd t1, 8(sp)
    // la t0, target         # 1: auipc t0, %pcrel_hi(target)
    //                       # addi t0, t0, %pcrel_lo(1b)
    // li t1, 1              # addi t1, zero, 1
    // amoadd.d zero, t0, t1
    // ld t0, 0(sp)
    // ld t1, 8(sp)
    // addi sp, sp, 16
    InstructionListType Insts;
    spillRegs(Insts, {RISCV::X10, RISCV::X11});

    createLA(Insts, RISCV::X10, Target, Ctx);

    MCInst LI = MCInstBuilder(RISCV::ADDI)
    .addReg(RISCV::X11)
    .addReg(RISCV::X0)
    .addImm(1);
    Insts.push_back(LI);

    MCInst AMOADD = MCInstBuilder(RISCV::AMOADD_D)
            .addReg(RISCV::X0)
            .addReg(RISCV::X10)
            .addReg(RISCV::X11);
    Insts.push_back(AMOADD);

    reloadRegs(Insts, {RISCV::X10, RISCV::X11});
    return Insts;
  }

  InstructionListType createInstrumentedIndirectCall(MCInst &&CallInst,
    MCSymbol *HandlerFuncAddr,
    int CallSiteID,
    MCContext *Ctx) override {
      // Code sequence used to enter indirect call instrumentation helper:
      //   spillRegs (x10,x11)
      //   
    InstructionListType Insts;
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.emplace_back(CallInst);
    convertIndirectCallToLoad(Insts.back(), RISCV::X10); 
    InstructionListType LoadImm = createLoadImmediate(RISCV::X11, CallSiteID);
    Insts.insert(Insts.end(), LoadImm.begin(), LoadImm.end());
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.resize(Insts.size() + 2);
    InstructionListType Addr = materializeAddress(HandlerFuncAddr, Ctx, RISCV::X10);
    assert(Addr.size() == 2 && "Invalid Addr size");
    std::copy(Addr.begin(), Addr.end(), Insts.end() - Addr.size());
    Insts.emplace_back();
    createIndirectCallInst(Insts.back(), isTailCall(CallInst), RISCV::X10);

    // Carry over metadata including tail call marker if present.
    stripAnnotations(Insts.back());
    moveAnnotations(std::move(CallInst), Insts.back());

    return Insts;
  }

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                            const MCSymbol *IndCallHandler,
                            MCContext *Ctx) override {
    InstructionListType Insts;
    // Code sequence used to check whether InstrTampoline was initialized
    // and call it if so, returns via IndCallHandler
    //   spillRegs x10, x11 
    //   mrs     x11, nzcv 这段没有实现，看后面怎么用
    //   adr     x10, InstrTrampoline -> adrp + add
    //   ldr     x10, [x10]
    //   addi    x10, x10, #0x0
    //   bne     x10,x0,IndCallHandler
    //   str     x30, [sp, #-16]!
    //   blr     x0
    //   ldr     x30, [sp], #16
    //   b       IndCallHandler
    spillRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.resize(Insts.size() + 2);
    InstructionListType Addr = materializeAddress(InstrTrampoline, Ctx, RISCV::X10);
    assert(Addr.size() == 2 && "Invalid Addr size");
    std::copy(Addr.begin(), Addr.end(), Insts.end() - Addr.size());
    Insts.emplace_back();
    createLD(Insts.back(), RISCV::X10, RISCV::X10, 0);

    InstructionListType cmpJmp = 
         createCmpJE(RISCV::X10, 0, IndCallHandler, Ctx);
    Insts.insert(Insts.end(), cmpJmp.begin(), cmpJmp.end());
    spillRegs(Insts, {RISCV::X1});
    MCInst JALR = MCInstBuilder(RISCV::JALR)
                  .addReg(RISCV::X1)
                  .addReg(RISCV::X10)
                  .addImm(0);
    Insts.push_back(JALR);
    spillRegs(Insts, {RISCV::X1});
    Insts.emplace_back();
    createDirectCall(Insts.back(), IndCallHandler, Ctx, /*IsTailCall*/ true);
    return Insts;
  }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    InstructionListType Insts;

    reloadRegs(Insts, {RISCV::X10, RISCV::X11});
    
    createRegInc(Insts[2], RISCV::X2, 16);   
    reloadRegs(Insts, {RISCV::X28});
    reloadRegs(Insts, {RISCV::X10, RISCV::X11});
    Insts.emplace_back();
    createIndirectBranch(Insts.back(), RISCV::X28,0);        
    return Insts;
  }
  
  InstructionListType createInstrumentedIndTailCallHandlerExitBB() const override {
     return createInstrumentedIndCallHandlerExitBB();
  }

  InstructionListType createGetter(MCContext *Ctx, const char *name) const {
    InstructionListType Insts;
    MCSymbol *Locs = Ctx->getOrCreateSymbol(name);
    Insts.resize(Insts.size() + 2);
    InstructionListType Addr = materializeAddress(Locs, Ctx, RISCV::X10);
    assert(Addr.size() == 2 && "Invalid Addr size");
    std::copy(Addr.begin(), Addr.end(), Insts.end() - Addr.size());
    Insts.emplace_back();
    createLD(Insts.back(), RISCV::X10, RISCV::X10, 0);
    Insts.emplace_back();
    createReturn(Insts.back());
    return Insts;
  }

  InstructionListType createNumCountersGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_num_counters");
  }

  InstructionListType
  createInstrLocationsGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_instr_locations");
  }

  InstructionListType createInstrTablesGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_instr_tables");
  }

  InstructionListType createInstrNumFuncsGetter(MCContext *Ctx) const override {
    return createGetter(Ctx, "__bolt_instr_num_funcs");
  }

  InstructionListType materializeAddress(const MCSymbol *Target, MCContext *Ctx,
    MCPhysReg RegName,
    int64_t Addend = 0) const override {
    InstructionListType Insts(2);

    Insts[0].setOpcode(RISCV::LUI);
    Insts[0].clear();
    Insts[0].addOperand(MCOperand::createReg(RegName));      
    Insts[0].addOperand(MCOperand::createImm(0));            
    setOperandToSymbolRef(Insts[0], /* OpNum */ 1, Target, Addend, Ctx,
    ELF::R_RISCV_HI20);

    Insts[1].setOpcode(RISCV::ADDI);
    Insts[1].clear();
    Insts[1].addOperand(MCOperand::createReg(RegName));      
    Insts[1].addOperand(MCOperand::createReg(RegName));      
    Insts[1].addOperand(MCOperand::createImm(0));            
    setOperandToSymbolRef(Insts[1], /* OpNum */ 2, Target, Addend, Ctx,
    ELF::R_RISCV_LO12_I);
    return Insts;
  }

  InstructionListType createSymbolTrampoline(const MCSymbol *TgtSym,
    MCContext *Ctx) override {
    InstructionListType Insts;
    createTailCall(Insts.emplace_back(), TgtSym, Ctx);
    return Insts;
  }

  void createLA(InstructionListType &Insts, unsigned DestReg,
    const MCSymbol *Target, MCContext *Ctx) const {
      Insts = materializeAddress(Target, Ctx, DestReg);
  }

  void createRegInc(MCInst &Inst, unsigned Reg, int64_t Imm) const {
    Inst = MCInstBuilder(RISCV::ADDI).addReg(Reg).addReg(Reg).addImm(Imm);
  }

  void createSPInc(MCInst &Inst, int64_t Imm) const {
    createRegInc(Inst, RISCV::X2, Imm);
  }

  void createStore(MCInst &Inst, unsigned Reg, unsigned BaseReg,
    int64_t Offset) const {
    Inst = MCInstBuilder(RISCV::SD).addReg(Reg).addReg(BaseReg).addImm(Offset);
  }

  void createLoad(MCInst &Inst, unsigned Reg, unsigned BaseReg,
    int64_t Offset) const {
    Inst = MCInstBuilder(RISCV::LD).addReg(Reg).addReg(BaseReg).addImm(Offset);
  }

  void spillRegs(InstructionListType &Insts,const SmallVector<unsigned> &Regs) const {
    createSPInc(Insts.emplace_back(), -Regs.size() * 8);

    int64_t Offset = 0;
    for (auto Reg : Regs) {
      createStore(Insts.emplace_back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }
  }

  void reloadRegs(InstructionListType &Insts,
    const SmallVector<unsigned> &Regs) const {
      int64_t Offset = 0;
      for (auto Reg : Regs) {
      createLoad(Insts.emplace_back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }
    createSPInc(Insts.emplace_back(), Regs.size() * 8);
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    auto *RISCVExpr = dyn_cast<RISCVMCExpr>(Expr);
    if (RISCVExpr && RISCVExpr->getSubExpr())
      return getTargetSymbol(RISCVExpr->getSubExpr());

    return MCPlusBuilder::getTargetSymbol(Expr);
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (!OpNum && !getSymbolRefOperandNum(Inst, OpNum))
      return nullptr;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    return getTargetSymbol(Op.getExpr());
  }

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  uint64_t analyzePLTEntry(MCInst &Instruction, InstructionIterator Begin,
                           InstructionIterator End,
                           uint64_t BeginPC) const override {
    auto I = Begin;

    assert(I != End);
    auto &AUIPC = *I++;
    assert(AUIPC.getOpcode() == RISCV::AUIPC);
    assert(AUIPC.getOperand(0).getReg() == RISCV::X28);

    assert(I != End);
    auto &LD = *I++;
    assert(LD.getOpcode() == RISCV::LD);
    assert(LD.getOperand(0).getReg() == RISCV::X28);
    assert(LD.getOperand(1).getReg() == RISCV::X28);

    assert(I != End);
    auto &JALR = *I++;
    (void)JALR;
    assert(JALR.getOpcode() == RISCV::JALR);
    assert(JALR.getOperand(0).getReg() == RISCV::X6);
    assert(JALR.getOperand(1).getReg() == RISCV::X28);

    assert(I != End);
    auto &NOP = *I++;
    (void)NOP;
    assert(isNoop(NOP));

    assert(I == End);

    auto AUIPCOffset = AUIPC.getOperand(1).getImm() << 12;
    auto LDOffset = LD.getOperand(2).getImm();
    return BeginPC + AUIPCOffset + LDOffset;
  }

  bool replaceImmWithSymbolRef(MCInst &Inst, const MCSymbol *Symbol,
                               int64_t Addend, MCContext *Ctx, int64_t &Value,
                               uint32_t RelType) const override {
    unsigned ImmOpNo = -1U;

    for (unsigned Index = 0; Index < MCPlus::getNumPrimeOperands(Inst);
         ++Index) {
      if (Inst.getOperand(Index).isImm()) {
        ImmOpNo = Index;
        break;
      }
    }

    if (ImmOpNo == -1U)
      return false;

    Value = Inst.getOperand(ImmOpNo).getImm();
    setOperandToSymbolRef(Inst, ImmOpNo, Symbol, Addend, Ctx, RelType);
    return true;
  }

  const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                 MCContext &Ctx,
                                 uint32_t RelType) const override {
    switch (RelType) {
    default:
      return Expr;
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_TLS_GOT_HI20:
      // The GOT is reused so no need to create GOT relocations
    case ELF::R_RISCV_PCREL_HI20:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_PCREL_HI, Ctx);
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_PCREL_LO, Ctx);
    case ELF::R_RISCV_HI20:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_HI, Ctx);
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_LO, Ctx);
    case ELF::R_RISCV_CALL:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_CALL, Ctx);
    case ELF::R_RISCV_CALL_PLT:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_CALL_PLT, Ctx);
    }
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    return false;
  }

  bool isCallAuipc(const MCInst &Inst) const {
    if (Inst.getOpcode() != RISCV::AUIPC)
      return false;

    const auto &ImmOp = Inst.getOperand(1);
    if (!ImmOp.isExpr())
      return false;

    const auto *ImmExpr = ImmOp.getExpr();
    if (!isa<RISCVMCExpr>(ImmExpr))
      return false;

    switch (cast<RISCVMCExpr>(ImmExpr)->getSpecifier()) {
    default:
      return false;
    case RISCVMCExpr::VK_CALL:
    case RISCVMCExpr::VK_CALL_PLT:
      return true;
    }
  }

  bool isRISCVCall(const MCInst &First, const MCInst &Second) const override {
    if (!isCallAuipc(First))
      return false;

    assert(Second.getOpcode() == RISCV::JALR);
    return true;
  }

  uint16_t getMinFunctionAlignment() const override {
    if (STI->hasFeature(RISCV::FeatureStdExtC) ||
        STI->hasFeature(RISCV::FeatureStdExtZca))
      return 2;
    return 4;
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createRISCVMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                        const MCInstrInfo *Info,
                                        const MCRegisterInfo *RegInfo,
                                        const MCSubtargetInfo *STI) {
  return new RISCVMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
