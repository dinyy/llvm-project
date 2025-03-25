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

#include "RISCVMCSymbolizer.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "bolt/Utils/CommandLineOpts.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace {

class RISCVMCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::createLoad;
  using MCPlusBuilder::MCPlusBuilder;

  std::unique_ptr<MCSymbolizer>
  createTargetSymbolizer(BinaryFunction &Function,
                         bool CreateNewSymbols) const override {
    return std::make_unique<RISCVMCSymbolizer>(Function, CreateNewSymbols);
  }
  
  bool equals(const MCTargetExpr &A, const MCTargetExpr &B,
              CompFuncTy Comp) const override {
    const auto &RISCVExprA = cast<RISCVMCExpr>(A);
    const auto &RISCVExprB = cast<RISCVMCExpr>(B);
    if (RISCVExprA.getKind() != RISCVExprB.getKind())
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

  bool shouldRecordCodeRelocation(uint64_t RelType) const override {
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

bool isLoadByte(const MCInst &Inst) const{
    switch (Inst.getOpcode()) {
    // 基础指令集
    case RISCV::LB:   // 有符号字节加载
    case RISCV::LBU:  // 无符号字节加载
    // 压缩指令扩展
    case RISCV::C_LBU:
        return true;
    // 向量扩展
    // case RISCV::VLB:
    // case RISCV::VLBU:
    //     return true;
    default:
        return false;
    }
}
bool isLoadHalf(const MCInst &Inst) const{
  switch (Inst.getOpcode()) {
  case RISCV::LH:   // 有符号半字
  case RISCV::LHU:  // 无符号半字
  case RISCV::C_LHU:
  // case RISCV::VLH:
  // case RISCV::VLHU:
  //     return true;
  default:
      return false;
  }
}
// 对应LDRW（32位）
bool isLoadWord(const MCInst &Inst) const{
  switch (Inst.getOpcode()) {
  case RISCV::LW:
  case RISCV::C_LW:
  // case RISCV::VLW:
  //     return true;
  default:
      return false;
  }
}

// 对应LDRX（64位）
bool isLoadDouble(const MCInst &Inst) const{
  switch (Inst.getOpcode()) {
  case RISCV::LD:    // RV64
  case RISCV::C_LD:
  // case RISCV::VLD:
  //     return true;
  default:
      return false;
  }
}

  bool isPush(const MCInst &Inst) const override {
    return isStoreToStack(Inst);
  };

  bool isPop(const MCInst &Inst) const override {
    return isLoadFromStack(Inst);
  };

  bool isRISCVLoadReserved(const MCInst &Inst) const override{
    const unsigned Opcode = Inst.getOpcode();
    return Opcode == RISCV::LR_W ||   // 32位加载保留指令
           Opcode == RISCV::LR_D;     // 64位加载保留指令
  }
  
  bool isRISCVStoreConditional(const MCInst &Inst) const override{
    const unsigned Opcode = Inst.getOpcode();
    return Opcode == RISCV::SC_W ||   // 32位条件存储指令
           Opcode == RISCV::SC_D;     // 64位条件存储指令
  }

  bool isStoreReg(const MCInst &Inst) const{
    unsigned Opcode = Inst.getOpcode();
    // 标准存储指令
    if (Opcode >= RISCV::SB && Opcode <= RISCV::SD) return true;
    
    // 压缩指令扩展（C扩展）
    if (Opcode >= RISCV::C_SW && Opcode <= RISCV::C_SD) return true;
    
    // 向量扩展（V扩展）
    //if (Opcode >= RISCV::VSB && Opcode <= RISCV::VSXEIGEN) return true;
    
    return false;
}

bool isAtomicStore(const MCInst &Inst) const{
  // 检测原子扩展指令（A扩展）
  switch (Inst.getOpcode()) {
  case RISCV::SC_W:   // Store Conditional Word
  case RISCV::SC_D:   // Store Conditional Doubleword
  case RISCV::AMOSWAP_W:
  case RISCV::AMOADD_D:
  // ... 其他AMO指令
      return true;
  default:
      return false;
  }
  return false;
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

  /// Matching pattern for RISC-V:
///
///    AUIPC x6, imm_hi
///    ADDI  x6, x6, imm_lo
///    JALR  x0, x6, 0
///
uint64_t matchLinkerVeneer(InstructionIterator Begin, InstructionIterator End,
    uint64_t Address, const MCInst &CurInst,
    MCInst *&TargetHiBits, MCInst *&TargetLowBits,
    uint64_t &Target) const override {
  // 检查当前指令是否为 JALR x0, x6, 0
  if (CurInst.getOpcode() != RISCV::JALR || 
  !CurInst.getOperand(0).isReg() ||
  CurInst.getOperand(0).getReg() != RISCV::X0 || // rd 必须是 x0
  !CurInst.getOperand(1).isReg() ||
  CurInst.getOperand(1).getReg() != RISCV::X6 || // rs1 必须是 x6
  !CurInst.getOperand(2).isImm() ||
  CurInst.getOperand(2).getImm() != 0)           // offset 必须为 0
  return 0;

  auto I = End;
  if (I == Begin)
  return 0;

  // 检查前一条指令是否为 ADDI x6, x6, imm_lo
  --I;
  Address -= 4;  // RISC-V 指令宽度为 4 字节
  if (I == Begin || 
  I->getOpcode() != RISCV::ADDI ||
  MCPlus::getNumPrimeOperands(*I) < 3 ||
  !I->getOperand(0).isReg() ||
  I->getOperand(0).getReg() != RISCV::X6 ||  // 目标寄存器 x6
  !I->getOperand(1).isReg() ||
  I->getOperand(1).getReg() != RISCV::X6 ||  // 源寄存器 x6 
  !I->getOperand(2).isImm())
  return 0;

  TargetLowBits = &*I;
  int64_t ImmLo = I->getOperand(2).getImm();
  ImmLo = SignExtend64(ImmLo, 12);  // 符号扩展 12 位立即数

  // 检查再前一条指令是否为 AUIPC x6, imm_hi
  --I;
  Address -= 4;
  if (I->getOpcode() != RISCV::AUIPC ||
  MCPlus::getNumPrimeOperands(*I) < 2 ||
  !I->getOperand(0).isReg() ||
  I->getOperand(0).getReg() != RISCV::X6 ||  // 目标寄存器 x6
  !I->getOperand(1).isImm())
  return 0;

  TargetHiBits = &*I;
  int64_t ImmHi = I->getOperand(1).getImm();

  // 计算目标地址：
  // AUIPC 的 PC 是当前指令地址，计算结果为 (Address + (imm_hi << 12))
  // 加上 ADDI 的符号扩展 imm_lo
  Target = (Address + (ImmHi << 12)) + ImmLo;

  return 3;  // 匹配 3 条指令
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
    //TODO:analyzeIndirectBranchFragment

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
  // 默认使用非压缩指令 JAL，后续可能根据目标地址优化为压缩指令
  unsigned Opcode = IsTailCall ? RISCV::JAL : RISCV::JAL;
  Inst.setOpcode(Opcode);
  Inst.clear();

  // 添加目标寄存器：尾调用时使用 X0，否则使用 X1 (ra)
  Inst.addOperand(MCOperand::createReg(IsTailCall ? RISCV::X0 : RISCV::X1));

  // 创建符号表达式
  const MCSymbolRefExpr *Expr = 
    MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx);
  const MCExpr *TargetExpr = getTargetExprFor(Inst, Expr, *Ctx, 0);
  
  // 添加符号操作数
  Inst.addOperand(MCOperand::createExpr(TargetExpr));

  // 如果需要尾调用，转换为压缩形式或设置尾调用属性
  if (IsTailCall)
    convertJmpToTailCall(Inst);
}

void loadReg(MCInst &Inst, unsigned DestReg, unsigned AddrReg) const {
  // 根据 XLEN 选择加载指令
  Inst.setOpcode(RISCV::LD);    // 64位加载
  Inst.addOperand(MCOperand::createReg(DestReg));
  Inst.addOperand(MCOperand::createReg(AddrReg));
  Inst.addOperand(MCOperand::createImm(0)); // 偏移量 0
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
        RISCVMCExpr::VK_RISCV_CALL, *Ctx)));
  }

  void createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    return createCall(RISCV::PseudoCALL, Inst, Target, Ctx);
  }

  void createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createCall(RISCV::PseudoTAIL, Inst, Target, Ctx);
  }

bool mayStore(const MCInst &Inst) const override{
    return isStoreReg(Inst) || 
           isAtomicStore(Inst);
}
bool isStoreToStack(const MCInst &Inst) const {
  // 1. 先检查是否为存储指令
  if (!mayStore(Inst))
    return false;

  // 2. 处理压缩存储指令的特殊情况（如 C.SWSP）
  switch (Inst.getOpcode()) {
  case RISCV::C_SWSP:  // RV32C 压缩存储指令
  case RISCV::C_SDSP:  // RV64C 压缩存储指令
    return true;       // 这些指令隐式使用 sp 作为基址
  default:
    break;
  }

  // 3. 遍历源操作数检查基址寄存器
  for (const MCOperand &Operand : useOperands(Inst)) {
    if (!Operand.isReg())
      continue;

    // 4. 检查寄存器是否为栈指针（x2 或别名 sp）
    const unsigned Reg = Operand.getReg();
    if (Reg == RISCV::X2 || Reg == RISCV::SP)
      return true;
  }

  return false;
}

bool mayLoad(const MCInst &Inst) const {
  return isLoadByte(Inst)  ||   // 对应LDRB
         isLoadHalf(Inst)  ||   // 对应LDRH
         isLoadWord(Inst)  ||   // 对应LDRW
         isLoadDouble(Inst);   // 对应LDRX
}


  bool isLoadFromStack(const MCInst &Inst) const {
     // 检查压缩格式的栈加载指令
  switch (Inst.getOpcode()) {
    case RISCV::C_LWSP:  // RV32C 压缩指令
    case RISCV::C_LDSP:  // RV64C 压缩指令
      return true;
    default:
      break;
    }
  
    // 标准加载指令检查
    if (!mayLoad(Inst))
      return false;
  
    for (const MCOperand &Operand : useOperands(Inst)) {
      if (!Operand.isReg())
        continue;
      
      const unsigned Reg = Operand.getReg();
      if (Reg == RISCV::X2 || Reg == RISCV::SP)
        return true;
    }
    return false;
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
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
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
    // 创建加载寄存器指令（ld aX, offset(sp)）
void createLoadReg(MCInst &Inst, unsigned DestReg, unsigned BaseReg, int Offset) const{
  assert(BaseReg == RISCV::SP && "Only SP-based loads are supported");
  Inst = MCInstBuilder(RISCV::LD)  // RV64 的 ld 指令
      .addReg(DestReg)             // 目标寄存器（如 a0, a1, a6）
      .addReg(BaseReg)             // 基址寄存器（sp）
      .addImm(Offset);             // 偏移量（0, 8, 等）
}

// 创建立即数加法指令（addi sp, sp, imm）
void createAddImm(MCInst &Inst, unsigned DestReg, unsigned SrcReg, int Imm) const{
  assert(DestReg == SrcReg && "Only same-register addi is supported");
  Inst = MCInstBuilder(RISCV::ADDI)
      .addReg(DestReg)            // 目标寄存器（必须与源相同）
      .addReg(SrcReg)             // 源寄存器
      .addImm(Imm);               // 立即数（16, 等）
}

// 创建间接跳转指令（jr a6）
void createIndirectBranch(MCInst &Inst, unsigned TargetReg) const{
  Inst = MCInstBuilder(RISCV::JALR)
      .addReg(RISCV::X0)          // 丢弃返回地址（x0 是零寄存器）
      .addReg(TargetReg)          // 目标寄存器（a6）
      .addImm(0);                 // 偏移量（0）
}

// 生成 ADDI 指令
void createADDI(MCInst &Inst, unsigned Dest, unsigned Src, int64_t Imm) const{
  Inst.setOpcode(RISCV::ADDI);
  Inst.addOperand(MCOperand::createReg(Dest));     // 目标寄存器
  Inst.addOperand(MCOperand::createReg(Src));  // 源寄存器
  Inst.addOperand(MCOperand::createImm(Imm));      // 立即数
}

// 生成双字存储指令 (SD)
void createSD(MCInst &Inst, unsigned Src, unsigned Base, int64_t Offset) const{
  Inst.setOpcode(RISCV::SD);
  Inst.addOperand(MCOperand::createReg(Src));      // 存储数据寄存器
  Inst.addOperand(MCOperand::createReg(Base));     // 基址寄存器
  Inst.addOperand(MCOperand::createImm(Offset));   // 偏移量
}

// 生成双字加载指令 (LD)
void createLD(MCInst &Inst, unsigned Dest, unsigned Base, int64_t Offset) const{
  Inst.setOpcode(RISCV::LD);
  Inst.addOperand(MCOperand::createReg(Dest));     // 目标寄存器
  Inst.addOperand(MCOperand::createReg(Base));     // 基址寄存器
  Inst.addOperand(MCOperand::createImm(Offset));   // 偏移量
}

// 生成间接跳转指令 (JALR)
void createJALR(MCInst &Inst, unsigned Dest, unsigned Src, int64_t Offset) const{
  Inst.setOpcode(RISCV::JALR);
  Inst.addOperand(MCOperand::createReg(Dest));     // 返回地址寄存器
  Inst.addOperand(MCOperand::createReg(Src));      // 基址寄存器
  Inst.addOperand(MCOperand::createImm(Offset));   // 偏移量
}


// 生成等于零跳转指令 (BEQZ 伪指令),TODO:可以尝试压缩指令，优化
InstructionListType createBEQZ(unsigned SrcReg, const MCSymbol *Target, MCContext *Ctx) const {
  MCInst Inst;
    // 展开为标准 BEQ 指令
    Inst.setOpcode(RISCV::BEQ);
    Inst.addOperand(MCOperand::createReg(SrcReg));
    Inst.addOperand(MCOperand::createReg(RISCV::X0));

  // 添加符号表达式操作数
  const MCExpr *Expr = getTargetExprFor(
      Inst, 
      MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
      *Ctx,
      0
  );
  Inst.addOperand(MCOperand::createExpr(Expr));

  return {Inst}; // 返回包含单个指令的列表
}

void createIndirectCallInst(MCInst &Inst, bool IsTailCall, unsigned AddrReg) const {
  if (IsTailCall) {
      // 尾调用使用 JALR x0, addr_reg, 0
      Inst.setOpcode(RISCV::JALR);
      Inst.addOperand(MCOperand::createReg(RISCV::X0));
      Inst.addOperand(MCOperand::createReg(AddrReg));
      Inst.addOperand(MCOperand::createImm(0));
  } else {
      // 普通调用使用 JALR ra, addr_reg, 0
      Inst.setOpcode(RISCV::JALR);
      Inst.addOperand(MCOperand::createReg(RISCV::X1));
      Inst.addOperand(MCOperand::createReg(AddrReg));
      Inst.addOperand(MCOperand::createImm(0));
  }
}

InstructionListType createLoadImmediate(unsigned DestReg, int Value) const {
  InstructionListType Insts;
  
  // 32位立即数加载（根据 RV32/RV64 调整）
  MCInst lui;
  lui.setOpcode(RISCV::LUI);
  lui.addOperand(MCOperand::createReg(DestReg));
  lui.addOperand(MCOperand::createImm((Value + 0x800) >> 12));
  Insts.push_back(lui);

  MCInst addi;
  addi.setOpcode(RISCV::ADDI);
  addi.addOperand(MCOperand::createReg(DestReg));
  addi.addOperand(MCOperand::createReg(DestReg));
  addi.addOperand(MCOperand::createImm(Value & 0xFFF));
  Insts.push_back(addi);

  return Insts;
}

void createPushRegisters(MCInst &Inst, unsigned Reg1, unsigned Reg2) const {
  // RISC-V 需要手动调整栈指针
  Inst.setOpcode(RISCV::ADDI);
  Inst.addOperand(MCOperand::createReg(RISCV::X2));  // 目标寄存器 sp
  Inst.addOperand(MCOperand::createReg(RISCV::X2));  // 源寄存器 sp
  Inst.addOperand(MCOperand::createImm(-16));         // 调整栈指针
  
  // 存储第一个寄存器
  MCInst sd1;
  sd1.setOpcode(RISCV::SD);
  sd1.addOperand(MCOperand::createReg(Reg1));
  sd1.addOperand(MCOperand::createReg(RISCV::X2));
  sd1.addOperand(MCOperand::createImm(8)); // 偏移量 8
  Inst = sd1; // 注意：实际实现需要处理多个指令
  
  // 存储第二个寄存器（需要额外指令槽）
  // 实际实现应考虑指令序列生成方式
}

void convertIndirectCallToLoad(MCInst &Inst, MCPhysReg Reg) override {
  bool IsTailCall = isTailCall(Inst);
  if (IsTailCall)
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);

  // 处理标准 JALR 指令（非压缩格式）
  if (Inst.getOpcode() == RISCV::JALR) {
    // 确保有足够操作数：JALR 应有 3 个操作数
    if (Inst.getNumOperands() < 3)
      return;

    // 获取源寄存器（第二个操作数）
    MCPhysReg SrcReg = Inst.getOperand(1).getReg();

    // 转换为 MV 指令：ADDI rd, rs, 0
    Inst.clear();
    Inst.setOpcode(RISCV::ADDI);
    Inst.addOperand(MCOperand::createReg(Reg));    // 目标寄存器
    Inst.addOperand(MCOperand::createReg(SrcReg)); // 源寄存器
    Inst.addOperand(MCOperand::createImm(0));      // 立即数 0
    return;
  }

  // 处理压缩格式 JALR（C.JALR）
  if (Inst.getOpcode() == RISCV::C_JALR) {
    // 确保至少1个操作数
    if (Inst.getNumOperands() < 1)
      return;

    MCPhysReg SrcReg = Inst.getOperand(0).getReg();

    // 转换为 ADDI 指令
    Inst.clear();
    Inst.setOpcode(RISCV::ADDI);
    Inst.addOperand(MCOperand::createReg(Reg));   // 目标寄存器
    Inst.addOperand(MCOperand::createReg(SrcReg));// 源寄存器
    Inst.addOperand(MCOperand::createImm(0));     // 立即数 0
    return;
  }

  // 处理尾调用伪指令（PseudoTAIL）
  if (Inst.getOpcode() == RISCV::PseudoTAIL) {
    // 确保至少1个操作数
    if (Inst.getNumOperands() < 1)
      return;

    MCPhysReg SrcReg = Inst.getOperand(0).getReg();

    // 转换为 ADDI 指令
    Inst.clear();
    Inst.setOpcode(RISCV::ADDI);
    Inst.addOperand(MCOperand::createReg(Reg));   // 目标寄存器
    Inst.addOperand(MCOperand::createReg(SrcReg));// 源寄存器
    Inst.addOperand(MCOperand::createImm(0));     // 立即数 0
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
    spillRegs(Insts, {RISCV::X5, RISCV::X6});

    createLA(Insts, RISCV::X5, Target, *Ctx);

    MCInst LI = MCInstBuilder(RISCV::ADDI)
    .addReg(RISCV::X6)
    .addReg(RISCV::X0)
    .addImm(1);
    Insts.push_back(LI);

    MCInst AMOADD = MCInstBuilder(RISCV::AMOADD_D)
            .addReg(RISCV::X0)
            .addReg(RISCV::X5)
            .addReg(RISCV::X6);
    Insts.push_back(AMOADD);

    reloadRegs(Insts, {RISCV::X5, RISCV::X6});
    return Insts;
}
InstructionListType createInstrumentedIndirectCall(MCInst &&CallInst,
  MCSymbol *HandlerFuncAddr,
  int CallSiteID,
  MCContext *Ctx) override {
  InstructionListType Insts;
  const unsigned ArgReg0 = RISCV::X10; // 原A0对应X10
  const unsigned ArgReg1 = RISCV::X11; // 原A1对应X11
  const unsigned SP = RISCV::X2;       // 栈指针寄存器
  const unsigned RA = RISCV::X1;       // 返回地址寄存器
  const unsigned ZeroReg = RISCV::X0;  // 零寄存器

  // 1. 保存寄存器到栈（使用显式寄存器编号）
  Insts.emplace_back();
  createPushRegisters(Insts.back(), ArgReg0, ArgReg1); // SD x10, -16(sp); SD x11, -8(sp); ADDI sp, sp, -16

  // 2. 保留原始调用指令
  Insts.emplace_back(CallInst);

  // 3. 将调用目标转换为加载到参数寄存器 x10
  convertIndirectCallToLoad(Insts.back(), ArgReg0); // 例如：MV x10, target_reg

  // 4. 加载 CallSiteID 到 x11 寄存器
  InstructionListType LoadImm = createLoadImmediate(ArgReg1, CallSiteID);
  Insts.insert(Insts.end(), LoadImm.begin(), LoadImm.end());

  // 5. 再次保存参数寄存器到栈
  Insts.emplace_back();
  createPushRegisters(Insts.back(), ArgReg0, ArgReg1);

  // 6. 生成处理函数地址到 x10（保持寄存器编号一致性）
  Insts.resize(Insts.size() + 2);
  InstructionListType Addr = materializeAddress(HandlerFuncAddr, Ctx, ArgReg0);
  assert(Addr.size() == 2 && "Invalid Addr size");
  std::copy(Addr.begin(), Addr.end(), Insts.end() - Addr.size());

  // 7. 生成间接调用指令（使用显式寄存器编号）
  Insts.emplace_back();
  createIndirectCallInst(Insts.back(), isTailCall(CallInst), ArgReg0);

  // 8. 转移元数据
  stripAnnotations(Insts.back());
  moveAnnotations(std::move(CallInst), Insts.back());

  return Insts;
}

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                            const MCSymbol *IndCallHandler,
                            MCContext *Ctx) override {
                              InstructionListType Insts;

    // 保存 a0 和 a1 到栈中 (sp 需 16 字节对齐)
    // addi sp, sp, -16
    // sd a0, 0(sp)
    // sd a1, 8(sp)
    Insts.emplace_back();
    createADDI(Insts.back(), RISCV::SP, RISCV::SP, -16);
    Insts.emplace_back();
    createSD(Insts.back(), RISCV::X10, RISCV::SP, 0);
    Insts.emplace_back();
    createSD(Insts.back(), RISCV::X11, RISCV::SP, 8);

    // 加载 InstrTrampoline 地址到 a0
    InstructionListType Addr = 
        materializeAddress(InstrTrampoline, Ctx, RISCV::X10);
    Insts.insert(Insts.end(), Addr.begin(), Addr.end());

    // 加载 Trampoline 地址值到 a0
    // ld a0, 0(a0)
    Insts.emplace_back();
    createLD(Insts.back(), RISCV::X10, RISCV::X10, 0);

    // 检查 a0 是否为 0，若为 0 跳转到处理程序
    InstructionListType cmpJmp = 
        createBEQZ(RISCV::X10, IndCallHandler, Ctx);
    Insts.insert(Insts.end(), cmpJmp.begin(), cmpJmp.end());

    // 保存返回地址 ra 到栈中
    // addi sp, sp, -8
    // sd ra, 0(sp)
    Insts.emplace_back();
    createADDI(Insts.back(), RISCV::SP, RISCV::SP, -8);
    Insts.emplace_back();
    createSD(Insts.back(), RISCV::X1, RISCV::SP, 0);

    // 通过 Trampoline 执行间接调用
    // jalr ra, a0, 0
    Insts.emplace_back();
    createJALR(Insts.back(), RISCV::X1, RISCV::X10, 0);

    // 恢复返回地址 ra
    // ld ra, 0(sp)
    // addi sp, sp, 8
    Insts.emplace_back();
    createLD(Insts.back(), RISCV::X1, RISCV::SP, 0);
    Insts.emplace_back();
    createADDI(Insts.back(), RISCV::SP, RISCV::SP, 8);

    // 尾调用至 IndCallHandler
    Insts.emplace_back();
    createDirectCall(Insts.back(), IndCallHandler, Ctx, /*IsTailCall*/ true);

    return Insts;
  }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    InstructionListType Insts(6);
    Insts.resize(6);  // 需要6条指令（原代码的Insts[5]会越界）

    // 1. 恢复 x10 (a0) 和 x11 (a1) 寄存器
    createLoadReg(Insts[0], RISCV::X10, RISCV::X2, 0);  // ld x10, 0(x2)
    createLoadReg(Insts[1], RISCV::X11, RISCV::X2, 8);  // ld x11, 8(x2)
    createAddImm(Insts[2], RISCV::X2, RISCV::X2, 16);   // addi x2, x2, 16

    // 2. 加载目标地址到 x16 (a6)
    createLoadReg(Insts[3], RISCV::X16, RISCV::X2, 0); // ld x16, 0(x2)
    createAddImm(Insts[4], RISCV::X2, RISCV::X2, 16);  // addi x2, x2, 16

    // 3. 跳转
    createIndirectBranch(Insts[5], RISCV::X16);         // jalr x0, x16, 0
    return Insts;
}

  InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const override {
    return createInstrumentedIndCallHandlerExitBB();
  }
  
  InstructionListType createGetter(MCContext *Ctx, const char *name) const {
    InstructionListType Insts(4);
    MCSymbol *Locs = Ctx->getOrCreateSymbol(name);
    
    // 使用临时寄存器 t0 (x5) 替代 AArch64 的 X0
    const unsigned AddrReg = RISCV::X5;  // t0 寄存器
    
    // 生成符号地址到寄存器（RISC-V 需要两条指令）
    InstructionListType Addr = materializeAddress(Locs, Ctx, AddrReg);
    std::copy(Addr.begin(), Addr.end(), Insts.begin());
    assert(Addr.size() == 2 && "Invalid Addr size");
  
    // 从地址寄存器加载数据到返回值寄存器 a0 (x10)
    loadReg(Insts[2], RISCV::X10, AddrReg); // a0 = *t0
  
    // 生成返回指令（使用 JALR x0, x1, 0）
    createReturn(Insts[3]);
    
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

  InstructionListType createSymbolTrampoline(const MCSymbol *TgtSym,
                              MCContext *Ctx) override {
  InstructionListType Insts;
  createTailCall(Insts.emplace_back(), TgtSym, Ctx);
  return Insts;
  }

  const RISCVMCExpr *createSymbolRefExpr(const MCSymbol *Target,
                          RISCVMCExpr::VariantKind VK,
                          MCContext &Ctx) const {
  return RISCVMCExpr::create(MCSymbolRefExpr::create(Target, Ctx), VK, Ctx);
  }

  void createAuipcInstPair(InstructionListType &Insts, unsigned DestReg,
            const MCSymbol *Target, unsigned SecondOpcode,
            MCContext &Ctx) const {
  MCInst AUIPC = MCInstBuilder(RISCV::AUIPC)
        .addReg(DestReg)
        .addExpr(createSymbolRefExpr(
            Target, RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx));
  MCSymbol *AUIPCLabel = Ctx.createNamedTempSymbol("pcrel_hi");
  // AUIPC.setSymbol(AUIPCLabel);
  Insts.push_back(AUIPC);

  MCInst SecondInst =
  MCInstBuilder(SecondOpcode)
  .addReg(DestReg)
  .addReg(DestReg)
  .addExpr(createSymbolRefExpr(AUIPCLabel,
                          RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx));
  Insts.push_back(SecondInst);
  }

  void createLA(InstructionListType &Insts, unsigned DestReg,
  const MCSymbol *Target, MCContext &Ctx) const {
  createAuipcInstPair(Insts, DestReg, Target, RISCV::ADDI, Ctx);
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

  void spillRegs(InstructionListType &Insts,
  const SmallVector<unsigned> &Regs) const {
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

    auto *BinExpr = dyn_cast<MCBinaryExpr>(Expr);
    if (BinExpr)
      return getTargetSymbol(BinExpr->getLHS());

    auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr);
    if (SymExpr && SymExpr->getKind() == MCSymbolRefExpr::VK_None)
      return &SymExpr->getSymbol();

    return nullptr;
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
                               uint64_t RelType) const override {
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
                                 uint64_t RelType) const override {
    switch (RelType) {
    default:
      return Expr;
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_TLS_GOT_HI20:
      // The GOT is reused so no need to create GOT relocations
    case ELF::R_RISCV_PCREL_HI20:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx);
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);
    case ELF::R_RISCV_HI20:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_HI, Ctx);
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_LO, Ctx);
    case ELF::R_RISCV_CALL:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_CALL, Ctx);
    case ELF::R_RISCV_CALL_PLT:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_CALL_PLT, Ctx);
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

    switch (cast<RISCVMCExpr>(ImmExpr)->getKind()) {
    default:
      return false;
    case RISCVMCExpr::VK_RISCV_CALL:
    case RISCVMCExpr::VK_RISCV_CALL_PLT:
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

  int getPCRelEncodingSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
        // 处理 RISC-V 的分支指令
        case RISCV::BEQ:
        case RISCV::BNE:
        case RISCV::BLT:
        case RISCV::BGE:
        case RISCV::BLTU:
        case RISCV::BGEU:
            return 12; // B-type 指令有 12 位偏移量
        
        // 处理 RISC-V 的跳转指令
        case RISCV::JAL:
            return 20; // J-type 指令有 20 位偏移量
        
        // 处理其他可能的 PC 相关指令
        case RISCV::AUIPC:
            return 20; // AUIPC 指令有 20 位偏移量
        
        default:
            llvm_unreachable("Failed to get pcrel encoding size");
            return 0;
    }
}


  bool isCompressedInstruction(unsigned Opcode) const{
    // 根据RISC-V压缩指令集的opcode范围进行判断
    // 压缩指令的opcode范围通常是0x00到0x1F和0x20到0x3F等
    // 这里是一个简单的示例，实际实现需要根据具体的opcode定义来调整
      if ((Opcode >= 0x00 && Opcode <= 0x1F) || (Opcode >= 0x20 && Opcode <= 0x3F)) {
          return true;
      }
      return false;
  } 
  std::optional<uint32_t>
  getInstructionSize(const MCInst &Inst) const override  {
    
    unsigned Opcode = Inst.getOpcode();

    // 检查是否是压缩指令
    // 这里假设有一个函数或宏可以判断opcode是否属于压缩指令集
    // 例如，可以使用一个哈希表或条件判断来检查
    if (isCompressedInstruction(Opcode)) 
        return 2; // 压缩指令是2字节 
    return 4; // 基本指令是4字节
  }
  

};


} // end anonymous namespace

namespace llvm {
namespace bolt {


MCPlusBuilder *createRISCVMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                        const MCInstrInfo *Info,
                                        const MCRegisterInfo *RegInfo,
                                        const MCSubtargetInfo *STI) {
  if (opts::Instrument && !STI->getFeatureBits()[RISCV::FeatureStdExtA]) {
    errs() << "BOLT-ERROR: Instrumention on RISC-V requires the A extension "
              "but it is not enabled for the input binary";
    exit(1);
  }
  return new RISCVMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
