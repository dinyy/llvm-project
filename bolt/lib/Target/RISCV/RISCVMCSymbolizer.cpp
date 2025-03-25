//===- bolt/Target/RISCV/RISCVMCSymbolizer.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVMCSymbolizer.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/Relocation.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"
#include "MCTargetDesc/RISCVMCExpr.h"

#define DEBUG_TYPE "bolt-symbolizer"

namespace llvm {
namespace bolt {

    RISCVMCSymbolizer::~RISCVMCSymbolizer() {}

bool RISCVMCSymbolizer::tryAddingSymbolicOperand(
    MCInst &Inst, raw_ostream &CStream, int64_t Value, uint64_t InstAddress,
    bool IsBranch, uint64_t ImmOffset, uint64_t ImmSize, uint64_t InstSize) {
  BinaryContext &BC = Function.getBinaryContext();
  MCContext *Ctx = BC.Ctx.get();

  // NOTE: the callee may incorrectly set IsBranch.
  if (BC.MIB->isBranch(Inst) || Inst.getOpcode() == RISCV::JAL || Inst.getOpcode() == RISCV::JALR || BC.MIB->isCall(Inst))
    return false;
  const uint64_t InstOffset = InstAddress - Function.getAddress();
  const Relocation *Relocation = Function.getRelocationAt(InstOffset);
  
    /// Add symbolic operand to the instruction with an optional addend.
  auto addOperand = [&](const MCSymbol *Symbol, uint64_t Addend,
      uint64_t RelType) {
    const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, *Ctx);
    switch (RelType) {
      case ELF::R_RISCV_HI20:
        Expr = RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_HI, *Ctx);
        break;
      case ELF::R_RISCV_LO12_I:
      case ELF::R_RISCV_LO12_S:
        Expr = RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_LO, *Ctx);
        break;
    }


    if (Addend)
    Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Addend, *Ctx),
                      *Ctx);
    Inst.addOperand(MCOperand::createExpr(
    BC.MIB->getTargetExprFor(Inst, Expr, *Ctx, RelType)));
  };

  if (Relocation) {
    switch (Relocation->Type) {
    case ELF::R_RISCV_CALL:
    case ELF::R_RISCV_CALL_PLT: {
      const MCSymbol *Symbol = Ctx->getOrCreateSymbol(Relocation->Symbol->getName());
      addOperand(Symbol, Relocation->Addend, Relocation->Type);
      return true;
    }
    case ELF::R_RISCV_PCREL_HI20: {
      const MCSymbol *Symbol = Ctx->getOrCreateSymbol(Relocation->Symbol->getName());
      addOperand(Symbol, Relocation->Addend, ELF::R_RISCV_PCREL_HI20);
      // 可能需要添加对应的 LO12 重定位
      return true;
    }
    // 处理其他 RISC-V 重定位类型
    }
  }  
  Value += InstAddress;
  const MCSymbol *TargetSymbol;
  uint64_t TargetOffset;
  if (!CreateNewSymbols) {
    if (BinaryData *BD = BC.getBinaryDataContainingAddress(Value)) {
      TargetSymbol = BD->getSymbol();
      TargetOffset = Value - BD->getAddress();
    } else {
      return false;
    }
  } else {
    std::tie(TargetSymbol, TargetOffset) =
        BC.handleAddressRef(Value, Function, /*IsPCRel*/ true);
  }
  addOperand(TargetSymbol, TargetOffset, 0);

  return true;
}

void RISCVMCSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &CStream,
                                                          int64_t Value,
                                                          uint64_t Address) {}

} // namespace bolt
} // namespace llvm
