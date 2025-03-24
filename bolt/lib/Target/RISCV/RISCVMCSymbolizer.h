//===- bolt/Target/RISCV/RISCVMCSymbolizer.cpp --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_TARGET_RISCV_RISCVMCSYMBOLIZER_H
#define BOLT_TARGET_RISCV_RISCVMCSYMBOLIZER_H

#include "bolt/Core/BinaryFunction.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"

namespace llvm {
namespace bolt {

class RISCVMCSymbolizer : public MCSymbolizer {
protected:
  BinaryFunction &Function;
  bool CreateNewSymbols{true};

public:
  RISCVMCSymbolizer(BinaryFunction &Function, bool CreateNewSymbols = true)
      : MCSymbolizer(*Function.getBinaryContext().Ctx.get(), nullptr),
        Function(Function), CreateNewSymbols(CreateNewSymbols) {}

  RISCVMCSymbolizer(const RISCVMCSymbolizer &) = delete;
  RISCVMCSymbolizer &operator=(const RISCVMCSymbolizer &) = delete;
  virtual ~RISCVMCSymbolizer();

  bool tryAddingSymbolicOperand(MCInst &Inst, raw_ostream &CStream,
                                int64_t Value, uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t OpSize,
                                uint64_t InstSize) override;

  void tryAddingPcLoadReferenceComment(raw_ostream &CStream, int64_t Value,
                                       uint64_t Address) override;
};

} // namespace bolt
} // namespace llvm

#endif
