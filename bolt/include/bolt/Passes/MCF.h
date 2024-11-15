//===- bolt/Passes/MCF.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_MCF_H
#define BOLT_PASSES_MCF_H

namespace llvm {
namespace bolt {

class BinaryFunction;
class DataflowInfoManager;

enum MCFCostFunction : char {
  MCF_DISABLE = 0,
  MCF_LINEAR,
  MCF_QUADRATIC,
  MCF_LOG, 
  MCF_BLAMEFTS,
  MCF_PROFI
};


/// Various thresholds and options controlling the behavior of the profile
/// inference algorithm. Default values are tuned for several large-scale
/// applications, and can be modified via corresponding command-line flags.
struct ProfiParams {
  /// Evenly distribute flow when there are multiple equally likely options.
  bool EvenFlowDistribution{false};

  /// Evenly re-distribute flow among unknown subgraphs.
  bool RebalanceUnknown{false};

  /// Join isolated components having positive flow.
  bool JoinIslands{false};

  /// The cost of increasing a block's count by one.
  unsigned CostBlockInc{0};

  /// The cost of decreasing a block's count by one.
  unsigned CostBlockDec{0};

  /// The cost of increasing a count of zero-weight block by one.
  unsigned CostBlockZeroInc{0};

  /// The cost of increasing the entry block's count by one.
  unsigned CostBlockEntryInc{0};

  /// The cost of decreasing the entry block's count by one.
  unsigned CostBlockEntryDec{0};

  /// The cost of increasing an unknown block's count by one.
  unsigned CostBlockUnknownInc{0};

  /// The cost of increasing a jump's count by one.
  unsigned CostJumpInc{0};

  /// The cost of increasing a fall-through jump's count by one.
  unsigned CostJumpFTInc{0};

  /// The cost of decreasing a jump's count by one.
  unsigned CostJumpDec{0};

  /// The cost of decreasing a fall-through jump's count by one.
  unsigned CostJumpFTDec{0};

  /// The cost of increasing an unknown jump's count by one.
  unsigned CostJumpUnknownInc{0};

  /// The cost of increasing an unknown fall-through jump's count by one.
  unsigned CostJumpUnknownFTInc{0};

};

/// Implement the idea in "SamplePGO - The Power of Profile Guided Optimizations
/// without the Usability Burden" by Diego Novillo to make basic block counts
/// equal if we show that A dominates B, B post-dominates A and they are in the
/// same loop and same loop nesting level.
void equalizeBBCounts(DataflowInfoManager &Info, BinaryFunction &BF);

/// Fill edge counts based on the basic block count. Used in nonLBR mode when
/// we only have bb count.
void estimateEdgeCounts(BinaryFunction &BF);

/// Entry point for computing a min-cost flow for the CFG with the goal
/// of fixing the flow of the CFG edges, that is, making sure it obeys the
/// flow-conservation equation  SumInEdges = SumOutEdges.
///
/// To do this, we create an instance of the min-cost flow problem in a
/// similar way as the one discussed in the work of Roy Levin "Completing
/// Incomplete Edge Profile by Applying Minimum Cost Circulation Algorithms".
/// We do a few things differently, though. We don't populate edge counts using
/// weights coming from a static branch prediction technique and we don't
/// use the same cost function.
///
/// If cost function BlameFTs is used, assign all remaining flow to
/// fall-throughs. This is used when the sampling is based on taken branches
/// that do not account for them.
void solveMCF(BinaryFunction &BF, MCFCostFunction CostFunction);

} // end namespace bolt
} // end namespace llvm

#endif // BOLT_PASSES_MCF_H
