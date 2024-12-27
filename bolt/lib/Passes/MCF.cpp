//===- bolt/Passes/MCF.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions for solving minimum-cost flow problem.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/MCF.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SampleProfileInference.h"
#include <algorithm>
#include <map>
#include <vector>
#include<queue>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "mcf"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;
static cl::opt<bool> MaxInAndOut(
    "max-in-and-out",
    cl::desc("find the sub-graph has in and out, let it max"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> IterativeGuess(
    "iterative-guess",
    cl::desc("in non-LBR mode, guess edge counts using iterative technique"),
    cl::Hidden, cl::cat(BoltOptCategory));
static cl::opt<bool> UseProfi(
    "mcf-use-profi",
    cl::desc("in MCF, use profi algorithm"),
    cl::Hidden, cl::cat(BoltOptCategory));
cl::opt<unsigned> StaleMatchingMaxFuncSize(
    "max-func-size",
    cl::desc("The maximum size of a function to consider for inference."),
    cl::init(10000), cl::Hidden, cl::cat(BoltOptCategory));
static cl::opt<bool> SampleProfileEvenFlowDistribution(
    "bolt-sample-profile-even-flow-distribution", cl::init(true), cl::Hidden,
    cl::desc("Try to evenly distribute flow when there are multiple equally "
             "likely options."));

static cl::opt<bool> SampleProfileRebalanceUnknown(
    "bolt-sample-profile-rebalance-unknown", cl::init(true), cl::Hidden,
    cl::desc("Evenly re-distribute flow among unknown subgraphs."));

static cl::opt<bool> SampleProfileJoinIslands(
    "bolt-sample-profile-join-islands", cl::init(true), cl::Hidden,
    cl::desc("Join isolated components having positive flow."));

static cl::opt<unsigned> SampleProfileProfiCostBlockInc(
    "bolt-sample-profile-profi-cost-block-inc", cl::init(10), cl::Hidden,
    cl::desc("The cost of increasing a block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockDec(
    "bolt-sample-profile-profi-cost-block-dec", cl::init(20), cl::Hidden,
    cl::desc("The cost of decreasing a block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockEntryInc(
    "bolt-sample-profile-profi-cost-block-entry-inc", cl::init(40), cl::Hidden,
    cl::desc("The cost of increasing the entry block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockEntryDec(
    "bolt-sample-profile-profi-cost-block-entry-dec", cl::init(10), cl::Hidden,
    cl::desc("The cost of decreasing the entry block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockZeroInc(
    "bolt-sample-profile-profi-cost-block-zero-inc", cl::init(11), cl::Hidden,
    cl::desc("The cost of increasing a count of zero-weight block by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockUnknownInc(
    "bolt-sample-profile-profi-cost-block-unknown-inc", cl::init(0), cl::Hidden,
    cl::desc("The cost of increasing an unknown block's count by one."));


} // namespace opts


namespace llvm {
namespace bolt {

namespace {
  using BasicBlockListType = SmallVector<BinaryBasicBlock *, 0>;

// Edge Weight Inference Heuristic
//
// We start by maintaining the invariant used in LBR mode where the sum of
// pred edges count is equal to the block execution count. This loop will set
// pred edges count by balancing its own execution count in different pred
// edges. The weight of each edge is guessed by looking at how hot each pred
// block is (in terms of samples).
// There are two caveats in this approach. One is for critical edges and the
// other is for self-referencing blocks (loops of 1 BB). For critical edges,
// we can't infer the hotness of them based solely on pred BBs execution
// count. For each critical edge we look at the pred BB, then look at its
// succs to adjust its weight.
//
//    [ 60  ]       [ 25 ]
//       |      \     |
//    [ 10  ]       [ 75 ]
//
// The illustration above shows a critical edge \. We wish to adjust bb count
// 60 to 50 to properly determine the weight of the critical edge to be
// 50 / 75.
// For self-referencing edges, we attribute its weight by subtracting the
// current BB execution count by the sum of predecessors count if this result
// is non-negative.
using EdgeWeightMap =
    DenseMap<std::pair<const BinaryBasicBlock *, const BinaryBasicBlock *>,
             double>;

template <class NodeT>
void updateEdgeWeight(EdgeWeightMap &EdgeWeights, const BinaryBasicBlock *A,
                      const BinaryBasicBlock *B, double Weight);

template <>
void updateEdgeWeight<BinaryBasicBlock *>(EdgeWeightMap &EdgeWeights,
                                          const BinaryBasicBlock *A,
                                          const BinaryBasicBlock *B,
                                          double Weight) {
  EdgeWeights[std::make_pair(A, B)] = Weight;
}

template <>
void updateEdgeWeight<Inverse<BinaryBasicBlock *>>(EdgeWeightMap &EdgeWeights,
                                                   const BinaryBasicBlock *A,
                                                   const BinaryBasicBlock *B,
                                                   double Weight) {
  EdgeWeights[std::make_pair(B, A)] = Weight;
}

template <class NodeT>
void computeEdgeWeights(BinaryBasicBlock *BB, EdgeWeightMap &EdgeWeights) {
  typedef GraphTraits<NodeT> GraphT;
  typedef GraphTraits<Inverse<NodeT>> InvTraits;

  double TotalChildrenCount = 0.0;
  SmallVector<double, 4> ChildrenExecCount;
  // First pass computes total children execution count that directly
  // contribute to this BB.
  for (typename GraphT::ChildIteratorType CI = GraphT::child_begin(BB),
                                          E = GraphT::child_end(BB);
       CI != E; ++CI) {
    typename GraphT::NodeRef Child = *CI;
    double ChildExecCount = Child->getExecutionCount();
    // Is self-reference?
    if (Child == BB) {
      ChildExecCount = 0.0; // will fill this in second pass
    } else if (GraphT::child_end(BB) - GraphT::child_begin(BB) > 1 &&
               InvTraits::child_end(Child) - InvTraits::child_begin(Child) >
                   1) {
      // Handle critical edges. This will cause a skew towards crit edges, but
      // it is a quick solution.
      double CritWeight = 0.0;
      uint64_t Denominator = 0;
      for (typename InvTraits::ChildIteratorType
               II = InvTraits::child_begin(Child),
               IE = InvTraits::child_end(Child);
           II != IE; ++II) {
        typename GraphT::NodeRef N = *II;
        Denominator += N->getExecutionCount();
        if (N != BB)
          continue;
        CritWeight = N->getExecutionCount();
      }
      if (Denominator)
        CritWeight /= static_cast<double>(Denominator);
      ChildExecCount *= CritWeight;
    }
    ChildrenExecCount.push_back(ChildExecCount);
    TotalChildrenCount += ChildExecCount;
  }
  // Second pass fixes the weight of a possible self-reference edge
  uint32_t ChildIndex = 0;
  for (typename GraphT::ChildIteratorType CI = GraphT::child_begin(BB),
                                          E = GraphT::child_end(BB);
       CI != E; ++CI) {
    typename GraphT::NodeRef Child = *CI;
    if (Child != BB) {
      ++ChildIndex;
      continue;
    }
    if (static_cast<double>(BB->getExecutionCount()) > TotalChildrenCount) {
      ChildrenExecCount[ChildIndex] =
          BB->getExecutionCount() - TotalChildrenCount;
      TotalChildrenCount += ChildrenExecCount[ChildIndex];
    }
    break;
  }
  // Third pass finally assigns weights to edges
  ChildIndex = 0;
  for (typename GraphT::ChildIteratorType CI = GraphT::child_begin(BB),
                                          E = GraphT::child_end(BB);
       CI != E; ++CI) {
    typename GraphT::NodeRef Child = *CI;
    double Weight = 1 / (GraphT::child_end(BB) - GraphT::child_begin(BB));
    if (TotalChildrenCount != 0.0)
      Weight = ChildrenExecCount[ChildIndex] / TotalChildrenCount;
    updateEdgeWeight<NodeT>(EdgeWeights, BB, Child, Weight);
    ++ChildIndex;
  }
}

template <class NodeT>
void computeEdgeWeights(BinaryFunction &BF, EdgeWeightMap &EdgeWeights) {
  for (BinaryBasicBlock &BB : BF)
    computeEdgeWeights<NodeT>(&BB, EdgeWeights);
}

/// Make BB count match the sum of all incoming edges. If AllEdges is true,
/// make it match max(SumPredEdges, SumSuccEdges).
void recalculateBBCounts(BinaryFunction &BF, bool AllEdges) {
  for (BinaryBasicBlock &BB : BF) {
    uint64_t TotalPredsEWeight = 0;
    for (BinaryBasicBlock *Pred : BB.predecessors())
      TotalPredsEWeight += Pred->getBranchInfo(BB).Count;

    if (TotalPredsEWeight > BB.getExecutionCount())
      BB.setExecutionCount(TotalPredsEWeight);

    if (!AllEdges)
      continue;

    uint64_t TotalSuccsEWeight = 0;
    for (BinaryBasicBlock::BinaryBranchInfo &BI : BB.branch_info())
      TotalSuccsEWeight += BI.Count;

    if (TotalSuccsEWeight > BB.getExecutionCount())
      BB.setExecutionCount(TotalSuccsEWeight);
  }
}

// This is our main edge count guessing heuristic. Look at predecessors and
// assign a proportionally higher count to pred edges coming from blocks with
// a higher execution count in comparison with the other predecessor blocks,
// making SumPredEdges match the current BB count.
// If "UseSucc" is true, apply the same logic to successor edges as well. Since
// some successor edges may already have assigned a count, only update it if the
// new count is higher.
void guessEdgeByRelHotness(BinaryFunction &BF, bool UseSucc,
                           EdgeWeightMap &PredEdgeWeights,
                           EdgeWeightMap &SuccEdgeWeights) {
  for (BinaryBasicBlock &BB : BF) {
    for (BinaryBasicBlock *Pred : BB.predecessors()) {
      double RelativeExec = PredEdgeWeights[std::make_pair(Pred, &BB)];
      RelativeExec *= BB.getExecutionCount();
      BinaryBasicBlock::BinaryBranchInfo &BI = Pred->getBranchInfo(BB);
      if (static_cast<uint64_t>(RelativeExec) > BI.Count)
        BI.Count = static_cast<uint64_t>(RelativeExec);
    }

    if (!UseSucc)
      continue;

    auto BI = BB.branch_info_begin();
    for (BinaryBasicBlock *Succ : BB.successors()) {
      double RelativeExec = SuccEdgeWeights[std::make_pair(&BB, Succ)];
      RelativeExec *= BB.getExecutionCount();
      if (static_cast<uint64_t>(RelativeExec) > BI->Count)
        BI->Count = static_cast<uint64_t>(RelativeExec);
      ++BI;
    }
  }
}

using ArcSet =
    DenseSet<std::pair<const BinaryBasicBlock *, const BinaryBasicBlock *>>;

/// Predecessor edges version of guessEdgeByIterativeApproach. GuessedArcs has
/// all edges we already established their count. Try to guess the count of
/// the remaining edge, if there is only one to guess, and return true if we
/// were able to guess.
bool guessPredEdgeCounts(BinaryBasicBlock *BB, ArcSet &GuessedArcs) {
  if (BB->pred_size() == 0)
    return false;

  uint64_t TotalPredCount = 0;
  unsigned NumGuessedEdges = 0;
  for (BinaryBasicBlock *Pred : BB->predecessors()) {
    if (GuessedArcs.count(std::make_pair(Pred, BB)))
      ++NumGuessedEdges;
    TotalPredCount += Pred->getBranchInfo(*BB).Count;
  }

  if (NumGuessedEdges != BB->pred_size() - 1)
    return false;

  int64_t Guessed =
      static_cast<int64_t>(BB->getExecutionCount()) - TotalPredCount;
  if (Guessed < 0)
    Guessed = 0;

  for (BinaryBasicBlock *Pred : BB->predecessors()) {
    if (GuessedArcs.count(std::make_pair(Pred, BB)))
      continue;

    Pred->getBranchInfo(*BB).Count = Guessed;
    GuessedArcs.insert(std::make_pair(Pred, BB));
    return true;
  }
  llvm_unreachable("Expected unguessed arc");
}

/// Successor edges version of guessEdgeByIterativeApproach. GuessedArcs has
/// all edges we already established their count. Try to guess the count of
/// the remaining edge, if there is only one to guess, and return true if we
/// were able to guess.
bool guessSuccEdgeCounts(BinaryBasicBlock *BB, ArcSet &GuessedArcs) {
  if (BB->succ_size() == 0)
    return false;

  uint64_t TotalSuccCount = 0;
  unsigned NumGuessedEdges = 0;
  auto BI = BB->branch_info_begin();
  for (BinaryBasicBlock *Succ : BB->successors()) {
    if (GuessedArcs.count(std::make_pair(BB, Succ)))
      ++NumGuessedEdges;
    TotalSuccCount += BI->Count;
    ++BI;
  }

  if (NumGuessedEdges != BB->succ_size() - 1)
    return false;

  int64_t Guessed =
      static_cast<int64_t>(BB->getExecutionCount()) - TotalSuccCount;
  if (Guessed < 0)
    Guessed = 0;

  BI = BB->branch_info_begin();
  for (BinaryBasicBlock *Succ : BB->successors()) {
    if (GuessedArcs.count(std::make_pair(BB, Succ))) {
      ++BI;
      continue;
    }

    BI->Count = Guessed;
    GuessedArcs.insert(std::make_pair(BB, Succ));
    return true;
  }
  llvm_unreachable("Expected unguessed arc");
}

/// Guess edge count whenever we have only one edge (pred or succ) left
/// to guess. Then make its count equal to BB count minus all other edge
/// counts we already know their count. Repeat this until there is no
/// change.
void guessEdgeByIterativeApproach(BinaryFunction &BF) {
  ArcSet KnownArcs;
  bool Changed = false;

  do {
    Changed = false;
    for (BinaryBasicBlock &BB : BF) {
      if (guessPredEdgeCounts(&BB, KnownArcs))
        Changed = true;
      if (guessSuccEdgeCounts(&BB, KnownArcs))
        Changed = true;
    }
  } while (Changed);

  // Guess count for non-inferred edges
  for (BinaryBasicBlock &BB : BF) {
    for (BinaryBasicBlock *Pred : BB.predecessors()) {
      if (KnownArcs.count(std::make_pair(Pred, &BB)))
        continue;
      BinaryBasicBlock::BinaryBranchInfo &BI = Pred->getBranchInfo(BB);
      BI.Count =
          std::min(Pred->getExecutionCount(), BB.getExecutionCount()) / 2;
      KnownArcs.insert(std::make_pair(Pred, &BB));
    }
    auto BI = BB.branch_info_begin();
    for (BinaryBasicBlock *Succ : BB.successors()) {
      if (KnownArcs.count(std::make_pair(&BB, Succ))) {
        ++BI;
        continue;
      }
      BI->Count =
          std::min(BB.getExecutionCount(), Succ->getExecutionCount()) / 2;
      KnownArcs.insert(std::make_pair(&BB, Succ));
      break;
    }
  }
}

/// Associate each basic block with the BinaryLoop object corresponding to the
/// innermost loop containing this block.
DenseMap<const BinaryBasicBlock *, const BinaryLoop *>
createLoopNestLevelMap(BinaryFunction &BF) {
  DenseMap<const BinaryBasicBlock *, const BinaryLoop *> LoopNestLevel;
  const BinaryLoopInfo &BLI = BF.getLoopInfo();

  for (BinaryBasicBlock &BB : BF)
    LoopNestLevel[&BB] = BLI[&BB];

  return LoopNestLevel;
}

} // end anonymous namespace

void equalizeBBCounts(DataflowInfoManager &Info, BinaryFunction &BF) {
  if (BF.begin() == BF.end())
    return;

  DominatorAnalysis<false> &DA = Info.getDominatorAnalysis();
  DominatorAnalysis<true> &PDA = Info.getPostDominatorAnalysis();
  auto &InsnToBB = Info.getInsnToBBMap();
  // These analyses work at the instruction granularity, but we really only need
  // basic block granularity here. So we'll use a set of visited edges to avoid
  // revisiting the same BBs again and again.
  DenseMap<const BinaryBasicBlock *, std::set<const BinaryBasicBlock *>>
      Visited;
  // Equivalence classes mapping. Each equivalence class is defined by the set
  // of BBs that obeys the aforementioned properties.
  DenseMap<const BinaryBasicBlock *, signed> BBsToEC;
  std::vector<std::vector<BinaryBasicBlock *>> Classes;

  BF.calculateLoopInfo();
  DenseMap<const BinaryBasicBlock *, const BinaryLoop *> LoopNestLevel =
      createLoopNestLevelMap(BF);

  for (BinaryBasicBlock &BB : BF)
    BBsToEC[&BB] = -1;

  for (BinaryBasicBlock &BB : BF) {
    auto I = BB.begin();
    if (I == BB.end())
      continue;

    DA.doForAllDominators(*I, [&](const MCInst &DomInst) {
      BinaryBasicBlock *DomBB = InsnToBB[&DomInst];
      if (Visited[DomBB].count(&BB))
        return;
      Visited[DomBB].insert(&BB);
      if (!PDA.doesADominateB(*I, DomInst))
        return;
      if (LoopNestLevel[&BB] != LoopNestLevel[DomBB])
        return;
      if (BBsToEC[DomBB] == -1 && BBsToEC[&BB] == -1) {
        BBsToEC[DomBB] = Classes.size();
        BBsToEC[&BB] = Classes.size();
        Classes.emplace_back();
        Classes.back().push_back(DomBB);
        Classes.back().push_back(&BB);
        return;
      }
      if (BBsToEC[DomBB] == -1) {
        BBsToEC[DomBB] = BBsToEC[&BB];
        Classes[BBsToEC[&BB]].push_back(DomBB);
        return;
      }
      if (BBsToEC[&BB] == -1) {
        BBsToEC[&BB] = BBsToEC[DomBB];
        Classes[BBsToEC[DomBB]].push_back(&BB);
        return;
      }
      signed BBECNum = BBsToEC[&BB];
      std::vector<BinaryBasicBlock *> DomEC = Classes[BBsToEC[DomBB]];
      std::vector<BinaryBasicBlock *> BBEC = Classes[BBECNum];
      for (BinaryBasicBlock *Block : DomEC) {
        BBsToEC[Block] = BBECNum;
        BBEC.push_back(Block);
      }
      DomEC.clear();
    });
  }

  for (std::vector<BinaryBasicBlock *> &Class : Classes) {
    uint64_t Max = 0ULL;
    for (BinaryBasicBlock *BB : Class)
      Max = std::max(Max, BB->getExecutionCount());
    for (BinaryBasicBlock *BB : Class)
      BB->setExecutionCount(Max);
  }
}

void applyInference(FlowFunction &Func) {
  ProfiParams Params;
  // Set the params from the command-line flags.
  Params.EvenFlowDistribution = opts::SampleProfileEvenFlowDistribution;
  Params.RebalanceUnknown = opts::SampleProfileRebalanceUnknown;
  Params.JoinIslands = opts::SampleProfileJoinIslands;
  Params.CostBlockInc = opts::SampleProfileProfiCostBlockInc;
  Params.CostBlockDec = opts::SampleProfileProfiCostBlockDec;
  Params.CostBlockEntryInc = opts::SampleProfileProfiCostBlockEntryInc;
  Params.CostBlockEntryDec = opts::SampleProfileProfiCostBlockEntryDec;
  Params.CostBlockZeroInc = opts::SampleProfileProfiCostBlockZeroInc;
  Params.CostBlockUnknownInc = opts::SampleProfileProfiCostBlockUnknownInc;

  applyFlowInference(Params, Func);
}

void assignProfile(BinaryFunction &BF,
                   const BinaryFunction::BasicBlockOrderType &BlockOrder,
                   FlowFunction &Func) {
  BinaryContext &BC = BF.getBinaryContext();

  assert(Func.Blocks.size() == BlockOrder.size() + 1);
  for (uint64_t I = 0; I < BlockOrder.size(); I++) {
    FlowBlock &Block = Func.Blocks[I + 1];
    BinaryBasicBlock *BB = BlockOrder[I];

    // Update block's count
    BB->setExecutionCount(Block.Flow);

    // Update jump counts: (i) clean existing counts and then (ii) set new ones
    auto BI = BB->branch_info_begin();
    for (const BinaryBasicBlock *DstBB : BB->successors()) {
      (void)DstBB;
      BI->Count = 0;
      BI->MispredictedCount = 0;
      ++BI;
    }
    for (FlowJump *Jump : Block.SuccJumps) {
      if (Jump->IsUnlikely)
        continue;
      if (Jump->Flow == 0)
        continue;

      BinaryBasicBlock &SuccBB = *BlockOrder[Jump->Target - 1];
      // Check if the edge corresponds to a regular jump or a landing pad
      if (BB->getSuccessor(SuccBB.getLabel())) {
        BinaryBasicBlock::BinaryBranchInfo &BI = BB->getBranchInfo(SuccBB);
        BI.Count += Jump->Flow;
      } else {
        BinaryBasicBlock *LP = BB->getLandingPad(SuccBB.getLabel());
        if (LP && LP->getKnownExecutionCount() < Jump->Flow)
          LP->setExecutionCount(Jump->Flow);
      }
    }

    // Update call-site annotations
    auto setOrUpdateAnnotation = [&](MCInst &Instr, StringRef Name,
                                     uint64_t Count) {
      if (BC.MIB->hasAnnotation(Instr, Name))
        BC.MIB->removeAnnotation(Instr, Name);
      // Do not add zero-count annotations
      if (Count == 0)
        return;
      BC.MIB->addAnnotation(Instr, Name, Count);
    };

    for (MCInst &Instr : *BB) {
      // Ignore pseudo instructions
      if (BC.MIB->isPseudo(Instr))
        continue;
      // Ignore jump tables
      const MCInst *LastInstr = BB->getLastNonPseudoInstr();
      if (BC.MIB->getJumpTable(*LastInstr) && LastInstr == &Instr)
        continue;

      if (BC.MIB->isIndirectCall(Instr) || BC.MIB->isIndirectBranch(Instr)) {
        auto &ICSP = BC.MIB->getOrCreateAnnotationAs<IndirectCallSiteProfile>(
            Instr, "CallProfile");
        if (!ICSP.empty()) {
          // Try to evenly distribute the counts among the call sites
          const uint64_t TotalCount = Block.Flow;
          const uint64_t NumSites = ICSP.size();
          for (uint64_t Idx = 0; Idx < ICSP.size(); Idx++) {
            IndirectCallProfile &CSP = ICSP[Idx];
            uint64_t CountPerSite = TotalCount / NumSites;
            // When counts cannot be exactly distributed, increase by 1 the
            // counts of the first (TotalCount % NumSites) call sites
            if (Idx < TotalCount % NumSites)
              CountPerSite++;
            CSP.Count = CountPerSite;
          }
        } else {
          ICSP.emplace_back(nullptr, Block.Flow, 0);
        }
      } else if (BC.MIB->getConditionalTailCall(Instr)) {
        // We don't know exactly the number of times the conditional tail call
        // is executed; conservatively, setting it to the count of the block
        setOrUpdateAnnotation(Instr, "CTCTakenCount", Block.Flow);
        BC.MIB->removeAnnotation(Instr, "CTCMispredCount");
      } else if (BC.MIB->isCall(Instr)) {
        setOrUpdateAnnotation(Instr, "Count", Block.Flow);
      }
    }
  }

  // Update function's execution count and mark the function inferred.
  BF.setExecutionCount(Func.Blocks[0].Flow);
  BF.setHasInferredProfile(true);
}

bool canApplyInference(const FlowFunction &Func) {
  if (Func.Blocks.size() > opts::StaleMatchingMaxFuncSize)
    return false;

  bool HasExitBlocks = llvm::any_of(
      Func.Blocks, [&](const FlowBlock &Block) { return Block.isExit(); });
  if (!HasExitBlocks)
    return false;

  return true;
}

void preprocessUnreachableBlocks(FlowFunction &Func) {
  const uint64_t NumBlocks = Func.Blocks.size();

  // Start bfs from the source
  std::queue<uint64_t> Queue;
  std::vector<bool> VisitedEntry(NumBlocks, false);
  for (uint64_t I = 0; I < NumBlocks; I++) {
    FlowBlock &Block = Func.Blocks[I];
    if (Block.isEntry()) {
      Queue.push(I);
      VisitedEntry[I] = true;
      break;
    }
  }
  while (!Queue.empty()) {
    const uint64_t Src = Queue.front();
    Queue.pop();
    for (FlowJump *Jump : Func.Blocks[Src].SuccJumps) {
      const uint64_t Dst = Jump->Target;
      if (!VisitedEntry[Dst]) {
        Queue.push(Dst);
        VisitedEntry[Dst] = true;
      }
    }
  }

  // Start bfs from all sinks
  std::vector<bool> VisitedExit(NumBlocks, false);
  for (uint64_t I = 0; I < NumBlocks; I++) {
    FlowBlock &Block = Func.Blocks[I];
    if (Block.isExit() && VisitedEntry[I]) {
      Queue.push(I);
      VisitedExit[I] = true;
    }
  }
  while (!Queue.empty()) {
    const uint64_t Src = Queue.front();
    Queue.pop();
    for (FlowJump *Jump : Func.Blocks[Src].PredJumps) {
      const uint64_t Dst = Jump->Source;
      if (!VisitedExit[Dst]) {
        Queue.push(Dst);
        VisitedExit[Dst] = true;
      }
    }
  }

  // Make all blocks of zero weight so that flow is not sent
  for (uint64_t I = 0; I < NumBlocks; I++) {
    FlowBlock &Block = Func.Blocks[I];
    if (Block.Weight == 0)
      continue;
    if (!VisitedEntry[I] || !VisitedExit[I]) {
      Block.Weight = 0;
      Block.HasUnknownWeight = true;
      Block.IsUnlikely = true;
      for (FlowJump *Jump : Block.SuccJumps) {
        if (Jump->Source == Block.Index && Jump->Target == Block.Index) {
          Jump->Weight = 0;
          Jump->HasUnknownWeight = true;
          Jump->IsUnlikely = true;
        }
      }
    }
  }
}

FlowFunction
createFlowFunction(const BinaryFunction::BasicBlockOrderType &BlockOrder) {
  FlowFunction Func;

  // Add a special "dummy" source so that there is always a unique entry point.
  // Because of the extra source, for all other blocks in FlowFunction it holds
  // that Block.Index == BB->getIndex() + 1
  FlowBlock EntryBlock;
  EntryBlock.Index = 0;
  Func.Blocks.push_back(EntryBlock);

  // Create FlowBlock for every basic block in the binary function
  for (const BinaryBasicBlock *BB : BlockOrder) {
    Func.Blocks.emplace_back();
    FlowBlock &Block = Func.Blocks.back();
    Block.Index = Func.Blocks.size() - 1;
    Block.Weight = BB->getExecutionCount();
    if(Block.Weight != 0)  Block.HasUnknownWeight = false;
    (void)BB;
    assert(Block.Index == BB->getIndex() + 1 &&
           "incorrectly assigned basic block index");
  }

  // Create FlowJump for each jump between basic blocks in the binary function
  std::vector<uint64_t> InDegree(Func.Blocks.size(), 0);
  for (const BinaryBasicBlock *SrcBB : BlockOrder) {
    std::unordered_set<const BinaryBasicBlock *> UniqueSuccs;
    // Collect regular jumps
    for (const BinaryBasicBlock *DstBB : SrcBB->successors()) {
      // Ignoring parallel edges
      if (UniqueSuccs.find(DstBB) != UniqueSuccs.end())
        continue;

      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = SrcBB->getIndex() + 1;
      Jump.Target = DstBB->getIndex() + 1;
      // Jump.Weight =  SrcBB->getBranchInfo(*DstBB).Count;
      // if(Jump.Weight != 0) Jump.HasUnknownWeight = false;
      InDegree[Jump.Target]++;
      UniqueSuccs.insert(DstBB);
    }
    // Collect jumps to landing pads
    for (const BinaryBasicBlock *DstBB : SrcBB->landing_pads()) {
      // Ignoring parallel edges
      if (UniqueSuccs.find(DstBB) != UniqueSuccs.end())
        continue;

      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = SrcBB->getIndex() + 1;
      Jump.Target = DstBB->getIndex() + 1;
      InDegree[Jump.Target]++;
      UniqueSuccs.insert(DstBB);
    }
  }

  // Add dummy edges to the extra sources. If there are multiple entry blocks,
  // add an unlikely edge from 0 to the subsequent ones
  assert(InDegree[0] == 0 && "dummy entry blocks shouldn't have predecessors");
  for (uint64_t I = 1; I < Func.Blocks.size(); I++) {
    const BinaryBasicBlock *BB = BlockOrder[I - 1];
    if (BB->isEntryPoint() || InDegree[I] == 0) {
      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = 0;
      Jump.Target = I;
      if (!BB->isEntryPoint())
        Jump.IsUnlikely = true;
    }
  }

  // Create necessary metadata for the flow function
  for (FlowJump &Jump : Func.Jumps) {
    Func.Blocks.at(Jump.Source).SuccJumps.push_back(&Jump);
    Func.Blocks.at(Jump.Target).PredJumps.push_back(&Jump);
  }
  return Func;
}

void applyMaxInAndOut(BinaryFunction &BF){
  std::map<BinaryBasicBlock*, int64_t> change;
  for (BinaryBasicBlock &Start : BF) {
    if(change[&Start]) continue;
    // outs()<<BF.getNames()[0] <<" "<<Start.getOffset()<<"\n";
    int64_t all_flow = 28235644416000;
    std::vector<BinaryBasicBlock *> NeedNodes;
    std::map<BinaryBasicBlock*, int64_t> mp;
    std::map<BinaryBasicBlock*, int64_t> Vis;
    std::map<BinaryBasicBlock*, int64_t> In;
    std::map<BinaryBasicBlock*, int64_t> Out;
    BasicBlockListType DFS = BF.dfs();
    for(int i=0;i<DFS.size();i++){
      BinaryBasicBlock* BB = DFS[i];
      auto successors_range = BB->successors();
      for (auto succBB : successors_range) {
        Out[BB] ++;
      }
      auto predessors_range = BB->predecessors();
      for(auto pred : predessors_range){
        In[BB] ++;
      }
    }
    // outs()<<BF.getNames()[0] <<" "<<BB->getOffset()<<" ,in = "<<In[BB]<<" ,out = "<<Out[BB]<<"\n";
    BinaryBasicBlock* NowBB = &Start;
    std::queue<BinaryBasicBlock*> Queue;
    mp[NowBB] = all_flow;
    Queue.push(NowBB);
    In[NowBB] = 0;

    while(!Queue.empty()){
      NowBB = Queue.front();Queue.pop();
      if(Vis[NowBB]) continue;
      Vis[NowBB] = 1;
      auto successors_range = NowBB->successors();
      // outs()<<"mp "<< BF.getNames()[0] <<" "<<NowBB->getOffset()<<" = "<<mp[NowBB]<<"\n";
      int64_t now_flow = mp[NowBB];
      int64_t cnt = 0;
      // outs()<<"cnt = "<<cnt<<"\n";
      for (auto succBB : successors_range) {
        // outs()<<"cnt = "<<cnt<<"\n";
        cnt ++;
      }
      // outs()<<"cnt = "<<cnt<<"\n";
      for (auto succBB : successors_range) {
        mp[succBB] += now_flow / cnt;
        In[succBB] --;
        // outs()<< BF.getNames()[0] <<" "<<NowBB->getOffset()<<" -> "<<succBB->getOffset()<<"in = "<<In[succBB]<<"\n";
        if(In[succBB] == 0)Queue.push(succBB);
      }
      }
      for (BinaryBasicBlock &BB : BF) {
      if(mp[&BB] == all_flow) NeedNodes.push_back(&BB); 
      // outs()<<"mp "<< BF.getNames()[0] <<" "<<BB.getOffset()<<" = "<<mp[&BB]<<"\n";
      }
      uint64_t exeCnt = 0;
      uint64_t bbCnt = 0;
      for(int i=0;i<NeedNodes.size();i++){
        BinaryBasicBlock* BB = NeedNodes[i];
        if(In[BB] != 0) continue;
        exeCnt += BB->getExecutionCount()*BB->getNumNonPseudos();
        bbCnt += BB->getNumNonPseudos();
      }
      exeCnt /=bbCnt;
      for(int i=0;i<NeedNodes.size();i++){
        BinaryBasicBlock* BB = NeedNodes[i];
        if(In[BB] != 0) continue;
        // outs()<<"QAQ1 "<< BF.getNames()[0] <<" "<<BB->getOffset()<<" "<<BB->getExecutionCount()<<"\n";
        BB->setExecutionCount(exeCnt);
        // outs()<<"QAQ2 "<< BF.getNames()[0] <<" "<<BB->getOffset()<<" "<<BB->getExecutionCount()<<"\n";
        change[BB] = 1;
      }

  }
  return ;
}

void EstimateEdgeCounts::runOnFunction(BinaryFunction &BF) {
  if(opts::MaxInAndOut){
    applyMaxInAndOut(BF);
  }

  if(opts::UseProfi){
      const BinaryFunction::BasicBlockOrderType BlockOrder(BF.getLayout().block_begin(), BF.getLayout().block_end());
      FlowFunction Func = createFlowFunction(BlockOrder);
      preprocessUnreachableBlocks(Func);
        // Check if profile inference can be applied for the instance.
      if (canApplyInference(Func)){
          applyInference(Func);
          assignProfile(BF, BlockOrder, Func);
      }
      // OutputBB(BF,true);
      return ;
  }

  EdgeWeightMap PredEdgeWeights;
  EdgeWeightMap SuccEdgeWeights;
  if (!opts::IterativeGuess) {
    computeEdgeWeights<Inverse<BinaryBasicBlock *>>(BF, PredEdgeWeights);
    computeEdgeWeights<BinaryBasicBlock *>(BF, SuccEdgeWeights);
  }
  if (opts::EqualizeBBCounts) {
    LLVM_DEBUG(BF.print(dbgs(), "before equalize BB counts"));
    auto Info = DataflowInfoManager(BF, nullptr, nullptr);
    equalizeBBCounts(Info, BF);
    LLVM_DEBUG(BF.print(dbgs(), "after equalize BB counts"));
  }
  if (opts::IterativeGuess)
    guessEdgeByIterativeApproach(BF);
  else
    guessEdgeByRelHotness(BF, /*UseSuccs=*/false, PredEdgeWeights,
                          SuccEdgeWeights);
  recalculateBBCounts(BF, /*AllEdges=*/false);
  // for (BinaryBasicBlock &BB : BF) {
  //    outs() <<"1 "<< BF.getNames()[0] <<" "<<BB.getOffset()<<" "<<BB.getExecutionCount()<<"\n";
  // }
}

Error EstimateEdgeCounts::runOnFunctions(BinaryContext &BC) {
  if (llvm::none_of(llvm::make_second_range(BC.getBinaryFunctions()),
                    [](const BinaryFunction &BF) {
                      return BF.getProfileFlags() == BinaryFunction::PF_SAMPLE;
                    }))
    return Error::success();

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };
  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return BF.getProfileFlags() != BinaryFunction::PF_SAMPLE;
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_BB_QUADRATIC, WorkFun,
      SkipFunc, "EstimateEdgeCounts");
  return Error::success();
}

} // namespace bolt
} // namespace llvm
