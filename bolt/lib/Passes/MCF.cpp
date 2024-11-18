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
/*
  1. 什么是unlikely的jump，似乎没有说明，也没看到哪个jump是unlikely的？
*/

#include "bolt/Passes/MCF.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/BitVector.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>
#include <stack>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "mcf"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> TimeOpts;

static cl::opt<bool> IterativeGuess(
    "iterative-guess",
    cl::desc("in non-LBR mode, guess edge counts using iterative technique"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> UseRArcs(
    "mcf-use-rarcs",
    cl::desc("in MCF, consider the possibility of cancelling flow to balance "
             "edges"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> UseProfi(
    "mcf-use-profi",
    cl::desc("in MCF, use profi algorithm"),
    cl::Hidden, cl::cat(BoltOptCategory));

//profi中带的参数，直接拿过来的
static cl::opt<bool> SampleProfileEvenFlowDistribution(
    "sample-profile-even-flow-distribution", cl::init(true), cl::Hidden,
    cl::desc("Try to evenly distribute flow when there are multiple equally "
             "likely options."));

static cl::opt<bool> SampleProfileRebalanceUnknown(
    "sample-profile-rebalance-unknown", cl::init(true), cl::Hidden,
    cl::desc("Evenly re-distribute flow among unknown subgraphs."));

static cl::opt<bool> SampleProfileJoinIslands(
    "sample-profile-join-islands", cl::init(true), cl::Hidden,
    cl::desc("Join isolated components having positive flow."));

static cl::opt<unsigned> SampleProfileProfiCostBlockInc(
    "sample-profile-profi-cost-block-inc", cl::init(10), cl::Hidden,
    cl::desc("The cost of increasing a block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockDec(
    "sample-profile-profi-cost-block-dec", cl::init(20), cl::Hidden,
    cl::desc("The cost of decreasing a block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockEntryInc(
    "sample-profile-profi-cost-block-entry-inc", cl::init(40), cl::Hidden,
    cl::desc("The cost of increasing the entry block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockEntryDec(
    "sample-profile-profi-cost-block-entry-dec", cl::init(10), cl::Hidden,
    cl::desc("The cost of decreasing the entry block's count by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockZeroInc(
    "sample-profile-profi-cost-block-zero-inc", cl::init(11), cl::Hidden,
    cl::desc("The cost of increasing a count of zero-weight block by one."));

static cl::opt<unsigned> SampleProfileProfiCostBlockUnknownInc(
    "sample-profile-profi-cost-block-unknown-inc", cl::init(0), cl::Hidden,
    cl::desc("The cost of increasing an unknown block's count by one."));

} // namespace opts

namespace llvm {
namespace bolt {

namespace {


/// A wrapper of a jump between two basic blocks.
struct FlowJump {
  uint64_t Source;
  uint64_t Target;
  uint64_t Weight{0};
  bool HasUnknownWeight{true};
  //这是什么意思?
  bool IsUnlikely{false};
  uint64_t Flow{0};
};

/*用到的结构体 */
/// A wrapper of a binary basic block.
struct FlowBlock {
  uint64_t Index;
  uint64_t Weight{0};
  bool HasUnknownWeight{true};
  bool IsUnlikely{false};
  uint64_t Flow{0};
  std::vector<FlowJump *> SuccJumps;
  std::vector<FlowJump *> PredJumps;

  /// Check if it is the entry block in the function.
  bool isEntry() const { return PredJumps.empty(); }

  /// Check if it is an exit block in the function.
  bool isExit() const { return SuccJumps.empty(); }
};


/// A wrapper of binary function with basic blocks and jumps.
struct FlowFunction {
  /// Basic blocks in the function.
  std::vector<FlowBlock> Blocks;
  /// Jumps between the basic blocks.
  std::vector<FlowJump> Jumps;
  /// The index of the entry block.
  uint64_t Entry{0};
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

  /// The cost of taking an unlikely block/jump.
  //设置一个unlikely的jump的代价
  const int64_t CostUnlikely = ((int64_t)1) << 30;
};




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



/// The minimum-cost maximum flow algorithm.
///
/// The algorithm finds the maximum flow of minimum cost on a given (directed)
/// network using a modified version of the classical Moore-Bellman-Ford
/// approach. The algorithm applies a number of augmentation iterations in which
/// flow is sent along paths of positive capacity from the source to the sink.
/// The worst-case time complexity of the implementation is O(v(f)*m*n), where
/// where m is the number of edges, n is the number of vertices, and v(f) is the
/// value of the maximum flow. However, the observed running time on typical
/// instances is sub-quadratic, that is, o(n^2).
///
/// The input is a set of edges with specified costs and capacities, and a pair
/// of nodes (source and sink). The output is the flow along each edge of the
/// minimum total cost respecting the given edge capacities.
static constexpr int64_t INF = ((int64_t)1) << 50;
class MinCostMaxFlow{  
public:
  MinCostMaxFlow(const ProfiParams &Params) : Params(Params) {}
  void initialize(uint64_t NodeCount,uint64_t SourceNode, uint64_t SinkNode) {
    Source = SourceNode;
    Target = SinkNode;
    Nodes = std::vector<Node>(NodeCount);
    Edges = std::vector<std::vector<Edge>>(NodeCount,std::vector<Edge>());
    if(Params.EvenFlowDistribution)
      AugmentingEdges = std::vector<std::vector<Edge *>>(NodeCount, std::vector<Edge *>());
  }
  int64_t run(){
    LLVM_DEBUG(dbgs() << "Starting profi for " << Nodes.size() << " nodes\n");
    //TODO:applyFlowAugmentation
    size_t AugmentationIters = applyFlowAugmentation();
    int64_t TotalCost = 0;
    int64_t TotalFlow = 0;
    for(uint64_t Src = 0; Src < Nodes.size() ; Src ++){
      for(auto &Edge:Edges[Src]){
        TotalCost += Edge.Cost * Edge.Flow;
        if(Src == Source)
          TotalFlow += Edge.Flow;
      }
    }
    // LLVM_DEBUG(dbgs() << "Completed profi after " << AugmentationIters
    //               << " iterations with " << TotalFlow << " total flow"
    //               << " of " << TotalCost << " cost\n");
    (void)TotalFlow;
    (void)AugmentationIters;
    return TotalCost;
  }

  void addEdge(uint64_t Src,uint64_t To,int64_t Capacity, int64_t Cost){
    assert(Capacity > 0 && "adding an edge of zero capacity");
    assert(Src != To && "loop edge are not supported");
    Edge SrcEdge;
    SrcEdge.To = To;
    SrcEdge.Cost = Cost;
    SrcEdge.Capacity = Capacity;
    SrcEdge.Flow = 0;
    SrcEdge.RevEdgeIndex = Edges[To].size();

    Edge ToEdge;
    ToEdge.To = Src;
    ToEdge.Cost = -Cost;
    ToEdge.Capacity = 0;//这个cap是否是残差网络的时候会更新
    ToEdge.Flow = 0;
    ToEdge.RevEdgeIndex = Edges[Src].size();
    
    Edges[Src].push_back(SrcEdge);
    Edges[To].push_back(ToEdge);
  }
  void addEdge(uint64_t Src,uint64_t To,int64_t Cost){
    addEdge(Src,To,INF,Cost);
  }
private:
  //返回找增广路的迭代次数
  size_t applyFlowAugmentation(){
    size_t AugmentationIters = 0;
    while(findAugmentingPath()){
      uint64_t PathCapacity = computeAugmentingPathCapacity();
      while(PathCapacity > 0){
        bool Progress = false;
        if(Params.EvenFlowDistribution){
          //找到哪些边能够成为增广路
          identifyShortestEdges(PathCapacity);

          //找到一个能够产生增广路的DAG
          auto AugmentingOrder = findAugmentingDAG();

          //把增广路用上
          Progress = augmentFlowAlongDAG(AugmentingOrder);
          PathCapacity = computeAugmentingPathCapacity();
        }

        if(!Progress){
          augmentFlowAlongPath(PathCapacity);
          PathCapacity = 0;

        }
        AugmentationIters++;
      }
    }
    return AugmentationIters;
  }
  //找增广路径
  bool findAugmentingPath(){
    for(auto &Node:Nodes){
      Node.Distance = INF;
      Node.PreNode = uint64_t(-1);
      Node.PreEdgeIndex = uint64_t(-1);
      Node.Taken = false;
    }

    std::queue<uint64_t> Queue;
    Queue.push(Source);
    Nodes[Source].Distance = 0;
    Nodes[Source].Taken = true;
    while(!Queue.empty()){
      uint64_t Src = Queue.front();
      Queue.pop();
      Nodes[Src].Taken = false;
      if(!Params.EvenFlowDistribution && Nodes[Target].Distance == 0)
        break;
      if(Nodes[Src].Distance > Nodes[Target].Distance)
        continue;
      for(uint64_t EdgeIdx = 0;EdgeIdx < Edges[Src].size();EdgeIdx ++){
        auto &Edge = Edges[Src][EdgeIdx];
        if(Edge.Flow < Edge.Capacity){
          uint64_t To = Edge.To;
          int64_t NewDistance = Nodes[Src].Distance + Edge.Cost;
          if(Nodes[To].Distance > NewDistance){
            Nodes[To].Distance = NewDistance;
            Nodes[To].PreNode = Src;
            Nodes[To].PreEdgeIndex = EdgeIdx;
            if(!Nodes[To].Taken){
              Queue.push(To);
              Nodes[To].Taken = true;
            }
          }
        }
      }
    }
    return Nodes[Target].Distance != INF;
  }

  //计算增广路的cap
  //有一个问题，这样子往回找到的路径是不均衡的，在两条都可以的情况下，是如何选择的
  uint64_t computeAugmentingPathCapacity(){
    uint64_t PathCapacity = INF;
    uint64_t Now = Target;
    while(Now != Source){
      uint64_t Pre = Nodes[Now].PreNode;
      auto &Edge = Edges[Pre][Nodes[Now].PreEdgeIndex];
      assert(Edge.Capacity >= Edge.Flow && "incorrect edge flow");
      uint64_t EdgeCapacity = uint64_t(Edge.Capacity - Edge.Flow);
      PathCapacity = std::min(PathCapacity,EdgeCapacity);
      Now = Pre;
    }
    return PathCapacity;
  }
  ///找到增广路的最短路径的候选点
  void identifyShortestEdges(uint64_t PathCapacity) {
    assert(PathCapacity > 0 && "found an incorrect augmenting DAG");
    uint64_t MinCapacity = std::max(PathCapacity/2,uint64_t(1));
    for(size_t Src = 0; Src < Nodes.size(); Src ++){
      if(Nodes[Src].Distance > Nodes[Target].Distance)
        continue;
      for(auto &Edge:Edges[Src]){
        uint64_t To = Edge.To;
        Edge.OnShortestPath = Src != Target && To != Source
          && Nodes[To].Distance <= Nodes[Target].Distance
          && Nodes[To].Distance == Nodes[Src].Distance + Edge.Cost ///为什么这里是cost?
          && Edge.Capacity > Edge.Flow
          && uint64_t(Edge.Capacity - Edge.Flow) >= MinCapacity; ///这个阈值有什么用吗?
      }
    }
  }
  //找到一个增广路的DAG排序
  std::vector<uint64_t> findAugmentingDAG() {
    typedef std::pair<uint64_t, uint64_t> StackItemType;
    std::stack<StackItemType> Stack;
    std::vector<uint64_t> AugmentingOrder;

    for(auto &Node:Nodes){
      Node.Discovery = 0;
      Node.Finish = 0;
      Node.NumCalls = 0;
      Node.Taken = false;
    }
    uint64_t Time = 0;
    Nodes[Target].Taken = true;

    Stack.emplace(Source,0);
    Nodes[Source].Discovery = ++Time;
    /// dfs搜索到一条最短路
    while(!Stack.empty()){
      auto NodeIdx = Stack.top().first;
      auto EdgeIdx = Stack.top().second;
      
      if(EdgeIdx < Edges[NodeIdx].size()){
        auto &Edge = Edges[NodeIdx][EdgeIdx];
        auto &To = Nodes[Edge.To];
        Stack.top().second ++;
        if(Edge.OnShortestPath){
          if(To.Discovery == 0 && To.NumCalls < MaxDfsCalls){
            To.Discovery = ++Time;
            Stack.emplace(Edge.To,0);
            To.NumCalls ++;
          }else if(To.Taken && To.Finish!= 0){
            Nodes[NodeIdx].Taken = true;
          }
        }
      }else{
        Stack.pop();
        ///这里是不是有改进的空间?
        if(!Nodes[NodeIdx].Taken)
          Nodes[NodeIdx].Discovery = 0;
        else{
          Nodes[NodeIdx].Finish = ++ Time;
          if(NodeIdx != Source){
            assert(!Stack.empty() && "empty stack while running bfs");
            Nodes[Stack.top().first].Taken = true;
          }
          AugmentingOrder.push_back(NodeIdx);
        }
      }
    }

    std::reverse(AugmentingOrder.begin(),AugmentingOrder.end());
    for(size_t Src:AugmentingOrder){
      AugmentingEdges[Src].clear();
      for(auto &Edge:Edges[Src]){
        uint64_t To = Edge.To;
        if(Edge.OnShortestPath && Nodes[Src].Taken && Nodes[To].Taken && Nodes[To].Finish < Nodes[Src].Finish){
          AugmentingEdges[Src].push_back(&Edge);
        }
      }
      assert((Src == Target || !AugmentingEdges[Src].empty()) && "incorrectly constructed augmenting edges");
    }
    return AugmentingOrder;
  }
  //把这条增广路用上
  //这个函数写的有点迷惑，有改进空间 ?
  bool augmentFlowAlongDAG(const std::vector<uint64_t> &AugmentingOrder){
    for(uint64_t Src:AugmentingOrder){
      Nodes[Src].FracFlow = 0;
      Nodes[Src].IntFlow = 0;
      for(auto &Edge : AugmentingEdges[Src]){
        Edge -> AugmentedFlow = 0;
      }
    }
    //表示当前被分流成多少个点
    //猜想是这样：ugmentingOrder里面放的是点，而AugmentingEdges[Src]里面放的是每个关键边
    //MaxFlowAmount的意思没有很明白?   先当成1.0的往下传，然后看每个大概占多少，以及传到最后大概剩多少，同时不断计算，从最初流过来的流量大概是多少
    uint64_t MaxFlowAmount = INF;
    Nodes[Source].FracFlow = 1.0;
    for(uint64_t Src : AugmentingOrder){
      assert((Src == Target || Nodes[Src].FracFlow > 0.0) &&"incorrectly computed fractional flow");
      uint64_t Degree = AugmentingEdges[Src].size();
      for(auto &Edge:AugmentingEdges[Src]){
        //把所有的流量平分给每个点
        double EdgeFlow = Nodes[Src].FracFlow / Degree;
        Nodes[Edge->To].FracFlow += EdgeFlow;
        if(Edge -> Capacity ==INF)
          continue;
        uint64_t MaxIntFlow = double(Edge->Capacity - Edge->Flow) / EdgeFlow;
        MaxFlowAmount = std::min(MaxFlowAmount,MaxIntFlow);
      }
    }

    if(MaxFlowAmount == 0)
      return false;

    Nodes[Source].IntFlow = MaxFlowAmount;
    for(uint64_t Src : AugmentingOrder){
      if(Src == Target)
        break;
      uint64_t Degree = AugmentingEdges[Src].size();
      uint64_t SuccFlow = (Nodes[Src].IntFlow + Degree - 1)/Degree;
      for(auto &Edge:AugmentingEdges[Src]){
        uint64_t To = Edge -> To;
        uint64_t EdgeFlow = std::min(Nodes[Src].IntFlow,SuccFlow);
        EdgeFlow = std::min(EdgeFlow,uint64_t(Edge->Capacity - Edge->Flow));
        Nodes[To].IntFlow += EdgeFlow;
        Nodes[Src].IntFlow -= EdgeFlow;
        Edge->AugmentedFlow += EdgeFlow;
      }
    }
    assert(Nodes[Target].IntFlow <= MaxFlowAmount);
    Nodes[Target].IntFlow = 0;
    for(size_t Idx = AugmentingOrder.size()-1 ; Idx > 0; Idx --){
      uint64_t Src = AugmentingOrder[Idx - 1];
      for(auto &Edge:AugmentingEdges[Src]){
        uint64_t To = Edge -> To;
        if(Nodes[To].IntFlow == 0)
          continue;
        uint64_t EdgeFlow = std::min(Nodes[To].IntFlow,Edge -> AugmentedFlow);
        Nodes[To].IntFlow -= EdgeFlow;
        Nodes[Src].IntFlow += EdgeFlow;
        Edge -> AugmentedFlow -= EdgeFlow;
      }
    }

    bool HasSaturatedEdges = false;
    for(uint64_t Src:AugmentingOrder){
      assert(Src == Source || Nodes[Src].IntFlow == 0);
      for(auto &Edge:AugmentingEdges[Src]){
        assert(uint64_t(Edge->Capacity - Edge->Flow) >= Edge->AugmentedFlow);
        auto &RevEdge = Edges[Edge->To][Edge->RevEdgeIndex];
        Edge->Flow += Edge->AugmentedFlow;
        RevEdge.Flow -= Edge->AugmentedFlow;
        if(Edge -> Capacity == Edge->Flow && Edge->AugmentedFlow > 0)
          HasSaturatedEdges = true;
      }
    }

    return HasSaturatedEdges;
  }

  void augmentFlowAlongPath(uint64_t PathCapacity){
    assert(PathCapacity > 0 && "found an incorrect augmenting path");
    uint64_t Now = Target;
    while(Now != Source){
      uint64_t Pre = Nodes[Now].PreNode;
      auto &Edge = Edges[Pre][Nodes[Now].PreEdgeIndex];
      auto &RevEdge = Edges[Now][Edge.RevEdgeIndex];
      Edge.Flow += PathCapacity;
      RevEdge.Flow -= PathCapacity;
      Now = Pre;                             
    }
  }
   


//dfs的时候最多搜索10次
static constexpr uint64_t MaxDfsCalls = 10;
struct Node{
    /// The cost of the cheapest path from the source to the current node.
    //从source到Node的最近距离
    int64_t Distance;
    /// The node preceding the current one in the path.
    //前一个节点
    uint64_t PreNode;
    /// The index of the edge between ParentNode and the current node.
    //前一个节点到当前节点边的index
    uint64_t PreEdgeIndex;
    //相当于vis
    bool Taken;
    //结束事件，什么意思还没搞懂？
    uint64_t Finsh;

    ///下面这是什么意思，为什么要分整数和分数?
    /// Data fields utilized in DAG-augmentation:
    /// Fractional flow.
    double FracFlow;
    /// Integral flow.
    uint64_t IntFlow;

    ///表示第一次被使用的时间
    uint64_t Discovery;
    /// 表示第一次放到stack中的时间
    uint64_t Finish;
    /// 表示被搜索到的次数
    uint64_t NumCalls;

};

struct Edge{
  int64_t Cost;
  int64_t Capacity;
  int64_t Flow;
  uint64_t To;
  //反向边在To的边里面的index
  uint64_t RevEdgeIndex;
  //表示这条边是否在最短路上，是否能够被选择
  bool OnShortestPath;

  //这是什么意思?
  uint64_t AugmentedFlow;
};

std::vector<Node> Nodes;
std::vector<std::vector<Edge>> Edges;
uint64_t Source;
uint64_t Target;
//从某个点出发的增广路
std::vector<std::vector<Edge *>> AugmentingEdges;
const ProfiParams &Params;

};




/*
  adjust Flow：
  1. 把所有的island进行处理：找一条最短路，然后流量+1
  2. 识别所有的unknown子图,把所有unknown的每条路的概率都变成50%
*/
class FlowAdjuster{
public:
  FlowAdjuster(const ProfiParams &Params,FlowFunction &Func):Params(Params),Func(Func){}
  void run(){
    if(Params.JoinIslands){
      joinIsolatedComponents();
    }
    if(Params.RebalanceUnknown){
      rebalanceUnknownSubgraphs();
    }
  }
private:
  void joinIsolatedComponents(){
    //找到所有能从source流到的block
    auto Visited = BitVector(NumBlocks(),false);
    findReachable(Func.Entry,Visited);

    for(uint64_t I = 0 ;I < NumBlocks();I++){
      auto &Block=Func.Blocks[I];
      if(Block.Flow > 0 && !Visited[I]){
        auto Path = findShortestPath(I);
        assert(Path.size() > 0 && Path[0]->Source == Func.Entry &&
               "incorrectly computed path adjusting control flow");
        Func.Blocks[Func.Entry].Flow += 1;
        for (auto &Jump : Path) {
          Jump->Flow += 1;
          Func.Blocks[Jump->Target].Flow += 1;
          // 这里的findReachable是不是没必要?为什么不加完以后直接findReachable?
          findReachable(Jump->Target, Visited);
        }
      }
    }
  }
  //bfs搜一遍所有有值的点
  void findReachable(uint64_t Src,BitVector &Visited){
    if(Visited[Src])
      return ;
    std::queue<uint64_t> Queue;
    Queue.push(Src);
    Visited[Src]=true;
    while(!Queue.empty()){
      Src = Queue.front();
      Queue.pop();
      for(auto *Jump:Func.Blocks[Src].SuccJumps){
        uint64_t Dst = Jump -> Target;
        if(Jump -> Flow > 0 && !Visited[Dst]){
          Queue.push(Dst);
          Visited[Dst] = true;
        }
      }
    }
  }

  std::vector<FlowJump *> findShortestPath(uint64_t BlockIdx){
    //源点到当前点的路径
    auto ForwardPath =  findShortestPath(Func.Entry, BlockIdx);
    //当前点到汇点的路径
    auto BackwardPath = findShortestPath(BlockIdx, AnyExitBlock);

    std::vector<FlowJump *> Result;
    Result.insert(Result.end(), ForwardPath.begin(), ForwardPath.end());
    Result.insert(Result.end(), BackwardPath.begin(), BackwardPath.end());
    return Result;
  }

  //用dijk找最短路
  std::vector<FlowJump *>findShortestPath(uint64_t Source,uint64_t Target){
    if(Source == Target)
      return std::vector<FlowJump *>();
    if(Func.Blocks[Source].isExit() && Target == AnyExitBlock)
      return std::vector<FlowJump *>();

    auto Distance = std::vector<int64_t>(NumBlocks(), INF);
    auto Parent = std::vector<FlowJump *>(NumBlocks(), nullptr);
    Distance[Source] = 0;
    std::set<std::pair<uint64_t,uint64_t>> Queue;
    Queue.insert(std::make_pair(Distance[Source], Source));

    //这里这种dijk写法是否有改进空间
    while(!Queue.empty()){
      uint64_t Src = Queue.begin() -> second;
      if(Src == Target || (Func.Blocks[Src].isExit() && Target == AnyExitBlock))
        break;
      for(auto *Jump:Func.Blocks[Src].SuccJumps){
        uint64_t Dst = Jump -> Target;
        int64_t JumpDist = jumpDistance(Jump);
        if(Distance[Dst] > Distance[Src] + JumpDist){
          Queue.erase(std::make_pair(Distance[Dst], Dst));
          Distance[Dst] = Distance[Src] + JumpDist;
          Parent[Dst] = Jump;
          Queue.insert(std::make_pair(Distance[Dst], Dst));
        }
      }
    }
    //如果没有给定Target，指定最近的exit block
    if(Target == AnyExitBlock){
      for(uint64_t I = 0; I < NumBlocks() ; I++){
        if(Func.Blocks[I].isExit() && Parent[I] != nullptr){
          if(Target == AnyExitBlock || Distance[Target] > Distance[I])
            Target = I;
        }
      }
    }
    assert(Parent[Target] != nullptr && "a path does not exist");

    // Extract the constructed path
    std::vector<FlowJump *> Result;
    uint64_t Now = Target;
    while (Now != Source) {
      assert(Now == Parent[Now]->Target && "incorrect parent jump");
      Result.push_back(Parent[Now]);
      Now = Parent[Now]->Source;
    }
    // Reverse the path, since it is extracted from Target to Source
    std::reverse(Result.begin(), Result.end());
    return Result;
  }

  /// A distance of a path for a given jump.
  /// In order to incite the path to use blocks/jumps with large positive flow,
  /// and avoid changing branch probability of outgoing edges drastically,
  /// set the jump distance so as:
  ///   - to minimize the number of unlikely jumps used and subject to that,
  ///   - to minimize the number of Flow == 0 jumps used and subject to that,
  ///   - minimizes total multiplicative Flow increase for the remaining edges.
  /// To capture this objective with integer distances, we round off fractional
  /// parts to a multiple of 1 / BaseDistance.
  /// 这里似乎有改进空间?
  int64_t jumpDistance(FlowJump *Jump) const{
    if(Jump->IsUnlikely)
      return Params.CostUnlikely;
    uint64_t BaseDistance = std::max(FlowAdjuster::MinBaseDistance,
                                     std::min(Func.Blocks[Func.Entry].Flow,Params.CostUnlikely/(2*(NumBlocks()+1))));
    if(Jump->Flow > 0)
      return BaseDistance + BaseDistance / Jump->Flow;
    return 2 * BaseDistance * (NumBlocks() + 1);
  } 

  void rebalanceUnknownSubgraphs() {
    //找到unknown的子图
    for(const FlowBlock &SrcBlock:Func.Blocks){
      if (!canRebalanceAtRoot(&SrcBlock))
        continue;

      //找到一个unknown的子图，然后沿着这条路直接填充 
      std::vector<FlowBlock *> UnknownBlocks;
      std::vector<FlowBlock *> KnownDstBlocks;
      findUnknownSubgraph(&SrcBlock, KnownDstBlocks, UnknownBlocks);
      FlowBlock *DstBlock = nullptr;
      //确定重新rebalance这个子图是否可行
      if (!canRebalanceSubgraph(&SrcBlock, KnownDstBlocks, UnknownBlocks,
                                DstBlock))
        continue;
      //里面有cycle的子图也无法处理
      // We cannot rebalance subgraphs containing cycles among unknown blocks
      if (!isAcyclicSubgraph(&SrcBlock, DstBlock, UnknownBlocks))
        continue;

      rebalanceUnknownSubgraph(&SrcBlock, DstBlock, UnknownBlocks);
    }
  
  }
  ///没有仔细看过?
    /// Rebalance a given subgraph rooted at SrcBlock, ending at DstBlock and
  /// having UnknownBlocks intermediate blocks.
  void rebalanceUnknownSubgraph(const FlowBlock *SrcBlock,
                                const FlowBlock *DstBlock,
                                const std::vector<FlowBlock *> &UnknownBlocks) {
    assert(SrcBlock->Flow > 0 && "zero-flow block in unknown subgraph");

    // Ditribute flow from the source block
    uint64_t BlockFlow = 0;
    // SrcBlock's flow is the sum of outgoing flows along non-ignored jumps
    for (auto *Jump : SrcBlock->SuccJumps) {
      if (ignoreJump(SrcBlock, DstBlock, Jump))
        continue;
      BlockFlow += Jump->Flow;
    }
    rebalanceBlock(SrcBlock, DstBlock, SrcBlock, BlockFlow);

    // Ditribute flow from the remaining blocks
    for (auto *Block : UnknownBlocks) {
      assert(Block->HasUnknownWeight && "incorrect unknown subgraph");
      uint64_t BlockFlow = 0;
      // Block's flow is the sum of incoming flows
      for (auto *Jump : Block->PredJumps) {
        BlockFlow += Jump->Flow;
      }
      Block->Flow = BlockFlow;
      rebalanceBlock(SrcBlock, DstBlock, Block, BlockFlow);
    }
  }
  
  
  ///没有仔细看过?
  /// Redistribute flow for a block in a subgraph rooted at SrcBlock,
  /// and ending at DstBlock.
  void rebalanceBlock(const FlowBlock *SrcBlock, const FlowBlock *DstBlock,
                      const FlowBlock *Block, uint64_t BlockFlow) {
    // Process all successor jumps and update corresponding flow values
    size_t BlockDegree = 0;
    for (auto *Jump : Block->SuccJumps) {
      if (ignoreJump(SrcBlock, DstBlock, Jump))
        continue;
      BlockDegree++;
    }
    // If all successor jumps of the block are ignored, skip it
    if (DstBlock == nullptr && BlockDegree == 0)
      return;
    assert(BlockDegree > 0 && "all outgoing jumps are ignored");

    // Each of the Block's successors gets the following amount of flow.
    // Rounding the value up so that all flow is propagated
    uint64_t SuccFlow = (BlockFlow + BlockDegree - 1) / BlockDegree;
    for (auto *Jump : Block->SuccJumps) {
      if (ignoreJump(SrcBlock, DstBlock, Jump))
        continue;
      uint64_t Flow = std::min(SuccFlow, BlockFlow);
      Jump->Flow = Flow;
      BlockFlow -= Flow;
    }
    assert(BlockFlow == 0 && "not all flow is propagated");
  }

  ///确认子图是否有cycle，还没仔细看过规则?
    /// Verify if the given unknown subgraph is acyclic, and if yes, reorder
  /// UnknownBlocks in the topological order (so that all jumps are "forward").
  bool isAcyclicSubgraph(const FlowBlock *SrcBlock, const FlowBlock *DstBlock,
                         std::vector<FlowBlock *> &UnknownBlocks) {
    // Extract local in-degrees in the considered subgraph
    auto LocalInDegree = std::vector<uint64_t>(NumBlocks(), 0);
    auto fillInDegree = [&](const FlowBlock *Block) {
      for (auto *Jump : Block->SuccJumps) {
        if (ignoreJump(SrcBlock, DstBlock, Jump))
          continue;
        LocalInDegree[Jump->Target]++;
      }
    };
    fillInDegree(SrcBlock);
    for (auto *Block : UnknownBlocks) {
      fillInDegree(Block);
    }
    // A loop containing SrcBlock
    if (LocalInDegree[SrcBlock->Index] > 0)
      return false;

    std::vector<FlowBlock *> AcyclicOrder;
    std::queue<uint64_t> Queue;
    Queue.push(SrcBlock->Index);
    while (!Queue.empty()) {
      FlowBlock *Block = &Func.Blocks[Queue.front()];
      Queue.pop();
      // Stop propagation once we reach DstBlock, if any
      if (DstBlock != nullptr && Block == DstBlock)
        break;

      // Keep an acyclic order of unknown blocks
      if (Block->HasUnknownWeight && Block != SrcBlock)
        AcyclicOrder.push_back(Block);

      // Add to the queue all successors with zero local in-degree
      for (auto *Jump : Block->SuccJumps) {
        if (ignoreJump(SrcBlock, DstBlock, Jump))
          continue;
        uint64_t Dst = Jump->Target;
        LocalInDegree[Dst]--;
        if (LocalInDegree[Dst] == 0) {
          Queue.push(Dst);
        }
      }
    }

    // If there is a cycle in the subgraph, AcyclicOrder contains only a subset
    // of all blocks
    if (UnknownBlocks.size() != AcyclicOrder.size())
      return false;
    UnknownBlocks = AcyclicOrder;
    return true;
  }



  ///还没仔细看过?
    /// Verify if rebalancing of the subgraph is feasible. If the checks are
  /// successful, set the unique destination block, DstBlock (can be null).
  bool canRebalanceSubgraph(const FlowBlock *SrcBlock,
                            const std::vector<FlowBlock *> &KnownDstBlocks,
                            const std::vector<FlowBlock *> &UnknownBlocks,
                            FlowBlock *&DstBlock) {
    // If the list of unknown blocks is empty, we don't need rebalancing
    if (UnknownBlocks.empty())
      return false;

    // If there are multiple known sinks, we can't rebalance
    if (KnownDstBlocks.size() > 1)
      return false;
    DstBlock = KnownDstBlocks.empty() ? nullptr : KnownDstBlocks.front();

    // Verify sinks of the subgraph
    for (auto *Block : UnknownBlocks) {
      if (Block->SuccJumps.empty()) {
        // If there are multiple (known and unknown) sinks, we can't rebalance
        if (DstBlock != nullptr)
          return false;
        continue;
      }
      size_t NumIgnoredJumps = 0;
      for (auto *Jump : Block->SuccJumps) {
        if (ignoreJump(SrcBlock, DstBlock, Jump))
          NumIgnoredJumps++;
      }
      // If there is a non-sink block in UnknownBlocks with all jumps ignored,
      // then we can't rebalance
      if (NumIgnoredJumps == Block->SuccJumps.size())
        return false;
    }

    return true;
  }

  //一路把这个unknown的blocks给推下去，为什么要这么做?  
  void findUnknownSubgraph(const FlowBlock *SrcBlock,std::vector<FlowBlock *> &KnownDstBlocks,std::vector<FlowBlock *> &UnknownBlocks) {
    auto Visited = BitVector(NumBlocks(), false);
    std::queue<uint64_t> Queue;

    Queue.push(SrcBlock->Index);
    Visited[SrcBlock->Index] = true;
    while(!Queue.empty()){
      auto &Block = Func.Blocks[Queue.front()];
      Queue.pop();
      for(auto *Jump:Block.SuccJumps){
        if(ignoreJump(SrcBlock, nullptr, Jump))
          continue;

        uint64_t Dst = Jump->Target;
        if(Visited[Dst])
          continue;
        Visited[Dst]=true;
        if (!Func.Blocks[Dst].HasUnknownWeight) {
          KnownDstBlocks.push_back(&Func.Blocks[Dst]);
        } else {
          Queue.push(Dst);
          UnknownBlocks.push_back(&Func.Blocks[Dst]);
        }
      }
    }

  
  }

  //决定这个jump是否要ignored，具体规则有点奇怪?
  bool ignoreJump(const FlowBlock *SrcBlock, const FlowBlock *DstBlock,
                  const FlowJump *Jump) {
    // Ignore unlikely jumps with zero flow
    if (Jump->IsUnlikely && Jump->Flow == 0)
      return true;

    auto JumpSource = &Func.Blocks[Jump->Source];
    auto JumpTarget = &Func.Blocks[Jump->Target];

    // Do not ignore jumps coming into DstBlock
    if (DstBlock != nullptr && JumpTarget == DstBlock)
      return false;

    // Ignore jumps out of SrcBlock to known blocks
    if (!JumpTarget->HasUnknownWeight && JumpSource == SrcBlock)
      return true;

    // Ignore jumps to known blocks with zero flow
    if (!JumpTarget->HasUnknownWeight && JumpTarget->Flow == 0)
      return true;

    return false;
  }

  //判断是否可以从root进行rebalance
  bool canRebalanceAtRoot(const FlowBlock *SrcBlock) {
    // Do not attempt to find unknown subgraphs from an unknown or a
    // zero-flow block+
    if (SrcBlock->HasUnknownWeight || SrcBlock->Flow == 0)
      return false;

    // Do not attempt to process subgraphs from a block w/o unknown sucessors
    bool HasUnknownSuccs = false;
    for (auto *Jump : SrcBlock->SuccJumps) {
      if (Func.Blocks[Jump->Target].HasUnknownWeight) {
        HasUnknownSuccs = true;
        break;
      }
    }
    if (!HasUnknownSuccs)
      return false;

    return true;
  }

  

const ProfiParams &Params;
FlowFunction &Func;
uint64_t NumBlocks() const { return Func.Blocks.size(); }
static constexpr uint64_t AnyExitBlock = uint64_t(-1);
//最小的baseDis，在island join的时候
static constexpr uint64_t MinBaseDistance = 10000;
};



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


void InitializationMCF(BinaryFunction &BF)
{
  llvm_unreachable("not implemented");

}

void solveMCF(BinaryFunction &BF, MCFCostFunction CostFunction) {
  llvm_unreachable("not implemented");


}


void estimateEdgeCounts(BinaryFunction &BF) {
  if(opts::UseProfi){
    

    InitializationMCF(BF);
    solveMCF(BF,MCF_PROFI);
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
}


} // namespace bolt
} // namespace llvm
