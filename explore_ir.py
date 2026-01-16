#!/usr/bin/env python3
"""
IR Evolution Explorer for tinygrad

This educational script traces tensor operations through tinygrad's compilation pipeline,
capturing and annotating IR at each transformation stage.

Pipeline stages captured:
1. Tensor API -> Lazy UOp Graph (tensor.uop)
2. Schedule -> Kernel UOps (ExecItem list)
3. Codegen passes -> Optimized UOps
4. Linearize -> Linear IR
5. Render -> Device Code

Usage:
  python explore_ir.py                    # Run all examples
  python explore_ir.py --example add      # Run specific example
  python explore_ir.py --report           # Generate markdown report
  python explore_ir.py --device CPU       # Use specific device
  python explore_ir.py --noopt            # Disable optimizations
  DEBUG=2 python explore_ir.py            # More verbose output
  VIZ=-1 python explore_ir.py             # Save VIZ trace for later exploration
"""
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

# Set up environment before imports
os.environ.setdefault("DEBUG", "0")

from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, pyrender, print_uops
from tinygrad.engine.schedule import complete_create_schedule_with_vars
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.codegen import full_rewrite_to_sink, get_program
from tinygrad.renderer import ProgramSpec, Estimates
from tinygrad.helpers import colored, DEBUG, NOOPT, getenv

# ============================================================================
# Data structures for capturing IR stages
# ============================================================================

@dataclass
class IRStage:
  """Represents a single stage in the IR pipeline."""
  name: str
  description: str
  uop: UOp | None = None
  uops_list: list[UOp] | None = None
  source_code: str | None = None
  estimates: Estimates | None = None

  def summary(self) -> str:
    """Return a brief summary of this stage."""
    if self.uop is not None:
      ops = list(self.uop.toposort())
      op_counts = {}
      for u in ops:
        op_counts[u.op] = op_counts.get(u.op, 0) + 1
      top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:5]
      return f"{len(ops)} UOps, top: {', '.join(f'{op.name}:{cnt}' for op,cnt in top_ops)}"
    elif self.uops_list is not None:
      return f"{len(self.uops_list)} linear UOps"
    elif self.source_code is not None:
      lines = self.source_code.strip().split('\n')
      return f"{len(lines)} lines of code"
    return "empty"


@dataclass
class ExampleTrace:
  """Complete trace of an example through the pipeline."""
  name: str
  description: str
  code: str
  stages: list[IRStage] = field(default_factory=list)


# ============================================================================
# Core tracing functionality
# ============================================================================

def trace_tensor(tensor: Tensor, name: str, description: str, optimize: bool = True) -> ExampleTrace:
  """
  Trace a tensor through the full tinygrad compilation pipeline.

  Args:
    tensor: The Tensor to trace
    name: Name for this trace
    description: Human-readable description
    optimize: Whether to apply optimizations

  Returns:
    ExampleTrace with all pipeline stages captured
  """
  trace = ExampleTrace(name=name, description=description, code="")
  device = tensor.device

  # Stage 1: Lazy UOp Graph
  lazy_uop = tensor.uop
  trace.stages.append(IRStage(
    name="Lazy UOp Graph",
    description="The lazy computation graph built from tensor operations. "
                "Each operation creates UOps that form a DAG.",
    uop=lazy_uop
  ))

  # Stage 2: Schedule
  # Create a sink of all tensors to schedule
  big_sink = UOp.sink(lazy_uop)
  tensor_map, schedule, var_vals = complete_create_schedule_with_vars(big_sink)

  if not schedule:
    trace.stages.append(IRStage(
      name="Schedule",
      description="No kernels scheduled (tensor may already be realized)",
    ))
    return trace

  # Capture schedule info
  schedule_desc = f"Scheduled {len(schedule)} kernel(s). "
  for i, si in enumerate(schedule):
    schedule_desc += f"\n  Kernel {i}: {si.ast.op}"

  trace.stages.append(IRStage(
    name="Schedule",
    description=schedule_desc,
  ))

  # For each kernel in the schedule, trace through codegen
  for kernel_idx, si in enumerate(schedule):
    if si.ast.op not in {Ops.SINK, Ops.PROGRAM}:
      continue  # Skip non-kernel operations (copies, views)

    ast = si.ast
    renderer = Device[device].renderer

    # Stage 3: Kernel AST (before optimization)
    trace.stages.append(IRStage(
      name=f"Kernel {kernel_idx} - Base AST",
      description="The kernel's abstract syntax tree before optimization passes. "
                  "This is the SINK-rooted graph that represents the computation.",
      uop=ast
    ))

    # Stage 4: Optimized kernel (after full_rewrite_to_sink)
    try:
      # Tag the AST to skip optimization if requested
      opt_ast = ast if optimize else ast.rtag("noopt")
      optimized = full_rewrite_to_sink(opt_ast, renderer, optimize=optimize)
      trace.stages.append(IRStage(
        name=f"Kernel {kernel_idx} - Optimized",
        description="After optimization passes including: range simplification, "
                    "load collapse, symbolic simplification, expander, devectorizer, etc.",
        uop=optimized
      ))
    except Exception as e:
      trace.stages.append(IRStage(
        name=f"Kernel {kernel_idx} - Optimization Error",
        description=f"Error during optimization: {e}",
      ))
      continue

    # Stage 5: Generate program (linearize + render)
    try:
      prg: ProgramSpec = get_program(ast, renderer)

      # Capture linear IR
      if prg.uops is not None:
        trace.stages.append(IRStage(
          name=f"Kernel {kernel_idx} - Linear IR",
          description="The linearized IR ready for rendering. UOps are now in "
                      "execution order with control flow.",
          uops_list=prg.uops,
          estimates=prg.estimates
        ))

      # Capture source code
      trace.stages.append(IRStage(
        name=f"Kernel {kernel_idx} - Source Code",
        description=f"Generated {renderer.device} code for execution.",
        source_code=prg.src,
        estimates=prg.estimates
      ))

    except Exception as e:
      trace.stages.append(IRStage(
        name=f"Kernel {kernel_idx} - Codegen Error",
        description=f"Error during code generation: {e}",
      ))

  return trace


def print_trace(trace: ExampleTrace, verbose: int = 1) -> None:
  """Print a trace to stdout with formatting."""
  print(f"\n{'='*80}")
  print(colored(f"Example: {trace.name}", "cyan"))
  print(f"{'='*80}")
  print(f"{trace.description}\n")

  for stage in trace.stages:
    print(colored(f"\n--- {stage.name} ---", "yellow"))
    print(f"{stage.description}")
    print(f"Summary: {stage.summary()}")

    if verbose >= 2:
      if stage.uop is not None:
        print("\nUOp Graph (pyrender):")
        print(pyrender(stage.uop))

      if stage.uops_list is not None and verbose >= 3:
        print("\nLinear UOps:")
        print_uops(stage.uops_list)

      if stage.source_code is not None:
        print("\nSource Code:")
        print(stage.source_code)

    if stage.estimates is not None:
      est = stage.estimates
      print(f"Estimates: ops={est.ops}, lds={est.lds}, mem={est.mem}")


# ============================================================================
# Example definitions
# ============================================================================

@dataclass
class Example:
  """An example tensor operation to trace."""
  name: str
  description: str
  create: Callable[[], Tensor]


EXAMPLES = [
  Example(
    name="add",
    description="Element-wise addition of two tensors. "
                "The simplest fused operation.",
    create=lambda: Tensor([1, 2, 3]) + Tensor([4, 5, 6])
  ),

  Example(
    name="mul",
    description="Element-wise multiplication. "
                "Similar to add but uses MUL op.",
    create=lambda: Tensor([1, 2, 3]) * Tensor([4, 5, 6])
  ),

  Example(
    name="sum",
    description="Reduction operation (sum). "
                "Introduces REDUCE_AXIS and range loops.",
    create=lambda: Tensor([1, 2, 3, 4, 5]).sum()
  ),

  Example(
    name="matmul",
    description="Matrix multiplication (a @ b). "
                "Creates complex indexing patterns with contractions.",
    create=lambda: Tensor.randn(4, 8) @ Tensor.randn(8, 4)
  ),

  Example(
    name="fused",
    description="Fused operation: (a + b).sum(). "
                "Shows how operations are combined into single kernels.",
    create=lambda: (Tensor([1, 2, 3, 4]) + Tensor([5, 6, 7, 8])).sum()
  ),

  Example(
    name="softmax",
    description="Softmax operation (exp(x - max(x)) / sum(exp(x - max(x)))). "
                "Multiple reductions and element-wise ops.",
    create=lambda: Tensor([1.0, 2.0, 3.0, 4.0]).softmax()
  ),

  Example(
    name="conv2d",
    description="2D convolution with small inputs. "
                "Creates complex tiled access patterns.",
    create=lambda: Tensor.randn(1, 1, 8, 8).conv2d(Tensor.randn(1, 1, 3, 3))
  ),

  Example(
    name="broadcast",
    description="Broadcasting: vector + matrix. "
                "Shows EXPAND operations and shape handling.",
    create=lambda: Tensor([1, 2, 3, 4]) + Tensor.ones(3, 4)
  ),

  Example(
    name="contiguous",
    description="Transpose followed by contiguous copy. "
                "Shows when data movement is needed.",
    create=lambda: Tensor.randn(4, 8).T.contiguous()
  ),

  Example(
    name="multi_output",
    description="Multiple outputs from a single source. "
                "Shows kernel scheduling with dependencies.",
    create=lambda: (lambda x: x.sum() + x.max())(Tensor([1, 2, 3, 4, 5]))
  ),
]


# ============================================================================
# Diff utilities
# ============================================================================

def compute_uop_diff(before: UOp, after: UOp) -> dict:
  """
  Compute the difference between two UOp graphs.

  Returns a dict with:
    - added: ops that appear only in after
    - removed: ops that appear only in before
    - changed_counts: {op: (before_count, after_count)} for ops with different counts
  """
  before_ops = list(before.toposort())
  after_ops = list(after.toposort())

  def count_ops(ops):
    counts = {}
    for u in ops:
      counts[u.op] = counts.get(u.op, 0) + 1
    return counts

  before_counts = count_ops(before_ops)
  after_counts = count_ops(after_ops)

  all_ops = set(before_counts.keys()) | set(after_counts.keys())

  added = {op for op in all_ops if before_counts.get(op, 0) == 0}
  removed = {op for op in all_ops if after_counts.get(op, 0) == 0}
  changed = {op: (before_counts.get(op, 0), after_counts.get(op, 0))
             for op in all_ops
             if before_counts.get(op, 0) != after_counts.get(op, 0)
             and op not in added and op not in removed}

  return {
    "before_total": len(before_ops),
    "after_total": len(after_ops),
    "added": added,
    "removed": removed,
    "changed": changed
  }


def format_diff(diff: dict) -> str:
  """Format a diff dict as human-readable text."""
  lines = []
  lines.append(f"UOps: {diff['before_total']} -> {diff['after_total']} ({diff['after_total'] - diff['before_total']:+d})")

  if diff['added']:
    lines.append(f"  + Added: {', '.join(op.name for op in diff['added'])}")
  if diff['removed']:
    lines.append(f"  - Removed: {', '.join(op.name for op in diff['removed'])}")
  if diff['changed']:
    changes = [f"{op.name}: {b}->{a}" for op, (b, a) in diff['changed'].items()]
    lines.append(f"  ~ Changed: {', '.join(changes)}")

  return '\n'.join(lines)


# ============================================================================
# Markdown report generation
# ============================================================================

def generate_markdown_report(traces: list[ExampleTrace], output_path: str = "ir_report.md") -> None:
  """Generate a markdown report from traces."""
  lines = [
    "# IR Evolution Explorer Report",
    "",
    "This report shows how tensor operations evolve through tinygrad's compilation pipeline.",
    "",
    "## Pipeline Overview",
    "",
    "```mermaid",
    "flowchart TD",
    "    A[Tensor API] -->|lazy construction| B[UOp Graph]",
    "    B -->|schedule| C[Kernel UOps]",
    "    C -->|codegen passes| D[Optimized UOps]",
    "    D -->|linearize| E[Linear IR]",
    "    E -->|render| F[Device Code]",
    "```",
    "",
    "## Examples",
    "",
  ]

  for trace in traces:
    lines.append(f"### {trace.name}")
    lines.append("")
    lines.append(trace.description)
    lines.append("")

    prev_uop = None
    for stage in trace.stages:
      lines.append(f"#### {stage.name}")
      lines.append("")
      lines.append(stage.description)
      lines.append("")
      lines.append(f"**Summary:** {stage.summary()}")
      lines.append("")

      # Add diff from previous stage if both have UOps
      if stage.uop is not None and prev_uop is not None:
        diff = compute_uop_diff(prev_uop, stage.uop)
        lines.append("**Changes from previous stage:**")
        lines.append("```")
        lines.append(format_diff(diff))
        lines.append("```")
        lines.append("")

      if stage.uop is not None:
        prev_uop = stage.uop
        lines.append("<details>")
        lines.append("<summary>UOp Graph (click to expand)</summary>")
        lines.append("")
        lines.append("```python")
        lines.append(pyrender(stage.uop))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

      if stage.source_code is not None:
        # Detect language from device
        lang = "c"  # default
        if "METAL" in stage.name or "metal" in stage.source_code.lower():
          lang = "metal"
        elif "CUDA" in stage.name or "__global__" in stage.source_code:
          lang = "cuda"
        elif "PYTHON" in stage.name:
          lang = "python"

        lines.append("<details>")
        lines.append("<summary>Source Code (click to expand)</summary>")
        lines.append("")
        lines.append(f"```{lang}")
        lines.append(stage.source_code)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

      if stage.estimates is not None:
        est = stage.estimates
        lines.append(f"**Estimates:** ops={est.ops}, loads/stores={est.lds}, memory={est.mem}")
        lines.append("")

    lines.append("---")
    lines.append("")

  # Write report
  with open(output_path, 'w') as f:
    f.write('\n'.join(lines))

  print(f"\nReport written to: {output_path}")


# ============================================================================
# Main entry point
# ============================================================================

def main():
  parser = argparse.ArgumentParser(description="IR Evolution Explorer for tinygrad")
  parser.add_argument("--example", "-e", type=str, help="Run specific example by name")
  parser.add_argument("--list", "-l", action="store_true", help="List available examples")
  parser.add_argument("--report", "-r", action="store_true", help="Generate markdown report")
  parser.add_argument("--output", "-o", type=str, default="ir_report.md", help="Report output path")
  parser.add_argument("--device", "-d", type=str, default=None, help="Device to use (CPU, CUDA, etc)")
  parser.add_argument("--noopt", action="store_true", help="Disable optimizations")
  parser.add_argument("--verbose", "-v", type=int, default=None, help="Verbosity level (1-3)")
  args = parser.parse_args()

  # Set device
  if args.device:
    Device.DEFAULT = args.device

  # List examples
  if args.list:
    print("Available examples:")
    for ex in EXAMPLES:
      print(f"  {ex.name:15s} - {ex.description[:60]}...")
    return

  # Select examples to run
  if args.example:
    examples = [ex for ex in EXAMPLES if ex.name == args.example]
    if not examples:
      print(f"Unknown example: {args.example}")
      print("Use --list to see available examples")
      return
  else:
    examples = EXAMPLES

  # Determine verbosity
  verbose = args.verbose if args.verbose is not None else (DEBUG.value + 1)

  # Run traces
  optimize = not args.noopt
  traces = []

  print(colored(f"IR Evolution Explorer - Device: {Device.DEFAULT}", "green"))
  print(colored(f"Optimization: {'enabled' if optimize else 'DISABLED'}", "yellow" if not optimize else "green"))
  print()

  for ex in examples:
    try:
      print(colored(f"Tracing: {ex.name}...", "cyan"))
      tensor = ex.create()
      trace = trace_tensor(tensor, ex.name, ex.description, optimize=optimize)
      trace.code = f"# {ex.name}\n# {ex.description}"
      traces.append(trace)

      if not args.report:
        print_trace(trace, verbose=verbose)

    except Exception as e:
      print(colored(f"Error tracing {ex.name}: {e}", "red"))
      import traceback
      traceback.print_exc()

  # Generate report if requested
  if args.report:
    generate_markdown_report(traces, args.output)


if __name__ == "__main__":
  main()
