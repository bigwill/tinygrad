#!/usr/bin/env python3
"""
IR Explorer Web Interface for tinygrad

A Compiler Explorer-style web interface that shows tensor operations
alongside their IR transformations and generated code.

Usage:
  python3 ir_explorer_web.py [--port 8000]

Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations
import argparse
import html
import json
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import os
import sys

# Set up environment
os.environ.setdefault("DEBUG", "0")

from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, pyrender
from tinygrad.engine.schedule import complete_create_schedule_with_vars
from tinygrad.codegen import full_rewrite_to_sink, get_program
from tinygrad.renderer import ProgramSpec

# ============================================================================
# Device Detection
# ============================================================================

def get_available_devices() -> list[str]:
  """Return list of available devices on this system."""
  available = ["CPU", "PYTHON"]  # These are always available

  # Try each device
  test_devices = ["CUDA", "METAL", "AMD", "GPU"]
  for dev in test_devices:
    try:
      _ = Device[dev]
      available.append(dev)
    except Exception:
      pass

  return available

AVAILABLE_DEVICES = get_available_devices()

# ============================================================================
# Stage Explanations
# ============================================================================

STAGE_EXPLANATIONS = {
  "source": """Your Python code using the tinygrad Tensor API. Operations like +, -, @, .sum() 
create a lazy computation graph - nothing executes yet.""",

  "lazy": """The lazy UOp (Universal Operation) graph. Each tensor operation becomes a UOp node.
Key ops: BUFFER (data storage), COPY (device transfer), ADD/MUL/etc (math ops), 
RESHAPE/EXPAND/PERMUTE (view ops), REDUCE_AXIS (reductions like sum/max).""",

  "schedule": """The scheduler analyzes the lazy graph and decides:
1. Which operations to FUSE into single kernels (better performance)
2. Which need separate kernels (e.g., reductions often can't fuse)
3. The execution ORDER based on data dependencies
COPY ops move data between devices. SINK ops are compute kernels.""",

  "ast": """The kernel's Abstract Syntax Tree before optimization.
- DEFINE_GLOBAL: Input/output buffer pointers
- RANGE: Loop iteration (LOOP=sequential, REDUCE=accumulation)
- INDEX: Memory addressing (buffer[index])
- LOAD/STORE: Memory operations (added during optimization)
- Operations happen inside the loop structure""",

  "optimized": """After optimization passes:
1. Range simplification - simplify loop bounds
2. Load collapse - optimize memory access patterns  
3. Symbolic simplification - constant folding, algebraic simplification
4. Expander - unroll small loops for performance
5. Devectorizer - handle SIMD operations
The graph is now ready for code generation.""",

  "code": """Final generated code for the target device.
This runs on your CPU/GPU. The compiler will further optimize this."""
}

def get_stage_explanation(stage_name: str) -> str:
  """Get a detailed explanation for a pipeline stage."""
  name_lower = stage_name.lower()
  if "source" in name_lower:
    return STAGE_EXPLANATIONS["source"]
  elif "lazy" in name_lower:
    return STAGE_EXPLANATIONS["lazy"]
  elif "schedule" in name_lower:
    return STAGE_EXPLANATIONS["schedule"]
  elif "ast" in name_lower:
    return STAGE_EXPLANATIONS["ast"]
  elif "optimized" in name_lower or "optim" in name_lower:
    return STAGE_EXPLANATIONS["optimized"]
  elif "code" in name_lower:
    return STAGE_EXPLANATIONS["code"]
  return ""


def compute_diff(before_uop: UOp, after_uop: UOp) -> dict:
  """Compute detailed diff between two UOp graphs."""
  before_ops = list(before_uop.toposort())
  after_ops = list(after_uop.toposort())

  def count_ops(ops):
    counts = {}
    for u in ops:
      counts[u.op.name] = counts.get(u.op.name, 0) + 1
    return counts

  before_counts = count_ops(before_ops)
  after_counts = count_ops(after_ops)

  all_ops = set(before_counts.keys()) | set(after_counts.keys())

  added = {op: after_counts[op] for op in all_ops if before_counts.get(op, 0) == 0}
  removed = {op: before_counts[op] for op in all_ops if after_counts.get(op, 0) == 0}
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


def format_diff_text(diff: dict) -> str:
  """Format diff as human-readable text with explanations."""
  lines = []
  delta = diff['after_total'] - diff['before_total']
  lines.append(f"UOp count: {diff['before_total']} → {diff['after_total']} ({delta:+d})")

  if diff['added']:
    lines.append("")
    lines.append("✚ ADDED:")
    for op, count in diff['added'].items():
      explanation = get_op_explanation(op)
      lines.append(f"  • {op} (×{count}){': ' + explanation if explanation else ''}")

  if diff['removed']:
    lines.append("")
    lines.append("✖ REMOVED:")
    for op, count in diff['removed'].items():
      explanation = get_op_explanation(op)
      lines.append(f"  • {op} (×{count}){': ' + explanation if explanation else ''}")

  if diff['changed']:
    lines.append("")
    lines.append("△ CHANGED:")
    for op, (before, after) in diff['changed'].items():
      lines.append(f"  • {op}: {before} → {after}")

  return "\n".join(lines)


def get_op_explanation(op_name: str) -> str:
  """Get a brief explanation of what a UOp does."""
  explanations = {
    "LOAD": "reads from memory",
    "STORE": "writes to memory",
    "INDEX": "computes memory address",
    "RANGE": "loop iteration variable",
    "END": "marks end of loop scope",
    "REDUCE": "accumulates values (sum, max, etc.)",
    "DEFINE_GLOBAL": "declares buffer pointer",
    "DEFINE_LOCAL": "declares shared memory",
    "SPECIAL": "GPU thread/block index",
    "CONST": "constant value",
    "CAST": "type conversion",
    "ADD": "addition",
    "MUL": "multiplication", 
    "SINK": "root of kernel graph",
    "BUFFER": "data storage",
    "COPY": "device-to-device transfer",
    "RESHAPE": "changes logical shape (no data movement)",
    "EXPAND": "broadcasts to larger shape",
    "PERMUTE": "reorders dimensions",
    "REDUCE_AXIS": "reduction along axis",
    "CONTIGUOUS": "ensures contiguous memory layout",
  }
  return explanations.get(op_name, "")


def annotate_uop_code(code: str) -> str:
  """Add inline comments to pyrender output explaining key patterns."""
  annotations = [
    ("UOp.new_buffer", "# Create a new data buffer"),
    ("copy_to_device", "# Transfer data to compute device"),
    (".r(Ops.ADD", "# Reduction: sum along axis"),
    (".r(Ops.MUL", "# Reduction: product along axis"),
    (".r(Ops.MAX", "# Reduction: max along axis"),
    ("DEFINE_GLOBAL", "# Buffer pointer parameter"),
    ("UOp.range(", "# Loop from 0 to N"),
    (".index(", "# Compute memory address"),
    (".load()", "# Read from memory"),
    (".store(", "# Write to memory"),
    (".end(", "# End of loop scope"),
    (".sink()", "# Kernel graph root"),
    ("AxisType.LOOP", "# Sequential loop"),
    ("AxisType.REDUCE", "# Reduction loop (accumulates)"),
    ("AxisType.GLOBAL", "# GPU global thread"),
    ("AxisType.LOCAL", "# GPU local/shared thread"),
  ]

  lines = code.split('\n')
  result = []
  for line in lines:
    comment = ""
    for pattern, annotation in annotations:
      if pattern in line and "#" not in line:
        comment = "  " + annotation
        break
    result.append(line + comment)

  return '\n'.join(result)


# ============================================================================
# IR Tracing (reused from explore_ir.py)
# ============================================================================

def trace_code(code: str, device: str = "CPU") -> dict:
  """
  Execute Python code and trace the resulting tensor through the pipeline.

  Returns a dict with stages and any errors.
  """
  result = {
    "stages": [],
    "error": None,
    "device": device,
    "available_devices": AVAILABLE_DEVICES
  }

  # Check if device is available
  if device not in AVAILABLE_DEVICES:
    result["error"] = f"Device '{device}' is not available on this system.\nAvailable devices: {', '.join(AVAILABLE_DEVICES)}"
    return result

  # Set device
  old_default = Device.DEFAULT
  Device.DEFAULT = device

  try:
    # Create a restricted namespace for execution
    namespace = {
      "Tensor": Tensor,
      "dtypes": dtypes,
    }

    # Execute the code
    exec(code, namespace)

    # Find the result tensor (look for 'result' or last assigned tensor)
    tensor = None
    if "result" in namespace and isinstance(namespace["result"], Tensor):
      tensor = namespace["result"]
    else:
      # Find the last Tensor in namespace
      for name, val in namespace.items():
        if isinstance(val, Tensor) and not name.startswith("_"):
          tensor = val

    if tensor is None:
      result["error"] = "No Tensor found. Assign your result to 'result' or any variable."
      return result

    # Stage 1: Source code
    result["stages"].append({
      "name": "Source Code",
      "type": "python",
      "content": code,
      "description": get_stage_explanation("source")
    })

    # Stage 2: Lazy UOp Graph
    lazy_uop = tensor.uop
    lazy_code = annotate_uop_code(pyrender(lazy_uop))
    result["stages"].append({
      "name": "Lazy UOp Graph",
      "type": "python",
      "content": lazy_code,
      "description": get_stage_explanation("lazy"),
      "stats": {
        "total_uops": len(list(lazy_uop.toposort())),
        "op_counts": _count_ops(lazy_uop)
      }
    })

    # Stage 3: Schedule
    big_sink = UOp.sink(lazy_uop)
    tensor_map, schedule, var_vals = complete_create_schedule_with_vars(big_sink)

    schedule_info = []
    for i, si in enumerate(schedule):
      schedule_info.append(f"Kernel {i}: {si.ast.op.name}")
      if si.metadata:
        for m in si.metadata:
          schedule_info.append(f"    - {m.name}")

    result["stages"].append({
      "name": "Schedule",
      "type": "text",
      "content": f"Scheduled {len(schedule)} kernel(s):\n\n" + "\n".join(schedule_info),
      "description": get_stage_explanation("schedule")
    })

    # For each kernel, show the stages
    renderer = Device[device].renderer

    for kernel_idx, si in enumerate(schedule):
      if si.ast.op not in {Ops.SINK, Ops.PROGRAM}:
        continue

      ast = si.ast

      # Kernel AST
      ast_code = annotate_uop_code(pyrender(ast))
      result["stages"].append({
        "name": f"Kernel {kernel_idx} - AST",
        "type": "python",
        "content": ast_code,
        "description": get_stage_explanation("ast"),
        "stats": {
          "total_uops": len(list(ast.toposort())),
          "op_counts": _count_ops(ast)
        }
      })

      # Optimized
      try:
        optimized = full_rewrite_to_sink(ast, renderer, optimize=True)

        # Compute diff from AST to optimized
        diff = compute_diff(ast, optimized)
        diff_text = format_diff_text(diff)

        opt_code = annotate_uop_code(pyrender(optimized))
        result["stages"].append({
          "name": f"Kernel {kernel_idx} - Optimized",
          "type": "python",
          "content": opt_code,
          "description": get_stage_explanation("optimized"),
          "diff": diff_text,
          "stats": {
            "total_uops": len(list(optimized.toposort())),
            "op_counts": _count_ops(optimized)
          }
        })
      except Exception as e:
        result["stages"].append({
          "name": f"Kernel {kernel_idx} - Optimized",
          "type": "error",
          "content": f"Optimization error: {e}",
          "description": "Error during optimization"
        })
        continue

      # Generated code
      try:
        prg: ProgramSpec = get_program(ast, renderer)
        result["stages"].append({
          "name": f"Kernel {kernel_idx} - {device} Code",
          "type": _get_lang(device, prg.src),
          "content": prg.src,
          "description": get_stage_explanation("code"),
          "stats": {
            "lines": len(prg.src.strip().split("\n")),
            "est_ops": str(prg.estimates.ops),
            "est_mem": f"{prg.estimates.mem} bytes"
          }
        })
      except Exception as e:
        result["stages"].append({
          "name": f"Kernel {kernel_idx} - Code Generation",
          "type": "error",
          "content": f"Codegen error: {e}",
          "description": "Error during code generation"
        })

  except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

  finally:
    Device.DEFAULT = old_default

  return result


def _count_ops(uop: UOp) -> dict:
  """Count operations in a UOp graph."""
  counts = {}
  for u in uop.toposort():
    name = u.op.name
    counts[name] = counts.get(name, 0) + 1
  return dict(sorted(counts.items(), key=lambda x: -x[1])[:10])


def _get_lang(device: str, code: str) -> str:
  """Guess the language for syntax highlighting."""
  if "METAL" in device:
    return "cpp"  # Metal is close to C++
  elif "CUDA" in device or "__global__" in code:
    return "cpp"
  elif "PYTHON" in device:
    return "python"
  return "c"


# ============================================================================
# Web Server
# ============================================================================

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>tinygrad IR Explorer</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/c.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      background: #1e1e1e;
      color: #d4d4d4;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #252526;
      padding: 10px 20px;
      display: flex;
      align-items: center;
      gap: 20px;
      border-bottom: 1px solid #3c3c3c;
    }
    header h1 {
      font-size: 18px;
      font-weight: 600;
      color: #fff;
    }
    header h1 span { color: #569cd6; }
    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    select, button {
      background: #3c3c3c;
      border: 1px solid #555;
      color: #d4d4d4;
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
    }
    button:hover { background: #4c4c4c; }
    button.primary {
      background: #0e639c;
      border-color: #0e639c;
    }
    button.primary:hover { background: #1177bb; }
    .main {
      flex: 1;
      display: flex;
      overflow: hidden;
    }
    .editor-pane {
      width: 40%;
      min-width: 200px;
      max-width: 80%;
      display: flex;
      flex-direction: column;
    }
    .editor-header {
      background: #2d2d2d;
      padding: 8px 15px;
      font-size: 13px;
      color: #888;
      border-bottom: 1px solid #3c3c3c;
    }
    #editor {
      flex: 1;
      width: 100%;
      background: #1e1e1e;
      color: #d4d4d4;
      border: none;
      padding: 15px;
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
      font-size: 14px;
      line-height: 1.5;
      resize: none;
      outline: none;
    }
    /* Resizer handle */
    .resizer {
      width: 6px;
      background: #3c3c3c;
      cursor: col-resize;
      flex-shrink: 0;
      position: relative;
      transition: background 0.15s;
    }
    .resizer:hover, .resizer.dragging {
      background: #569cd6;
    }
    .resizer::after {
      content: "⋮";
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: #888;
      font-size: 14px;
      pointer-events: none;
    }
    .resizer:hover::after, .resizer.dragging::after {
      color: #fff;
    }
    .output-pane {
      flex: 1;
      display: flex;
      overflow: hidden;
      min-width: 200px;
    }
    /* Vertical sidebar for stages */
    .stage-sidebar {
      width: 180px;
      min-width: 140px;
      background: #252526;
      border-right: 1px solid #3c3c3c;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .stage-sidebar-header {
      padding: 10px 12px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #888;
      border-bottom: 1px solid #3c3c3c;
      flex-shrink: 0;
    }
    .stage-list {
      flex: 1;
      overflow-y: auto;
    }
    .stage-item {
      padding: 8px 12px;
      cursor: pointer;
      font-size: 12px;
      border-left: 3px solid transparent;
      color: #888;
      border-bottom: 1px solid #2d2d2d;
    }
    .stage-item:hover {
      background: #2d2d2d;
      color: #d4d4d4;
    }
    .stage-item.active {
      background: #37373d;
      color: #fff;
      border-left-color: #569cd6;
    }
    .stage-item .stage-type {
      font-size: 10px;
      color: #666;
      margin-top: 2px;
    }
    .stage-item.active .stage-type {
      color: #888;
    }
    .stage-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .stage {
      display: none;
      height: 100%;
      flex-direction: column;
    }
    .stage.active { display: flex; }
    .stage-header {
      background: #2d2d2d;
      padding: 10px 15px;
      border-bottom: 1px solid #3c3c3c;
      flex-shrink: 0;
    }
    .stage-header h3 {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .stage-desc {
      font-size: 12px;
      color: #aaa;
      line-height: 1.5;
      white-space: pre-wrap;
      margin-bottom: 10px;
      padding: 8px 10px;
      background: #252526;
      border-radius: 4px;
      border-left: 3px solid #569cd6;
    }
    .stage-diff {
      font-size: 11px;
      color: #d4d4d4;
      margin: 10px 0;
      padding: 10px;
      background: #1a1a2e;
      border-radius: 4px;
      border: 1px solid #333;
    }
    .stage-diff pre {
      margin: 0;
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
      white-space: pre-wrap;
    }
    .stage-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
      font-size: 11px;
    }
    .stage-stats span {
      background: #3c3c3c;
      padding: 3px 8px;
      border-radius: 3px;
    }
    .stage-code {
      flex: 1;
      overflow: auto;
      margin: 0;
    }
    .stage-code pre {
      margin: 0;
      padding: 15px;
      font-size: 13px;
      line-height: 1.5;
    }
    .stage-code code {
      font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    .error {
      background: #5a1d1d;
      color: #f48771;
      padding: 15px;
      font-family: monospace;
      white-space: pre-wrap;
    }
    .loading {
      display: none;
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: #333;
      padding: 20px 30px;
      border-radius: 8px;
      z-index: 100;
    }
    .loading.show { display: block; }
    .examples {
      font-size: 13px;
    }
    /* Prevent text selection while dragging */
    body.resizing {
      user-select: none;
      cursor: col-resize;
    }
    @media (max-width: 900px) {
      .main { flex-direction: column; }
      .editor-pane { width: 100% !important; height: 35%; min-width: auto; max-width: none; }
      .resizer { display: none; }
      .output-pane { flex-direction: column; }
      .stage-sidebar { width: 100%; min-width: auto; height: auto; max-height: 120px; border-right: none; border-bottom: 1px solid #3c3c3c; }
      .stage-list { display: flex; flex-wrap: wrap; overflow-x: auto; }
      .stage-item { border-left: none; border-bottom: 2px solid transparent; }
      .stage-item.active { border-bottom-color: #569cd6; border-left-color: transparent; }
    }
  </style>
</head>
<body>
  <header>
    <h1><span>tinygrad</span> IR Explorer</h1>
    <div class="controls">
      <select id="device">
        <!-- Populated dynamically based on available devices -->
      </select>
      <select id="examples" class="examples">
        <option value="">Load Example...</option>
        <option value="add">Add: a + b</option>
        <option value="sum">Sum: x.sum()</option>
        <option value="matmul">Matmul: a @ b</option>
        <option value="fused">Fused: (a + b).sum()</option>
        <option value="softmax">Softmax</option>
        <option value="conv2d">Conv2D</option>
        <option value="broadcast">Broadcasting</option>
      </select>
      <button class="primary" onclick="compile()">Compile (Ctrl+Enter)</button>
    </div>
  </header>

  <div class="main">
    <div class="editor-pane" id="editorPane">
      <div class="editor-header">Python (tinygrad)</div>
      <textarea id="editor" spellcheck="false"># Simple element-wise addition
a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])
result = a + b</textarea>
    </div>

    <div class="resizer" id="resizer"></div>

    <div class="output-pane">
      <div class="stage-sidebar">
        <div class="stage-sidebar-header">Pipeline Stages</div>
        <div class="stage-list" id="stageList"></div>
      </div>
      <div class="stage-content" id="stageContent"></div>
    </div>
  </div>

  <div class="loading" id="loading">Compiling...</div>

  <script>
    const EXAMPLES = {
      add: `# Element-wise addition
a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])
result = a + b`,

      sum: `# Reduction (sum)
x = Tensor([1, 2, 3, 4, 5, 6, 7, 8])
result = x.sum()`,

      matmul: `# Matrix multiplication
a = Tensor.randn(4, 8)
b = Tensor.randn(8, 4)
result = a @ b`,

      fused: `# Fused operations
a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])
result = (a + b).sum()`,

      softmax: `# Softmax
x = Tensor([1.0, 2.0, 3.0, 4.0])
result = x.softmax()`,

      conv2d: `# 2D Convolution
x = Tensor.randn(1, 1, 8, 8)
w = Tensor.randn(1, 1, 3, 3)
result = x.conv2d(w)`,

      broadcast: `# Broadcasting
vec = Tensor([1, 2, 3, 4])
mat = Tensor.ones(3, 4)
result = vec + mat`
    };

    document.getElementById('examples').onchange = function() {
      if (this.value && EXAMPLES[this.value]) {
        document.getElementById('editor').value = EXAMPLES[this.value];
        this.value = '';
        compile();
      }
    };

    document.getElementById('editor').addEventListener('keydown', function(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        compile();
      }
      // Tab support
      if (e.key === 'Tab') {
        e.preventDefault();
        const start = this.selectionStart;
        const end = this.selectionEnd;
        this.value = this.value.substring(0, start) + '  ' + this.value.substring(end);
        this.selectionStart = this.selectionEnd = start + 2;
      }
    });

    // Resizer logic
    const resizer = document.getElementById('resizer');
    const editorPane = document.getElementById('editorPane');
    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
      isResizing = true;
      resizer.classList.add('dragging');
      document.body.classList.add('resizing');
    });

    document.addEventListener('mousemove', (e) => {
      if (!isResizing) return;
      const containerWidth = document.querySelector('.main').offsetWidth;
      const newWidth = e.clientX;
      const percent = (newWidth / containerWidth) * 100;
      if (percent > 15 && percent < 85) {
        editorPane.style.width = percent + '%';
      }
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        resizer.classList.remove('dragging');
        document.body.classList.remove('resizing');
      }
    });

    let currentTab = 0;

    function compile() {
      const code = document.getElementById('editor').value;
      const device = document.getElementById('device').value;
      const loading = document.getElementById('loading');

      loading.classList.add('show');

      fetch('/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, device })
      })
      .then(r => r.json())
      .then(data => {
        loading.classList.remove('show');
        renderStages(data);
      })
      .catch(err => {
        loading.classList.remove('show');
        renderStages({ error: err.toString(), stages: [] });
      });
    }

    function renderStages(data) {
      const stageList = document.getElementById('stageList');
      const stageContent = document.getElementById('stageContent');

      stageList.innerHTML = '';
      stageContent.innerHTML = '';

      // Update device dropdown if we got available_devices
      if (data.available_devices) {
        const select = document.getElementById('device');
        const currentValue = select.value;
        select.innerHTML = '';
        data.available_devices.forEach(dev => {
          const opt = document.createElement('option');
          opt.value = dev;
          opt.textContent = dev;
          if (dev === currentValue) opt.selected = true;
          select.appendChild(opt);
        });
      }

      if (data.error && data.stages.length === 0) {
        stageContent.innerHTML = `<div class="error">${escapeHtml(data.error)}</div>`;
        return;
      }

      // Clamp currentTab to valid range
      if (currentTab >= data.stages.length) currentTab = 0;

      data.stages.forEach((stage, i) => {
        // Sidebar item
        const item = document.createElement('div');
        item.className = 'stage-item' + (i === currentTab ? ' active' : '');
        item.innerHTML = `
          <div>${escapeHtml(stage.name)}</div>
          <div class="stage-type">${escapeHtml(stage.type || 'code')}</div>
        `;
        item.onclick = () => switchTab(i);
        stageList.appendChild(item);

        // Content
        const div = document.createElement('div');
        div.className = 'stage' + (i === currentTab ? ' active' : '');
        div.id = 'stage-' + i;

        let statsHtml = '';
        if (stage.stats) {
          statsHtml = '<div class="stage-stats">';
          for (const [k, v] of Object.entries(stage.stats)) {
            if (typeof v === 'object') {
              statsHtml += `<span>${k}: ${Object.entries(v).slice(0,5).map(([a,b])=>a+':'+b).join(', ')}</span>`;
            } else {
              statsHtml += `<span>${k}: ${v}</span>`;
            }
          }
          statsHtml += '</div>';
        }

        let diffHtml = '';
        if (stage.diff) {
          diffHtml = `<div class="stage-diff"><pre>${escapeHtml(stage.diff)}</pre></div>`;
        }

        div.innerHTML = `
          <div class="stage-header">
            <h3>${escapeHtml(stage.name)}</h3>
            <p class="stage-desc">${escapeHtml(stage.description || '')}</p>
            ${diffHtml}
            ${statsHtml}
          </div>
          <div class="stage-code">
            <pre><code class="language-${stage.type || 'python'}">${escapeHtml(stage.content)}</code></pre>
          </div>
        `;
        stageContent.appendChild(div);
      });

      // Highlight all code blocks
      document.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
      });

      // Show error if present but we have stages
      if (data.error) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = data.error;
        stageContent.insertBefore(errorDiv, stageContent.firstChild);
      }
    }

    function switchTab(i) {
      currentTab = i;
      document.querySelectorAll('.stage-item').forEach((t, j) => t.classList.toggle('active', i === j));
      document.querySelectorAll('.stage').forEach((s, j) => s.classList.toggle('active', i === j));
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    // Populate devices and do initial compile
    function init() {
      // First, get available devices
      fetch('/devices')
        .then(r => r.json())
        .then(data => {
          const select = document.getElementById('device');
          select.innerHTML = '';
          data.devices.forEach(dev => {
            const opt = document.createElement('option');
            opt.value = dev;
            opt.textContent = dev;
            select.appendChild(opt);
          });
          // Now compile with default device
          compile();
        })
        .catch(() => {
          // Fallback if /devices fails
          const select = document.getElementById('device');
          select.innerHTML = '<option value="CPU">CPU</option>';
          compile();
        });
    }

    init();
  </script>
</body>
</html>
'''


class IRExplorerHandler(BaseHTTPRequestHandler):
  def log_message(self, format, *args):
    # Quieter logging
    pass

  def do_GET(self):
    if self.path == "/" or self.path == "/index.html":
      self.send_response(200)
      self.send_header("Content-Type", "text/html")
      self.end_headers()
      self.wfile.write(HTML_TEMPLATE.encode())
    elif self.path == "/devices":
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(json.dumps({"devices": AVAILABLE_DEVICES}).encode())
    else:
      self.send_error(404)

  def do_POST(self):
    if self.path == "/compile":
      content_length = int(self.headers.get("Content-Length", 0))
      body = self.rfile.read(content_length).decode()

      try:
        data = json.loads(body)
        code = data.get("code", "")
        device = data.get("device", "CPU")

        result = trace_code(code, device)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

      except Exception as e:
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": str(e), "stages": []}).encode())
    else:
      self.send_error(404)


def main():
  parser = argparse.ArgumentParser(description="tinygrad IR Explorer Web Interface")
  parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on")
  parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
  args = parser.parse_args()

  server = HTTPServer((args.host, args.port), IRExplorerHandler)
  print(f"\n  tinygrad IR Explorer")
  print(f"  ====================")
  print(f"  Open http://{args.host}:{args.port} in your browser")
  print(f"  Press Ctrl+C to stop\n")

  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nShutting down...")
    server.shutdown()


if __name__ == "__main__":
  main()
