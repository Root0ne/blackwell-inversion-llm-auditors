"""Zero-cost deterministic client for `--dry-run`.

It plays a plausible agent: it calls the real oracle and runs real Python in the
sandbox (a Kikuchi-2 spectral test in Experiment 1, a correlation learner or an
artifact reader in Experiment 2) and answers from the tool output. Its choices depend
only on the prompt text, never on the hidden ground truth, so a dry run exercises the
whole pipeline (tool schemas, sandbox, oracle, parsers, records) at no cost.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from .base import COST_KEY, Message, ModelClient, ToolSchema

# Prompt substrings the mock uses to tell the experiments and access modes apart.
EXP2_PROMPT_MARKERS = ("3-variable majority", "majority rule")
CODE_ARTIFACT = "artifact.py"
WEIGHTS_ARTIFACT = "weights.npz"
RULE_JSON_PATTERN = r'\{"support":\s*\[[^\]]+\],\s*"signs":\s*\[[^\]]+\]\}'
DEFAULT_RULE_ANSWER = '{"support": [0, 1, 2], "signs": [1, 1, 1]}'


class MockDryRunClient(ModelClient):
    """Deterministic stand-in for a model; it never spends money and never checks the cap.

    Experiment 2 is recognised by the majority-rule wording, so the other Experiment 2
    task types (dlist3, dtree2) are answered as if they were Experiment 1 and fail to parse.
    """

    def __init__(self) -> None:
        super().__init__(max_cost_usd=0.0)

    @staticmethod
    def _turn_count(messages: List[Message]) -> int:
        return sum(1 for m in messages if m.get("role") == "assistant")

    @staticmethod
    def _has_tool(tools: Optional[List[ToolSchema]], name: str) -> bool:
        return bool(tools and any(t.get("name") == name for t in tools))

    @staticmethod
    def _extract_last_tool_output(messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for blk in m["content"]:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        raw = str(blk.get("content", ""))
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict) and "stdout" in parsed:
                                return str(parsed["stdout"])
                        except json.JSONDecodeError:
                            pass
                        return raw
        return ""

    def create_message(
        self,
        *,
        model: str,
        system: str,
        messages: List[Message],
        max_tokens: int,
        tools: Optional[List[ToolSchema]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        thinking_budget: int = 0,
        thinking_mode: str = "adaptive",
        effort: str = "",
    ) -> Dict[str, Any]:
        # The Anthropic Messages API requires strictly alternating user/assistant roles.
        for i, m in enumerate(messages):
            expected_role = "user" if (i % 2 == 0) else "assistant"
            if m.get("role") != expected_role:
                raise AssertionError(
                    f"Anthropic role-alternation violation at index {i}: got {m.get('role')!r}, expected {expected_role!r}"
                )
        turn = self._turn_count(messages)
        first_prompt = str(messages[0]["content"]) if messages else ""
        has_py = self._has_tool(tools, "run_python")
        is_exp2 = any(marker in first_prompt for marker in EXP2_PROMPT_MARKERS)
        has_wb_code = CODE_ARTIFACT in first_prompt
        has_wb_relu = WEIGHTS_ARTIFACT in first_prompt

        # Pseudo-random bits derived from the prompt text only, never from the ground truth.
        h_int = int.from_bytes(hashlib.sha256(first_prompt.encode("utf-8")).digest()[:4], "little")

        if is_exp2:
            return self._mock_exp2(turn, has_py, has_wb_code, has_wb_relu, h_int, messages)
        return self._mock_exp1(turn, has_py, has_wb_code, has_wb_relu, model, h_int, messages)

    def _mock_exp1(
        self,
        turn: int,
        has_py: bool,
        has_wb_code: bool,
        has_wb_relu: bool,
        model: str,
        h_int: int,
        messages: List[Message],
    ) -> Dict[str, Any]:
        if turn == 0 and has_py and (has_wb_code or has_wb_relu):
            # White box with Python: run a Kikuchi-2 spectral test on the artifact in the sandbox.
            if has_wb_relu:
                code = (
                    "import numpy as np\n"
                    "from scipy.sparse import coo_matrix\n"
                    "from scipy.sparse.linalg import eigsh\n"
                    "d = np.load('weights.npz')\n"
                    "W1 = 0.5 * (d['W1'] @ d['W0']) if 'W0' in d else d['W1']\n"
                    "b1, w2, n = d['b1'], d['w2'], int(d['n'])\n"
                    "groups = {}\n"
                    "for j in range(W1.shape[0]):\n"
                    "    row = W1[j]; sc = np.max(np.abs(row))\n"
                    "    if sc < 1e-12: continue\n"
                    "    supp = tuple(int(v) for v in np.where(np.abs(row) > 0.5*sc)[0])\n"
                    "    if len(supp) != 4: continue\n"
                    "    sg = tuple(1 if row[i] > 0 else -1 for i in supp)\n"
                    "    groups.setdefault((supp, sg), []).append((b1[j]/sc, w2[j]*sc))\n"
                    "N = n*(n-1)//2\n"
                    "pid = lambda u, v: u*(2*n-u-1)//2 + (v-u-1)\n"
                    "def _score(ys_arr):\n"
                    "    rows, cols, vals = [], [], []\n"
                    "    for (a,b,c,e), yt in zip(cl_list, ys_arr):\n"
                    "        for p1, p2 in [((a,b),(c,e)),((a,c),(b,e)),((a,e),(b,c))]:\n"
                    "            i, j = pid(*p1), pid(*p2)\n"
                    "            rows.extend([i,j]); cols.extend([j,i]); vals.extend([yt,yt])\n"
                    "    M = coo_matrix((vals,(rows,cols)), shape=(N,N)).tocsr()\n"
                    "    deg = np.array(np.abs(M).sum(axis=1)).ravel()\n"
                    "    inv = 1.0/np.sqrt(deg + max(float(np.mean(deg)), 1.0))\n"
                    "    return float(eigsh(M.multiply(inv[:,None]).multiply(inv[None,:]), k=1, which='LA', return_eigenvectors=False)[0])\n"
                    "cl_list, ys_list = [], []\n"
                    "for (supp, sg), items in groups.items():\n"
                    "    items.sort(key=lambda t: t[0], reverse=True)\n"
                    "    yp = -1.0 if items[0][1] > 0 else 1.0\n"
                    "    cl_list.append(supp); ys_list.append(yp * sg[0]*sg[1]*sg[2]*sg[3])\n"
                    "ev_real = _score(ys_list)\n"
                    "rng = np.random.default_rng(7)\n"
                    "ev_null = float(np.mean([_score(rng.choice([-1.0, 1.0], size=len(ys_list))) for _ in range(3)]))\n"
                    "print(f'KIKUCHI_DIFF={ev_real - ev_null:+.6f} M={len(groups)}')\n"
                )
            else:
                code = (
                    "import artifact, numpy as np\n"
                    "from scipy.sparse import coo_matrix\n"
                    "from scipy.sparse.linalg import eigsh\n"
                    "if hasattr(artifact, 'CLAUSES'):\n"
                    "    cl, ys, n = artifact.CLAUSES, [float(v) for v in artifact.SIGNS], artifact.N\n"
                    "else:\n"
                    "    cl, sg, yp, n = artifact._T, artifact._S, artifact._Y, artifact._N\n"
                    "    ys = [float(y * (s[0]*s[1]*s[2]*s[3])) for s, y in zip(sg, yp)]\n"
                    "N = n*(n-1)//2\n"
                    "pid = lambda u, v: u*(2*n-u-1)//2 + (v-u-1)\n"
                    "def _score(ys_arr):\n"
                    "    rows, cols, vals = [], [], []\n"
                    "    for (a,b,c,e), val in zip(cl, ys_arr):\n"
                    "        for p1, p2 in [((a,b),(c,e)),((a,c),(b,e)),((a,e),(b,c))]:\n"
                    "            i, j = pid(*p1), pid(*p2)\n"
                    "            rows.extend([i,j]); cols.extend([j,i]); vals.extend([val,val])\n"
                    "    M = coo_matrix((vals,(rows,cols)), shape=(N,N)).tocsr()\n"
                    "    deg = np.array(np.abs(M).sum(axis=1)).ravel()\n"
                    "    inv = 1.0/np.sqrt(deg + max(float(np.mean(deg)), 1.0))\n"
                    "    return float(eigsh(M.multiply(inv[:,None]).multiply(inv[None,:]), k=1, which='LA', return_eigenvectors=False)[0])\n"
                    "ev_real = _score(ys)\n"
                    "rng = np.random.default_rng(7)\n"
                    "ev_null = float(np.mean([_score(rng.choice([-1.0, 1.0], size=len(ys))) for _ in range(3)]))\n"
                    "print(f'KIKUCHI_DIFF={ev_real - ev_null:+.6f} M={len(cl)}')\n"
                )
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mock_py_1",
                        "name": "run_python",
                        "input": {"code": code},
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 60},
                COST_KEY: 0.0,
            }

        if turn == 0 and not (has_wb_code or has_wb_relu):
            # Black box: spend part of the query budget on random inputs.
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mock_ora_1",
                        "name": "query_oracle_random",
                        "input": {"count": 20, "seed": int(h_int % 10000)},
                    }
                ],
                "usage": {"input_tokens": 80, "output_tokens": 25},
                COST_KEY: 0.0,
            }

        # Decide from the spectral excess over a sign-permutation null printed by the sandbox.
        last_out = self._extract_last_tool_output(messages)
        m_ev = re.search(r"KIKUCHI_DIFF=([-+0-9.]+)\s+M=(\d+)", last_out)
        if m_ev:
            diff = float(m_ev.group(1))
            # Mock models whose id contains "small" guess on ReLU weights, to exercise the H3 analysis.
            if "small" in model.lower() and has_wb_relu:
                dec = "PLANTED" if (h_int % 2 == 0) else "CLEAN"
            else:
                dec = "PLANTED" if diff >= 0.012 else "CLEAN"
        else:
            dec = "PLANTED" if (h_int % 2 == 0) else "CLEAN"

        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"decision": dec, "confidence": 0.75, "rationale": "dry-run"}),
                }
            ],
            "usage": {"input_tokens": 60, "output_tokens": 20},
            COST_KEY: 0.0,
        }

    def _mock_exp2(
        self,
        turn: int,
        has_py: bool,
        has_wb_code: bool,
        has_wb_relu: bool,
        h_int: int,
        messages: List[Message],
    ) -> Dict[str, Any]:
        # Black box: query 80 random inputs, then fit a correlation learner in the sandbox.
        if not (has_wb_code or has_wb_relu):
            if turn == 0:
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "mock_bb_q",
                            "name": "query_oracle_random",
                            "input": {"count": 80, "seed": int(h_int % 10000)},
                        }
                    ],
                    "usage": {"input_tokens": 80, "output_tokens": 20},
                    COST_KEY: 0.0,
                }
            if turn == 1 and has_py:
                code = (
                    "import json, numpy as np\n"
                    "d = json.load(open('queries.json'))\n"
                    "X = np.array([r['x'] for r in d['records']], dtype=np.float64)\n"
                    "y = np.array([2*r['y'] - 1 for r in d['records']], dtype=np.float64)\n"
                    "c = (X.T @ y) / len(y)\n"
                    "top = sorted(int(i) for i in np.argsort(-np.abs(c))[:3])\n"
                    "sg = [1 if c[i] >= 0 else -1 for i in top]\n"
                    "print(json.dumps({'support': top, 'signs': sg}))\n"
                )
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "mock_bb_py",
                            "name": "run_python",
                            "input": {"code": code},
                        }
                    ],
                    "usage": {"input_tokens": 90, "output_tokens": 45},
                    COST_KEY: 0.0,
                }
            last_out = self._extract_last_tool_output(messages)
            m_js = re.search(RULE_JSON_PATTERN, last_out)
            ans = m_js.group(0) if m_js else DEFAULT_RULE_ANSWER
            return {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": ans}],
                "usage": {"input_tokens": 50, "output_tokens": 20},
                COST_KEY: 0.0,
            }

        # White box: read the artifact in the sandbox.
        if turn == 0 and has_py:
            if has_wb_code:
                # Takes the rule from the readable decoy branch if present, else from the score line.
                code = (
                    "import json, re\n"
                    "src = open('artifact.py').read()\n"
                    "m = re.search(r'def _branch_a\\(x\\):\\s*return 1 if \\((.+)\\) > 0', src)\n"
                    "if not m:\n"
                    "    m = re.search(r'score = ([^\\n]+)', src)\n"
                    "expr = m.group(1) if m else ''\n"
                    "pairs = [(int(idx), int(sgn)) for sgn, idx in re.findall(r'\\(([+-]1)\\) \\* x\\[(\\d+)\\]', expr)]\n"
                    "pairs.sort(key=lambda t: t[0])\n"
                    "print(json.dumps({'support': [p[0] for p in pairs], 'signs': [p[1] for p in pairs]}))\n"
                )
            else:
                # Takes the rule from the largest-norm row of W1, which is the decoy.
                code = (
                    "import json, numpy as np\n"
                    "d = np.load('weights.npz')\n"
                    "W1 = d['W1']\n"
                    "norms = np.linalg.norm(W1, axis=1)\n"
                    "row = W1[int(np.argmax(norms))]\n"
                    "top = sorted(int(i) for i in np.argsort(-np.abs(row))[:3])\n"
                    "sg = [1 if row[i] >= 0 else -1 for i in top]\n"
                    "print(json.dumps({'support': top, 'signs': sg}))\n"
                )
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "mock_wb_py",
                        "name": "run_python",
                        "input": {"code": code},
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 55},
                COST_KEY: 0.0,
            }
        last_out = self._extract_last_tool_output(messages)
        m_js = re.search(RULE_JSON_PATTERN, last_out)
        ans = m_js.group(0) if m_js else DEFAULT_RULE_ANSWER
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": ans}],
            "usage": {"input_tokens": 60, "output_tokens": 20},
            COST_KEY: 0.0,
        }
