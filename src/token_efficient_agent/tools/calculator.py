from langchain_core.tools import tool

VERBOSE_DESCRIPTION = (
    "Perform mathematical calculations and unit conversions. Accepts a mathematical "
    "expression as a string and evaluates it. Supports basic arithmetic (+, -, *, /), "
    "exponentiation (**), modulo (%), parentheses for grouping, and common math functions "
    "(sqrt, abs, round, min, max, sum). Also supports percentage calculations.\n\n"
    "For financial calculations, values are rounded to 2 decimal places by default. "
    "Use this tool whenever you need to compute derived metrics like growth rates, "
    "averages, ratios, or aggregate statistics from data you've gathered.\n\n"
    "Examples:\n"
    "  calculator(expression='(2300000 - 2000000) / 2000000 * 100')  -> 15.0 (% growth)\n"
    "  calculator(expression='sum([4250000, 3890000, 3420000]) / 3')  -> avg revenue\n"
    "  calculator(expression='sqrt(144) + 2**3')  -> 20.0\n\n"
    "Note: Does not support symbolic math, calculus, or matrix operations."
)

CALCULATOR_PRUNED = "Evaluate a mathematical expression. Supports arithmetic, percentages, and common math functions."

CALCULATOR_ROUTING = "Math calculator."


@tool(description=VERBOSE_DESCRIPTION)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    import ast
    import math
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }
    safe_funcs = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in safe_funcs:
            args = [_eval(a) for a in node.args]
            return safe_funcs[node.func.id](*args)
        if isinstance(node, ast.List):
            return [_eval(el) for el in node.elts]
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        if isinstance(result, float):
            result = round(result, 4)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


calculator_tool = calculator
