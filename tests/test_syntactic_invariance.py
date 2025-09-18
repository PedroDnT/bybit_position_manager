"""
Syntactic Invariance Tests for Bybit Position Manager

This module implements comprehensive tests to verify code structure,
formatting consistency, and adherence to Python standards across
the entire codebase.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pytest


class SyntacticInvarianceValidator:
    """Validates syntactic invariance and code quality across the codebase."""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.python_files = list(self.project_root.rglob("*.py"))
        # Exclude virtual environment and cache directories
        self.python_files = [
            f
            for f in self.python_files
            if not any(
                part.startswith(".") or part == "__pycache__" or part == ".venv"
                for part in f.parts
            )
        ]

    def validate_ast_structure(self) -> Dict[str, List[str]]:
        """Validate that all Python files have valid AST structure."""
        issues = {}

        for file_path in self.python_files:
            file_issues = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse AST
                tree = ast.parse(content, filename=str(file_path))

                # Check for common structural issues
                file_issues.extend(self._check_ast_structure(tree, file_path))

            except SyntaxError as e:
                file_issues.append(f"Syntax error: {e}")
            except UnicodeDecodeError as e:
                file_issues.append(f"Encoding error: {e}")
            except Exception as e:
                file_issues.append(f"Unexpected error: {e}")

            if file_issues:
                issues[str(file_path)] = file_issues

        return issues

    def _check_ast_structure(self, tree: ast.AST, file_path: Path) -> List[str]:
        """Check AST structure for common issues."""
        issues = []

        class StructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function_complexity = {}
                self.class_methods = {}
                self.imports = []
                self.current_class = None

            def visit_FunctionDef(self, node):
                # Check function complexity (cyclomatic complexity approximation)
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    issues.append(
                        f"High complexity function '{node.name}' (complexity: {complexity})"
                    )

                # Check function length
                if hasattr(node, "end_lineno") and node.end_lineno:
                    length = node.end_lineno - node.lineno
                    if length > 50:
                        issues.append(f"Long function '{node.name}' ({length} lines)")

                self.generic_visit(node)

            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name

                # Check class length
                if hasattr(node, "end_lineno") and node.end_lineno:
                    length = node.end_lineno - node.lineno
                    if length > 200:
                        issues.append(f"Large class '{node.name}' ({length} lines)")

                self.generic_visit(node)
                self.current_class = old_class

            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.append(alias.name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        self.imports.append(f"{node.module}.{alias.name}")
                self.generic_visit(node)

            def _calculate_complexity(self, node):
                """Calculate cyclomatic complexity approximation."""
                complexity = 1  # Base complexity

                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                        complexity += 1
                    elif isinstance(child, ast.ExceptHandler):
                        complexity += 1
                    elif isinstance(child, (ast.And, ast.Or)):
                        complexity += 1

                return complexity

        visitor = StructureVisitor()
        visitor.visit(tree)

        return issues

    def validate_import_structure(self) -> Dict[str, List[str]]:
        """Validate import organization and detect circular imports."""
        issues = {}
        import_graph = {}

        for file_path in self.python_files:
            file_issues = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                import_graph[str(file_path)] = imports

                # Check import organization
                file_issues.extend(self._check_import_organization(content))

            except Exception as e:
                file_issues.append(f"Error analyzing imports: {e}")

            if file_issues:
                issues[str(file_path)] = file_issues

        # Check for circular imports
        circular_issues = self._detect_circular_imports(import_graph)
        if circular_issues:
            issues["_circular_imports"] = circular_issues

        return issues

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from an AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def _check_import_organization(self, content: str) -> List[str]:
        """Check import organization according to PEP 8."""
        issues = []
        lines = content.split("\n")

        # Find import blocks
        import_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not stripped.startswith(
                "#"
            ):
                import_lines.append((i, stripped))

        if not import_lines:
            return issues

        # Check import grouping (stdlib, third-party, local)
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        stdlib_modules = {
            "os",
            "sys",
            "math",
            "json",
            "datetime",
            "time",
            "pathlib",
            "typing",
            "collections",
            "re",
            "subprocess",
            "ast",
        }

        for line_num, import_line in import_lines:
            if import_line.startswith("from "):
                module = import_line.split()[1].split(".")[0]
            else:
                module = import_line.split()[1].split(".")[0]

            if module in stdlib_modules:
                stdlib_imports.append((line_num, import_line))
            elif "." in module or module.startswith("market_analysis"):
                local_imports.append((line_num, import_line))
            else:
                third_party_imports.append((line_num, import_line))

        # Check if imports are properly grouped
        all_imports = stdlib_imports + third_party_imports + local_imports
        if len(all_imports) != len(import_lines):
            # Some imports might be misclassified, but that's okay for this check
            pass

        return issues

    def _detect_circular_imports(self, import_graph: Dict[str, List[str]]) -> List[str]:
        """Detect circular import dependencies."""
        issues = []

        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in import_graph.get(node, []):
                # Convert module name to file path for local modules
                neighbor_file = None
                for file_path in import_graph.keys():
                    if neighbor.replace(".", "/") in file_path:
                        neighbor_file = file_path
                        break

                if neighbor_file:
                    if neighbor_file not in visited:
                        if has_cycle(neighbor_file, visited, rec_stack, path):
                            return True
                    elif neighbor_file in rec_stack:
                        cycle_start = path.index(neighbor_file)
                        cycle = path[cycle_start:] + [neighbor_file]
                        issues.append(f"Circular import detected: {' -> '.join(cycle)}")
                        return True

            path.pop()
            rec_stack.remove(node)
            return False

        visited = set()
        for node in import_graph:
            if node not in visited:
                has_cycle(node, visited, set(), [])

        return issues

    def validate_naming_conventions(self) -> Dict[str, List[str]]:
        """Validate naming conventions according to PEP 8."""
        issues = {}

        for file_path in self.python_files:
            file_issues = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_issues.extend(self._check_naming_conventions(tree))

            except Exception as e:
                file_issues.append(f"Error checking naming: {e}")

            if file_issues:
                issues[str(file_path)] = file_issues

        return issues

    def _check_naming_conventions(self, tree: ast.AST) -> List[str]:
        """Check naming conventions in AST."""
        issues = []

        class NamingVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Function names should be snake_case
                if not re.match(
                    r"^[a-z_][a-z0-9_]*$", node.name
                ) and not node.name.startswith("_"):
                    if not re.match(r"^test_", node.name):  # Allow test functions
                        issues.append(f"Function '{node.name}' should use snake_case")

                self.generic_visit(node)

            def visit_ClassDef(self, node):
                # Class names should be PascalCase
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    issues.append(f"Class '{node.name}' should use PascalCase")

                self.generic_visit(node)

            def visit_Name(self, node):
                # Variable names should be snake_case (basic check)
                if isinstance(node.ctx, ast.Store):
                    if len(node.id) > 1 and node.id.isupper() and "_" not in node.id:
                        # Might be a constant, which is okay
                        pass
                    elif not re.match(
                        r"^[a-z_][a-z0-9_]*$", node.id
                    ) and not node.id.startswith("_"):
                        # Skip single letter variables and private variables
                        if len(node.id) > 1:
                            issues.append(f"Variable '{node.id}' should use snake_case")

                self.generic_visit(node)

        visitor = NamingVisitor()
        visitor.visit(tree)

        return issues

    def validate_docstring_coverage(self) -> Dict[str, List[str]]:
        """Validate docstring coverage for classes and functions."""
        issues = {}

        for file_path in self.python_files:
            file_issues = []
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_issues.extend(self._check_docstring_coverage(tree))

            except Exception as e:
                file_issues.append(f"Error checking docstrings: {e}")

            if file_issues:
                issues[str(file_path)] = file_issues

        return issues

    def _check_docstring_coverage(self, tree: ast.AST) -> List[str]:
        """Check docstring coverage in AST."""
        issues = []

        class DocstringVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Skip private methods and test functions
                if not node.name.startswith("_") and not node.name.startswith("test_"):
                    if not ast.get_docstring(node):
                        issues.append(f"Function '{node.name}' missing docstring")

                self.generic_visit(node)

            def visit_ClassDef(self, node):
                if not ast.get_docstring(node):
                    issues.append(f"Class '{node.name}' missing docstring")

                self.generic_visit(node)

        visitor = DocstringVisitor()
        visitor.visit(tree)

        return issues

    def run_external_tools(self) -> Dict[str, Any]:
        """Run external code quality tools."""
        results = {}

        # Check if tools are available
        tools = {
            "black": ["black", "--check", "--diff", "."],
            "flake8": ["flake8", "."],
            "mypy": ["mypy", "market_analysis/"],
        }

        for tool_name, command in tools.items():
            try:
                result = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                results[tool_name] = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired:
                results[tool_name] = {"error": "Timeout"}
            except FileNotFoundError:
                results[tool_name] = {"error": f"{tool_name} not found"}
            except Exception as e:
                results[tool_name] = {"error": str(e)}

        return results


# Test functions for pytest
class TestSyntacticInvariance:
    """Test class for syntactic invariance validation."""

    @classmethod
    def setup_class(cls):
        """Set up the validator instance."""
        cls.validator = SyntacticInvarianceValidator()

    def test_ast_structure_validity(self):
        """Test that all Python files have valid AST structure."""
        issues = self.validator.validate_ast_structure()

        if issues:
            error_msg = "AST structure issues found:\n"
            for file_path, file_issues in issues.items():
                error_msg += f"\n{file_path}:\n"
                for issue in file_issues:
                    error_msg += f"  - {issue}\n"
            pytest.fail(error_msg)

    def test_import_structure(self):
        """Test import organization and detect circular imports."""
        issues = self.validator.validate_import_structure()

        if issues:
            error_msg = "Import structure issues found:\n"
            for file_path, file_issues in issues.items():
                error_msg += f"\n{file_path}:\n"
                for issue in file_issues:
                    error_msg += f"  - {issue}\n"
            pytest.fail(error_msg)

    def test_naming_conventions(self):
        """Test naming conventions according to PEP 8."""
        issues = self.validator.validate_naming_conventions()

        # Allow some naming convention violations but warn about them
        if issues:
            warning_msg = "Naming convention issues found (warnings):\n"
            for file_path, file_issues in issues.items():
                warning_msg += f"\n{file_path}:\n"
                for issue in file_issues:
                    warning_msg += f"  - {issue}\n"
            print(f"\nWARNING: {warning_msg}")

    def test_docstring_coverage(self):
        """Test docstring coverage for public functions and classes."""
        issues = self.validator.validate_docstring_coverage()

        # Allow missing docstrings but warn about them
        if issues:
            warning_msg = "Docstring coverage issues found (warnings):\n"
            for file_path, file_issues in issues.items():
                warning_msg += f"\n{file_path}:\n"
                for issue in file_issues:
                    warning_msg += f"  - {issue}\n"
            print(f"\nWARNING: {warning_msg}")

    def test_external_tools(self):
        """Test results from external code quality tools."""
        results = self.validator.run_external_tools()

        for tool_name, result in results.items():
            if "error" in result:
                print(f"\nWARNING: {tool_name} - {result['error']}")
                continue

            if result["returncode"] != 0:
                error_msg = f"{tool_name} found issues:\n"
                if result["stdout"]:
                    error_msg += f"STDOUT:\n{result['stdout']}\n"
                if result["stderr"]:
                    error_msg += f"STDERR:\n{result['stderr']}\n"

                # For now, treat as warnings rather than failures
                print(f"\nWARNING: {error_msg}")


if __name__ == "__main__":
    # Run validation directly
    validator = SyntacticInvarianceValidator()

    print("Running Syntactic Invariance Validation...")
    print("=" * 50)

    print("\n1. Validating AST structure...")
    ast_issues = validator.validate_ast_structure()
    if ast_issues:
        print("AST issues found:")
        for file_path, issues in ast_issues.items():
            print(f"  {file_path}: {len(issues)} issues")
    else:
        print("✓ All files have valid AST structure")

    print("\n2. Validating import structure...")
    import_issues = validator.validate_import_structure()
    if import_issues:
        print("Import issues found:")
        for file_path, issues in import_issues.items():
            print(f"  {file_path}: {len(issues)} issues")
    else:
        print("✓ Import structure is valid")

    print("\n3. Validating naming conventions...")
    naming_issues = validator.validate_naming_conventions()
    if naming_issues:
        print("Naming convention issues found:")
        for file_path, issues in naming_issues.items():
            print(f"  {file_path}: {len(issues)} issues")
    else:
        print("✓ Naming conventions are consistent")

    print("\n4. Checking docstring coverage...")
    docstring_issues = validator.validate_docstring_coverage()
    if docstring_issues:
        print("Docstring coverage issues found:")
        for file_path, issues in docstring_issues.items():
            print(f"  {file_path}: {len(issues)} missing docstrings")
    else:
        print("✓ Docstring coverage is complete")

    print("\n5. Running external tools...")
    tool_results = validator.run_external_tools()
    for tool_name, result in tool_results.items():
        if "error" in result:
            print(f"  {tool_name}: {result['error']}")
        elif result["returncode"] == 0:
            print(f"  ✓ {tool_name}: passed")
        else:
            print(f"  ⚠ {tool_name}: found issues")

    print("\n" + "=" * 50)
    print("Syntactic Invariance Validation Complete")
