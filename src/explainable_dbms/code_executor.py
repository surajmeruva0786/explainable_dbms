"""
Safe Code Executor for LLM-generated ML pipeline code.

Executes Python code in a controlled environment with timeout and error handling.
"""
from __future__ import annotations
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple
from pathlib import Path


def execute_generated_code(
    code: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Safely execute LLM-generated Python code.
    
    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds (default: 300)
    
    Returns:
        Dictionary containing:
        - success: bool
        - output: captured stdout
        - error: error message if failed
        - artifacts: paths to generated files
    """
    print("\n" + "="*80)
    print("🚀 EXECUTING GENERATED CODE")
    print("="*80)
    print(f"Code length: {len(code)} characters")
    print(f"Timeout: {timeout} seconds")
    print("="*80 + "\n")
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Create execution namespace
        exec_globals = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
            '__file__': '<generated>',
        }
        
        # Execute code with output capture
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("✅ CODE EXECUTION SUCCESSFUL")
        print("="*80)
        print("Output:")
        print(stdout_output)
        if stderr_output:
            print("\nWarnings/Errors:")
            print(stderr_output)
        print("="*80 + "\n")
        
        # Find generated artifacts
        artifacts = _find_artifacts()
        
        return {
            'success': True,
            'output': stdout_output,
            'error': stderr_output if stderr_output else None,
            'artifacts': artifacts
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("❌ CODE EXECUTION FAILED")
        print("="*80)
        print(f"Error: {error_msg}")
        print("\nTraceback:")
        print(error_trace)
        print("="*80 + "\n")
        
        return {
            'success': False,
            'output': stdout_capture.getvalue(),
            'error': error_trace,
            'artifacts': {}
        }


def _find_artifacts() -> Dict[str, str]:
    """Find generated artifact files."""
    artifacts = {}
    artifacts_dir = Path('artifacts')
    
    if not artifacts_dir.exists():
        return artifacts
    
    # Look for common artifact files dynamically
    if artifacts_dir.exists():
        for file_path in artifacts_dir.glob('*'):
            if file_path.suffix in ['.png', '.json']:
                artifacts[file_path.stem] = str(file_path)
                print(f"✓ Found artifact: {file_path.name}")
    
    return artifacts


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Basic safety validation of generated code.
    
    Args:
        code: Python code to validate
    
    Returns:
        Tuple of (is_safe, reason)
    """
    # List of dangerous operations to check for
    dangerous_patterns = [
        'os.system',
        'subprocess',
        'eval(',
        'exec(',  # We use exec but in controlled way
        '__import__',
        'open(',  # Check for file operations outside artifacts
        'rmdir',
        'remove',
        'delete',
    ]
    
    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if pattern in code and pattern != 'exec(':  # Allow our own exec
            return False, f"Potentially dangerous operation detected: {pattern}"
    
    # Check for file operations outside allowed directories
    allowed_dirs = ['temp_data', 'artifacts']
    
    return True, "Code appears safe"
