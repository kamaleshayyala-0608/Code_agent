# Enterprise Code Refactoring Specification (Project Memory)

## 1. Single Responsibility Principle (SRP)
- Functions should perform exactly one logical task.
- Large monolithic components must be organized internally into clean, modular helper routines.

## 2. Type Safety & Annotations
- Python functions must contain PEP 484 type annotations for arguments and returns.
- TypeScript/React code must use explicit type signatures instead of `any`.

## 3. Performance & Optimization
- React components must avoid unnecessary re-renders (use `useMemo` / `useCallback` where appropriate).
- Avoid redundant array iterations (`.map`, `.filter`, `.reduce` chaining).
- In Python, use generator expressions for large data operations.

## 4. Security & Error Handling
- Never hardcode credentials, API keys, or raw secrets.
- Use guarded exception handlers and avoid bare `except:`.
- Prevent SQL injection and command injection through proper parameterization.

## 5. Clean Code & Naming
- Eliminate magic numbers by extracting named constants at the module header.
- Remove dead code, unused imports, and unreferenced variables.
- Standardize spacing, line endings, and file formatting.
