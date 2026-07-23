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

## 6. No Magic Numbers
- Replace hardcoded numbers, strings, or flags with named constants or enums to clarify intent and allow single-source edits.

## 7. Use Type Hints
- In Python, leverage explicit type annotations for function signatures, class attributes, and complex variables.
- Ensure all function inputs and return values have descriptive type annotations.

## 8. Proper Exception Handling
- Use specific exception handling clauses; avoid catch-all `except Exception:` blocks unless logs/re-raises are properly structured.
- Never let exceptions fail silently. Log detailed context and implement graceful degradation or user alerts where applicable.

## 9. No Nested Functions
- Avoid defining local helper functions inside other functions.
- Promote them to file-level helpers, standalone utility functions, or methods to enhance readability, unit testability, and reuse.

## 10. Maximum Function Length
- Maintain small, digestible functions. No function should exceed 40-50 lines of active logic.
- Break larger methods down along functional boundaries.

## 11. Maximum Class Length
- Keep class declarations concise, aiming for under 300 lines.
- Offload responsibilities to auxiliary helper classes or service modules when a class grows too large.

## 12. Logging Standards
- Integrate structured, contextual logging. Do not use plain print statements for production code.
- Categorize logs with correct severity levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

## 13. Caching Rules
- Use in-memory or redis/file-based caching for slow, resource-heavy, or repetitive API integrations or calculations.
- Ensure cache key schemas are unique and expire properly.

## 14. Performance Rules
- Minimize redundant work, nested loops, and slow queries.
- Optimize runtime performance: leverage memoization, lazy loading, and efficient data structures.

## 15. Security Rules
- Prevent injection vectors and sanitize/validate inputs at all trust boundaries.
- Keep credentials, API keys, and sensitive settings out of code; fetch them dynamically from environment variables or secure storage.

## 16. Import Ordering
- Organize imports systematically:
  1. Standard library imports
  2. Third-party library dependencies
  3. Local project modules / relative imports
- Keep import groups separated by a single blank line and sorted alphabetically.

## 17. Folder Structure Rules
- Group modules logically by feature/domain or clean architectural layers.
- Do not bypass module boundaries (no circular references).

## 18. Testing Rules
- Write unit tests or verification functions for critical utility paths and core business logic.
- Keep components testable by decoupling side effects from logic.

