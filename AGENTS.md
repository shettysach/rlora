# AGENTS.md

Read `AIM.md` before making architectural decisions. When working toward a milestone, read the corresponding milestone document as well.

This is a research repository, so requirements and experiments will change quickly. However, the code will be read and modified by humans.

## Code

- Prefer simple, readable, idiomatic code.
- Avoid unnecessary abstractions, wrappers, configuration, and dependencies.
- Avoid LOC growth when a smaller solution is equally clear.
- Prefer small, focused changes over broad refactors.
- Do not over-engineer for hypothetical future requirements.
- Do not add defensive checks for conditions guaranteed by construction, types, or an upstream API contract. Validate only states that can genuinely vary at runtime, such as user input and external artifacts.
- Performance-critical code may favor efficiency over elegance when the tradeoff is meaningful.

## Branches

`master` should always contain clean, understandable, maintainable code.

Experimental branches may trade code quality for iteration speed when useful. Temporary hacks, instrumentation, and task-specific experiments are acceptable there.

When a successful experiment is ported back to `master`, reimplement or clean it up rather than blindly merging experimental code.

## Research

Implement the simplest version that can answer the current research question.

Do not expand the scope beyond the current milestone unless it is necessary for correctness or significantly simplifies the implementation.

## Tests

Don't write and run unnecessary tests for every small feature. Test only core and risky functionality you feel is appropriate.
