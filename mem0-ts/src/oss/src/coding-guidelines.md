# Coding Guidelines Spec

Use this checklist when generating or reviewing code for the Oracle vector store.

## 1) Type safety first

* Avoid `any` in public APIs and implementation details.
* Prefer `unknown` over `any` when the actual shape is not guaranteed.
* Use official library types when available.

  * For `oracledb`, prefer `Connection`, `Pool`, `ConnectionAttributes`, `PoolAttributes`, and `DBError`.
* When narrowing unknown values, prefer property checks or `Partial<T>` over blind type assertions.

## 2) Public API types

* Replace `Record<string, any>` with `Record<string, unknown>` or a dedicated JSON type.
* Use a shared JSON/object type for payloads if the project has one.
* Keep public interfaces aligned with implementation types.
* Do not widen a public interface just to make one implementation compile.

## 3) Config handling

* Prefer a single canonical config property over aliases unless backward compatibility requires both.
* If an alias exists, document it clearly and keep the fallback explicit.
* Reject empty strings for required identifiers.
* Trim identifier inputs before validation.
* Allow omitted optional values to fall back to defaults.
* Avoid unnecessary casts when a property can be validated directly.

## 4) Constructor validation

* Validate user-facing values early in the constructor.
* For enums or finite string sets, normalize input once and validate against an allowlist.
* Keep validation errors specific and actionable.
* Do not check for `null` on a parameter that already has a default object unless callers may bypass TypeScript.

## 5) Connection and resource management

* Keep connection ownership rules explicit:

  * pooled connections are borrowed and released,
  * direct connections are owned and closed by the store only when appropriate.
* Always release pooled connections, even when the operation fails.
* Do not throw from `finally`.
* Cleanup failures should not replace the original operation error.
* If cleanup can fail, log it best-effort and preserve the original failure.

## 6) Transaction behavior

* The store should own its own transaction boundary unless the API explicitly says otherwise.
* Commit on success, rollback on failure for owned direct connections.
* For pooled connections, rely on release semantics unless there is a special reason to rollback manually.
* Never let a partial batch succeed silently when the API promises all-or-nothing behavior.

## 7) Batch write behavior

* Validate batch input lengths before execution.
* When using batch writes, treat batch errors as fatal if the API expects atomicity.
* Log the failing offsets and IDs when available.
* Throw a clear error after rollback rather than returning partial success.
* Prefer a single batch operation over per-row calls when performance matters.

## 8) Error handling

* Preserve the original error whenever possible.
* Avoid `throw` inside `finally`.
* Use helper predicates for expected Oracle errors, such as unique constraint violations.
* Prefer `Partial<oracledb.DBError>` or property narrowing for error inspection.
* Keep error messages stable enough for tests but precise enough for debugging.

## 9) Oracle-specific rules

* Validate Oracle version support before using vector features.
* Keep vector distance metrics in an allowlist.
* Keep index types in an allowlist.
* Validate index parameters against the selected index type.
* Enforce numeric bounds for index parameters.
* Reject unsupported index parameter keys early.
* Keep DDL generation aligned with the runtime validation rules.

## 10) SQL and identifier safety

* Quote identifiers through a dedicated helper.
* Validate identifier syntax before quoting.
* Do not interpolate raw user input into SQL fragments.
* Use bind parameters for values.
* Keep JSON path construction constrained and validated.
* Preserve consistency between query builders and test expectations.

## 11) Filtering rules

* Support the documented filter operators only.
* Validate logical operators (`$or`, `$and`, `$not`) as arrays of filter objects.
* Reject empty filter objects.
* Support shorthand existence and membership semantics only where documented.
* Keep wildcard and nested-path behavior aligned across search and list queries.
* Use the same filter builder for both search and count/list paths.

## 12) Defaults and normalization

* Apply defaults once, in one place.
* Normalize case for config enums before storing them.
* Avoid duplicating derived values in multiple fields unless needed.
* Keep collection name, index name, and dimension derivation consistent.

## 13) Testing guidelines

* Cover:

  * happy paths,
  * invalid config paths,
  * cleanup and rollback behavior,
  * direct connection and pooled connection flows,
  * version gating,
  * filter operators,
  * index generation,
  * unique constraint handling,
  * mutation and upsert behavior.
* Add at least one real integration test for each major Oracle mode:

  * pool,
  * direct connection,
  * HNSW index,
  * IVF index if supported,
  * duplicate insert rollback.
* When mocking, verify SQL shape, bind shape, and cleanup behavior.
* When using real Oracle tests, verify state survives failures.

## 14) Lint-friendly code

* Avoid unsafe `throw` inside `finally`.
* Prefer explicit control flow over clever cleanup tricks.
* Use `unknown` and narrow it safely.
* Keep code readable even when satisfying lint rules.

## 15) Review checklist

Before merging code, check:

* No new `any` in public or internal code unless absolutely unavoidable.
* No redundant aliases or duplicated config fields.
* No cleanup path that can hide the real error.
* No unvalidated SQL interpolation.
* No unchecked Oracle parameter values.
* No batch operation that can leave partial state when atomicity is expected.
* Tests cover both success and failure paths.

## Default implementation preferences

* Prefer `unknown` over `any`.
* Prefer official driver types over hand-rolled shapes.
* Prefer explicit validation over implicit assumptions.
* Prefer one canonical config field over aliases.
* Prefer best-effort cleanup that preserves the original failure.
* Prefer integration tests for real Oracle behavior and unit tests for SQL/rendering/cleanup details.

