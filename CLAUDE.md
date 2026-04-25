# Coding Bible

## Preamble

This document is not a style guide. It is not a set of preferences. It is not a list of
suggestions that can be weighed against convenience or time pressure.

Every rule in this document exists because its absence has caused real, costly, documented
failures in real systems — bugs that took days to trace, security holes that went undetected
for months, codebases that became unmaintainable within a year of being written. The rules
are the scar tissue. They are what you get when you ask experienced engineers what they wish
someone had forced them to do the first time.

When you are implementing something and a rule feels inconvenient, that feeling is a signal,
not a reason to deviate. Inconvenience usually means the rule is doing its job — preventing
the path of least resistance from becoming a future liability.

**These rules are non-negotiable and not subject to contextual override.** The following
exceptions are not acceptable justifications for deviation:

- "This is a prototype / MVP / quick fix." Prototypes become production. Quick fixes become
  permanent. Code written without discipline under time pressure is the most dangerous code
  that exists.
- "This is a small module that doesn't need it." Small modules grow. The time to enforce
  structure is before the module is large enough to make restructuring painful.
- "The architecture requires it." If the architecture requires violating these rules, the
  architecture is wrong. Fix the architecture.
- "It would take too long to do it correctly." It will take longer to fix it later. It always
  does.

If a genuine, documented edge case exists where a rule cannot be followed, that exception
must be explicitly marked in a comment at the relevant location, stating which rule is being
waived, why, and what mitigating measures are in place. Undocumented deviations are bugs.

---

## The Underlying Principle

All seven rules are expressions of one idea:

**Make the wrong thing hard to do accidentally.**

Good architecture is not about making correct behavior possible. Correct behavior is always
possible. Good architecture is about making incorrect behavior require deliberate, visible,
documented effort — so that mistakes are loud, localized, and recoverable rather than silent,
diffuse, and permanent.

When in doubt, ask: *am I making the wrong thing easy to do accidentally?* If yes, stop and
apply the relevant rule before continuing.

---

These rules are non-negotiable. They exist because every one of them corresponds to a
real failure mode with real consequences. If you think a rule doesn't apply to a specific
case, you are probably wrong. If you are certain it doesn't apply, document why explicitly
in a comment before proceeding.

---

## 1. Fail Fast, Fail Loud, No Defaults

Do not insert default behavior that fails silently. If something breaks, the system must
break immediately, obviously, and traceably.

**Rules:**
- Raise explicit, descriptive errors. Never swallow exceptions.
- Never return a default value to mask a missing or invalid state.
- Log the failure with enough context to reproduce it.

**Corollary — No Optimistic Assumptions:**
Do not assume inputs are valid, dependencies are available, or external calls succeed.
Assert and verify. If an assumption is required, document it and enforce it with a guard.

---

## 2. Trust Your Neighbors — Validate Input, Not Output

You own your output contract. You do not trust anyone else's input contract.

**Rules:**
- All external inputs (API calls, user input, file reads, environment variables) must be
  validated at the boundary before use.
- Internal module output does not need re-validation by the caller — the module is
  responsible for the correctness of what it emits.

**Corollary — Define Your Contracts:**
Every module must have an explicit, documented input/output contract. If a caller violates
the input contract, the module must fail loudly (see Rule 1), not silently compensate.

---

## 3. UNIX Philosophy — One Module, One Job

Each module does one thing only, and does that one thing well. If a module needs to do
multiple things, it must be split into multiple files.

**Rules:**
- No file should be the sole owner of more than one domain concept.
- If you find yourself writing "and" when describing what a file does, split it.
- Tight coupling between modules is a design failure, not an implementation detail.

**Corollary — No God Files:**
A module that knows everything, owns everything, or touches everything is a liability.
If removing a module would require changes across more than two other modules, it is
probably doing too much.

---

## 4. LOC Limits — No Bloated Monoliths

Files must not become monoliths. Complexity must be managed through decomposition,
not consolidation.

**Rules:**
- **Hard limit: 500 lines of code** (comments and docstrings excluded). Do not exceed this
  under any circumstances.
- **Soft limit: 400 lines of code.** Above 400 lines, actively look for split opportunities.
- If a file "requires" more than 500 lines, that is a signal the abstraction is wrong —
  split it into multiple focused files.

**Corollary — Splitting Is Not Optional:**
Exceeding the hard limit is never acceptable, even temporarily. Split first, implement after.

---

## 5. No Hidden State — Explicit Data Flow

All state must be explicitly passed or managed through clearly defined interfaces.
Hidden state is a bug waiting to be discovered at the worst possible time.

**Rules:**
- No global variables. No module-level mutable state unless explicitly documented and
  architecturally justified.
- No implicit state — if a function's behavior depends on something, that something must
  appear in its signature or be explicitly injected.
- No hidden coupling between modules. If two modules share state, that relationship must
  be visible and intentional.

**Corollary — Spooky Action at a Distance Is a Bug:**
If changing module A causes unexpected behavior in module B, and there is no explicit
interface between them, the architecture is broken. Fix the architecture, not the symptom.

---

## 6. Deterministic Behavior by Default

All functions must be deterministic unless explicitly documented otherwise.

**Rules:**
- No randomness, time-dependent logic, or non-deterministic behavior without explicit
  documentation and an injectable seed or clock interface for testing.
- Functions that depend on system time, random state, or external non-deterministic
  sources must be flagged with a `# NON-DETERMINISTIC:` comment explaining why.
- Non-deterministic behavior must be isolatable — wrap it so the rest of the system
  can be tested deterministically.

**Corollary — Reproducibility Is Not Optional:**
If a bug cannot be reliably reproduced, it cannot be reliably fixed. Non-determinism
is the enemy of reproducibility. Treat it like a dependency that must be explicitly
managed, not a feature.

---

## 7. Tests Are First-Class Citizens

Every module must have corresponding tests. Tests are not optional, not afterthoughts,
and not someone else's job.

**Rules:**
- Every module must have a corresponding test file covering:
  - **Normal cases** — expected inputs produce expected outputs.
  - **Edge cases** — boundary conditions, empty inputs, maximum values, type boundaries.
  - **Failure cases** — invalid inputs produce the correct explicit errors (see Rule 1).
- No module ships without tests. No exceptions.
- Tests must be runnable in isolation — no test should depend on the state left by
  another test.

**Corollary — Untested Code Is Unverified Assumptions:**
If there is no test for a behavior, that behavior is assumed, not verified. Assumptions
accumulate into system failures. If you cannot write a test for something, that is a signal
the design is wrong.

---

## 8. Release Tracking — Log Every Change

Every functional change must be recorded in `.release/UNRELEASED.md` **at the time
the change is made**, not retroactively before a release.

**Rules:**
- When you modify any source file in the package (bug fix, new feature, performance
  improvement, API change), add a bullet to `.release/UNRELEASED.md` under
  `## Changes` describing what changed, why, and the impact.
- Be specific: include function names, file paths, benchmark numbers, and what the
  change means to a user. "Fixed a bug" is not acceptable. "Fixed `power_crossover_be`
  TOST alpha convention to match R PowerTOST — was using `alpha/2` per test instead
  of `alpha`, resulting in overly conservative sample sizes" is.
- This file is committed to git and synced across all machines / sessions. If you
  make a change on Mac and someone else makes a change on Linux, both changes appear
  in the same file.
- When ready to release, run `python .release/release.py <version>`. This script:
  1. Bumps the version in `pyproject.toml` and `__init__.py`
  2. Prepends the UNRELEASED.md content into `CHANGELOG.md`
  3. Resets UNRELEASED.md for the next cycle
  4. Prints a checklist of remaining manual steps (commit, push, create release)
- The script refuses to release if UNRELEASED.md is empty.
- **Preferred release flow:** pre-stage README changes with `git add README.md`,
  then run `python .release/release.py --commit X.Y.Z`. That flag extends the
  script to commit, tag, and push — leaving only `gh release create` manual.
  See `.release/CHECKLIST.md` for the full flow and the historical foot-guns
  (README, git tag) that `--commit` was added to close.

**Corollary — No Surprise Releases:**
If a release goes out and someone asks "what changed?", the answer must already
be written down. The UNRELEASED.md file is the single source of truth for in-progress
changes. If it's empty, nothing has changed. If something changed but isn't in the
file, that's a process failure.

---

## 9. Cross-Project Scope Boundary — Stay in the Project You Were Invited Into

A Claude Code session is invoked on one project. It does not have a licence to edit
any other project visible on disk. Seeing sibling repos in `/mnt/projects/` does not
constitute an invitation to modify them.

**Rules:**
- Never modify source, tests, configs, or documentation in a sibling project (any
  repo outside the current one) on your own initiative.
- When you believe a change belongs in a sibling project — "this belongs upstream",
  "this helper is more general than the current project", "I should fix a bug in
  their API" — stop and raise it with the user. Describe what you'd change, where,
  and why. Wait for explicit authorisation before editing the sibling repo.
- "Explicit authorisation" means the user said, in the current session, something
  like "yes, also modify project X". It is NOT a precedent from a prior session. It
  is NOT "the user probably wouldn't mind". It is NOT inferred from the fact that
  the user depends on project X.

**Corollary — Scope Creep Is a Design Decision, Not a Technical One:**
Pushing project-A-specific functionality into general-purpose project B because it's
technically convenient dilutes project B's identity, breaks the expectations of other
downstream consumers, and forces project B to carry maintenance burden for a use case
it was never meant to serve. "Project A needs X, and X involves statistics, therefore
X belongs in pystatistics" is not a valid reasoning chain. X belongs in project A
until (a) multiple sibling projects demonstrably need it AND (b) a decision has been
made to promote it — either into an existing general-purpose library or into a new
dedicated sibling.

**What to do when you think a sibling needs a change:**
1. Implement the helper inside your OWN project first. It lives there until proven
   general.
2. Surface the "belongs upstream" thought to the user explicitly. Propose the
   migration as a separate task. Do not pre-emptively modify the sibling repo.
3. If the user authorises the migration, treat it as its own first-class task with
   its own review, tests, and release cadence — not as a side-effect of the
   originating feature.

---

## 10. Audience Awareness for Public-Facing Files

Before modifying any public-facing file (`README.md`, `pyproject.toml`,
user-facing documentation, the PyPI description), explicitly identify the
audience: **a stranger on the internet who has never seen this codebase.**
That person has no knowledge of `CLAUDE.md`, internal project names,
development rationale, rule numbers, or implementation history. They need
to know what the library does, how to install it, and how to use it —
nothing else.

**The test:** *would a first-time user finding this on PyPI find this
useful or confusing?* If confusing, omit it.

**Rules (all specific instances of the principle above):**

- **Changelog entries must not reference internal development artifacts.**
  Entries in `README.md`, `CHANGELOG.md`, or any public-facing document must
  never reference `CLAUDE.md`, internal rule numbers, internal project
  names, development process, or test counts. Acceptable content:
  user-facing behaviour changes, API additions/removals, bug fixes, and
  performance numbers.

  Example of what must **not** appear in a public changelog:
  > "a new Cross-Project Scope Boundary rule (Rule 9) has been added to
  > CLAUDE.md"

- **Do not document `CLAUDE.md` changes in `README.md`.** When updating
  `CLAUDE.md`, do not document those changes in `README.md` or any other
  public-facing file. `CLAUDE.md` is internal developer guidance; `README.md`
  reflects the public API, installation, and usage. `README.md` must never
  reference `CLAUDE.md` by name or by implication.

- **`README.md` is a user document, not a changelog mirror.** When updating
  `README.md` from `CHANGELOG.md` or release notes, do not copy entries
  verbatim or structurally. Translate them: extract only user-facing
  behaviour changes, rewrite in present tense describing the library's
  current state, and omit internal process details (references to
  `CLAUDE.md`, rule numbers, internal project names, test counts,
  development rationale). A user reading the README should understand what
  the library does today — not the history of how it got there.

**Corollary — The Internal Audience Already Has CLAUDE.md:**
Rationale that belongs in `CLAUDE.md` does not need to be duplicated into
public-facing files to "motivate" a change for users. Users do not need to
be sold on the rule behind the change — only told what changed.

---

## Meta-Rule

If any of these rules feel inconvenient, that feeling is the point. These rules exist
precisely because the convenient path leads to the failure modes each rule was written
to prevent. When in doubt, ask: *am I making the wrong thing easy to do accidentally?*
If yes, apply the relevant rule.