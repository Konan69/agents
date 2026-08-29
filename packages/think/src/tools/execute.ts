import type { JSONValue, Tool, ToolSet } from "ai";
import {
  createCodemodeRuntime,
  DynamicWorkerExecutor,
  truncateResult,
  type CodemodeConnector,
  type CodemodeRuntimeHandle,
  type Executor,
  type ProxyToolOutput
} from "@cloudflare/codemode";
import { ToolSetConnector } from "@cloudflare/codemode/ai";
import type { StateBackend, WorkspaceFsLike } from "@cloudflare/shell";
import { createWorkspaceStateBackend } from "@cloudflare/shell";
import { StateConnector } from "@cloudflare/shell/workers";
import {
  BrowserConnector,
  DurableBrowserSessionStore,
  type BrowserBinding,
  type BrowserConnectorSessionOptions
} from "agents/browser";
import type { WorkspaceLike } from "./workspace";

/**
 * The minimum agent surface for the `createExecuteTool(this)` one-liner.
 * Any Think agent satisfies it. The agent must be a Durable Object — its
 * `ctx` hosts the codemode runtime facet, and its `env` provides the
 * LOADER/BROWSER bindings. (`ctx` and `env` are not declared here because
 * they are `protected` on the DO base class in some type configurations,
 * which would break structural assignability of `this`.)
 */
export interface ExecuteToolAgent {
  workspace?: WorkspaceLike;
  /** Set by `createExecuteRuntime(agent)` so callables can reach the runtime. */
  codemode?: CodemodeRuntimeHandle;
}

export interface CreateExecuteToolOptions {
  /**
   * Durable Object state. The codemode runtime that backs the execute tool
   * lives in a facet of this DO — the tool must be created from inside a
   * Durable Object (e.g. a Think agent: pass `this.ctx`).
   */
  ctx: DurableObjectState;

  /**
   * AI SDK tools exposed inside the sandbox as `tools.*`. Tools with
   * `needsApproval` get the runtime's durable pause/approve/resume flow:
   * calling one pauses the execution until `approveExecution` /
   * `rejectExecution`. A function-valued `needsApproval` can't be evaluated
   * against sandbox arguments ahead of time, so it conservatively always
   * requires approval. Tools without an `execute` function (client-side /
   * provider-executed) are skipped — the sandbox can't call them.
   */
  tools?: ToolSet;

  /**
   * StateBackend exposed as `state.*` inside the sandbox — the full
   * filesystem API (readFile, writeFile, glob, searchFiles, replaceInFiles,
   * planEdits, …). Every method takes a single object argument.
   *
   * @example
   * ```ts
   * import { createWorkspaceStateBackend } from "@cloudflare/shell";
   * state: createWorkspaceStateBackend(this.workspace)
   * ```
   */
  state?: StateBackend;

  /**
   * Browser Rendering binding. When provided, the sandbox gets the `cdp.*`
   * connector (cdp.send, cdp.attachToTarget, cdp.spec, …) — a live browser
   * over the Chrome DevTools Protocol.
   *
   * Requires `"browser": { "binding": "BROWSER" }` in wrangler.jsonc.
   */
  browser?: BrowserBinding;

  /**
   * Browser session lifecycle (only with `browser`). Defaults to `dynamic`:
   * one-shot sessions the model can promote with `cdp.startSession()`.
   */
  session?: BrowserConnectorSessionOptions;

  /**
   * Additional connectors for the sandbox beyond `tools`, `state`, and `cdp`.
   * Each adds its own named namespace.
   */
  connectors?: CodemodeConnector[];

  /**
   * The executor that runs the generated code. If not provided, a
   * `DynamicWorkerExecutor` is created from `loader`.
   */
  executor?: Executor;

  /**
   * WorkerLoader binding for creating a `DynamicWorkerExecutor`.
   *
   * Requires `"worker_loaders": [{ "binding": "LOADER" }]` in wrangler.jsonc.
   */
  loader?: WorkerLoader;

  /**
   * Timeout in milliseconds for code execution. Defaults to 30000 (30s).
   * Only used when `loader` is provided (ignored if `executor` is given).
   */
  timeout?: number;

  /**
   * Controls outbound network access from sandboxed code.
   * - `null` (default): fetch() and connect() throw — sandbox is fully isolated.
   * - `undefined`: inherits parent Worker's network access.
   * - A `Fetcher`: all outbound requests route through this handler.
   *
   * Only used when `loader` is provided (ignored if `executor` is given).
   */
  globalOutbound?: Fetcher | null;

  /**
   * Custom tool description. Replaces the generated default entirely — the
   * default explains the codemode workflow and lists each configured
   * namespace (`tools.*`, `state.*`, `cdp.*`) with a usage hint.
   */
  description?: string;

  /**
   * Codemode runtime name — the durable identity of the tool's executions
   * and snippets. Defaults to `"execute"`. Adding or removing connectors
   * does NOT change the identity, so histories survive configuration
   * changes.
   */
  name?: string;
}

/**
 * The execute tool's moving parts, for hosts that need more than the tool:
 *
 * - `runtime` — the codemode runtime handle (approve/reject paused runs,
 *   `expirePaused`, audit via `executions()`, snippets).
 * - `connectors` — the connector set backing the runtime (e.g. the
 *   `BrowserConnector` for host-side `sessionInfo()` / `closeSession()` /
 *   `sweep()`).
 * - `tool` — what `createExecuteTool` returns.
 */
export interface ExecuteRuntime {
  runtime: CodemodeRuntimeHandle;
  connectors: CodemodeConnector[];
  tool: Tool;
}

function isAgent(
  source: CreateExecuteToolOptions | ExecuteToolAgent
): source is ExecuteToolAgent {
  // An options bag has no `env`, but guard against a hand-built object that
  // happens to carry one (e.g. spread from a worker handler): an explicit
  // `executor`/`loader` key marks it as options — agents never have those.
  // (Other option keys like `state`/`browser` can legitimately exist on agent
  // subclasses, so they can't discriminate.)
  return "env" in source && !("executor" in source) && !("loader" in source);
}

// The agent one-liner derives state from the workspace, which requires the
// full filesystem surface (`WorkspaceFsLike`) — a concrete `Workspace` has
// it; a minimal custom `WorkspaceLike` may not.
const WORKSPACE_FS_METHODS = [
  "readFile",
  "readFileBytes",
  "writeFile",
  "writeFileBytes",
  "appendFile",
  "exists",
  "stat",
  "lstat",
  "mkdir",
  "readDir",
  "rm",
  "cp",
  "mv",
  "symlink",
  "readlink",
  "glob"
] as const;

function workspaceFs(
  workspace: WorkspaceLike | undefined
): WorkspaceFsLike | undefined {
  if (!workspace) return undefined;
  const candidate = workspace as unknown as Record<string, unknown>;
  for (const method of WORKSPACE_FS_METHODS) {
    if (typeof candidate[method] !== "function") return undefined;
  }
  return workspace as unknown as WorkspaceFsLike;
}

function optionsFromAgent(agent: ExecuteToolAgent): CreateExecuteToolOptions {
  const env = ((agent as unknown as { env?: unknown }).env ?? {}) as {
    LOADER?: WorkerLoader;
    BROWSER?: BrowserBinding;
  };
  const ctx = (agent as unknown as { ctx?: DurableObjectState }).ctx;
  if (!ctx) {
    throw new Error(
      "createExecuteTool(agent) requires a Durable Object agent — " +
        "call createExecuteTool({ ctx, loader, ... }) with explicit options."
    );
  }
  const fs = workspaceFs(agent.workspace);
  return {
    ctx,
    loader: env.LOADER,
    state: fs ? createWorkspaceStateBackend(fs) : undefined,
    browser: env.BROWSER
  };
}

// A 16k-token budget at Codemode's cross-model estimate of four characters
// per token keeps the inline execute result near 64k characters. Oversized
// serialized results spill instead of silently losing their tail.
const EXECUTE_RESULT_INLINE_MAX_TOKENS = 16_000;
const EXECUTE_RESULT_INLINE_MAX_CHARS = EXECUTE_RESULT_INLINE_MAX_TOKENS * 4;
// The preview uses half the result budget so the enclosing tool protocol,
// execution id, hint, and future small metadata stay below the inline ceiling.
const EXECUTE_RESULT_PREVIEW_MAX_CHARS = EXECUTE_RESULT_INLINE_MAX_CHARS / 2;
// Workspace-backed Durable Object writes are kept below the 1 MB value limit.
const EXECUTE_RESULT_WRITE_CHUNK_MAX_CHARS = 900_000;
const EXECUTE_RESULT_SPILL_HINT =
  "Read it with read/grep or state.readFile in slices";

/** An oversized execute result replaced by a durable, model-retrievable file. */
type SpilledExecuteResult = {
  readonly spilled: true;
  readonly path: string;
  readonly bytes: number;
  readonly preview: string;
  readonly hint: typeof EXECUTE_RESULT_SPILL_HINT;
};

/** Preserve today's bounded behavior whenever durable spill is unavailable. */
function truncateExecuteResult(result: unknown): unknown {
  return truncateResult(result, {
    maxTokens: EXECUTE_RESULT_INLINE_MAX_TOKENS
  });
}

/** Return the exact JSON text persisted for a spill, when JSON supports it. */
function serializeExecuteResult(result: unknown): string | undefined {
  try {
    return JSON.stringify(result);
  } catch {
    return undefined;
  }
}

/** Render a stable lowercase SHA-256 digest for a content-addressed path. */
async function executeResultDigest(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join(
    ""
  );
}

/** Avoid splitting a UTF-16 surrogate pair across two separately encoded writes. */
function executeResultChunkEnd(serialized: string, offset: number): number {
  const proposedEnd = Math.min(
    offset + EXECUTE_RESULT_WRITE_CHUNK_MAX_CHARS,
    serialized.length
  );
  if (proposedEnd === serialized.length) return proposedEnd;

  const finalCodeUnit = serialized.charCodeAt(proposedEnd - 1);
  const nextCodeUnit = serialized.charCodeAt(proposedEnd);
  const endsWithHighSurrogate =
    finalCodeUnit >= 0xd800 && finalCodeUnit <= 0xdbff;
  const startsWithLowSurrogate =
    nextCodeUnit >= 0xdc00 && nextCodeUnit <= 0xdfff;
  return endsWithHighSurrogate && startsWithLowSurrogate
    ? proposedEnd - 1
    : proposedEnd;
}

/** Write the complete serialized result without crossing the per-write limit. */
async function writeExecuteResultSpill(
  state: StateBackend,
  path: string,
  serialized: string
): Promise<void> {
  let offset = 0;
  let firstChunk = true;
  while (offset < serialized.length) {
    const end = executeResultChunkEnd(serialized, offset);
    const chunk = serialized.slice(offset, end);
    if (firstChunk) {
      await state.writeFile(path, chunk);
      firstChunk = false;
    } else {
      await state.appendFile(path, chunk);
    }
    offset = end;
  }
}

/**
 * Keep small results inline and replace oversized results with an honest
 * `spilled: true` tagged shape whose path contains the full JSON value.
 */
async function spillExecuteResult(
  state: StateBackend | undefined,
  result: unknown
): Promise<unknown> {
  const serialized = serializeExecuteResult(result);
  if (
    serialized === undefined ||
    serialized.length <= EXECUTE_RESULT_INLINE_MAX_CHARS
  ) {
    return result;
  }
  if (!state) return truncateExecuteResult(result);

  try {
    const encoded = new TextEncoder().encode(serialized);
    const digest = await executeResultDigest(encoded);
    const path = `/tool-output/execute-${digest}.json`;
    await writeExecuteResultSpill(state, path, serialized);
    const spilled: SpilledExecuteResult = {
      spilled: true,
      path,
      bytes: encoded.byteLength,
      preview: serialized.slice(0, EXECUTE_RESULT_PREVIEW_MAX_CHARS),
      hint: EXECUTE_RESULT_SPILL_HINT
    };
    return spilled;
  } catch (error) {
    console.warn(
      "think: failed to spill oversized execute result; truncating it instead.",
      error
    );
    return truncateExecuteResult(result);
  }
}

/**
 * Build the codemode runtime behind the execute tool, returning the runtime
 * handle and connectors alongside the tool. Use this instead of
 * {@link createExecuteTool} when the host needs approvals, the audit trail,
 * snippets, or browser session management.
 *
 * When called with an agent, the runtime handle is also assigned to
 * `agent.codemode` so callables (and Think's built-in approval flow) can
 * reach it.
 */
export function createExecuteRuntime(
  source: CreateExecuteToolOptions | ExecuteToolAgent,
  overrides?: Partial<Omit<CreateExecuteToolOptions, "ctx">>
): ExecuteRuntime {
  const agent = isAgent(source) ? source : undefined;
  const options: CreateExecuteToolOptions = isAgent(source)
    ? { ...optionsFromAgent(source), ...overrides }
    : { ...source, ...overrides };

  if (agent && !options.executor && !options.loader) {
    throw new Error(
      "createExecuteTool(agent) requires a WorkerLoader binding named LOADER — " +
        'add `"worker_loaders": [{ "binding": "LOADER" }]` to wrangler.jsonc, ' +
        "or call createExecuteTool({ ctx, loader, ... }) with explicit options."
    );
  }

  let executor: Executor;
  if (options.executor) {
    executor = options.executor;
  } else if (options.loader) {
    executor = new DynamicWorkerExecutor({
      loader: options.loader,
      timeout: options.timeout,
      globalOutbound: options.globalOutbound
    });
  } else {
    throw new Error(
      "createExecuteTool requires either an `executor` or a `loader` " +
        '(WorkerLoader binding — `"worker_loaders": [{ "binding": "LOADER" }]` ' +
        "in wrangler.jsonc)."
    );
  }

  const connectors: CodemodeConnector[] = [];
  if (options.tools && Object.keys(options.tools).length > 0) {
    connectors.push(
      new ToolSetConnector(options.ctx, { tools: options.tools })
    );
  }
  if (options.state) {
    connectors.push(new StateConnector(options.ctx, options.state));
  }
  if (options.browser) {
    connectors.push(
      new BrowserConnector(options.ctx, {
        browser: options.browser,
        store: new DurableBrowserSessionStore(options.ctx.storage),
        session: options.session ?? { mode: "dynamic" },
        timeout: options.timeout
      })
    );
  }
  connectors.push(...(options.connectors ?? []));

  if (connectors.length === 0) {
    throw new Error(
      "createExecuteTool has nothing to expose — provide at least one of " +
        "`tools`, `state`, `browser`, or `connectors`."
    );
  }

  const runtime = createCodemodeRuntime({
    ctx: options.ctx,
    executor,
    connectors,
    name: options.name ?? "execute",
    transformResult: (result) => spillExecuteResult(options.state, result)
  });

  if (agent) {
    agent.codemode = runtime;
  }

  const baseTool = runtime.tool({
    description: options.description,
    connectorHints: connectorHints(options)
  });
  const baseExecute = baseTool.execute;
  const modelTool: Tool = {
    ...baseTool,
    /**
     * Codemode returns `calls` so persisted UI and audit views can inspect the
     * durable replay log. That trace can contain every nested call's complete
     * arguments and result, so it is metadata rather than model context. AI
     * SDK replay keeps the raw output on the UI part and applies this projection
     * only to the tool result sent to the model on this and later turns.
     */
    toModelOutput: ({ output }) => {
      const { calls: _calls, ...modelOutput } = output as ProxyToolOutput;
      return {
        type: "json",
        // Codemode tool outputs are JSON protocol values. Removing one field
        // preserves that contract even though `result` is typed as `unknown`.
        value: modelOutput as JSONValue
      };
    }
  };
  const tool: Tool = baseExecute
    ? ({
        ...modelTool,
        // Paused outputs land in the transcript (and the model context) as a
        // normal tool result; a gated call's raw args (e.g. a writeFile
        // payload) can be huge. Truncate them in the model-facing payload —
        // the runtime facet keeps the full args for the actual resume, and
        // the host can fetch detail via `pendingExecutions`.
        execute: async (input: unknown, callOptions: unknown) =>
          truncatePausedExecutionOutput(
            await (baseExecute as (i: unknown, o: unknown) => Promise<unknown>)(
              input,
              callOptions
            )
          )
      } as Tool)
    : modelTool;

  return {
    runtime,
    connectors,
    tool
  };
}

/**
 * One-line usage hints for the namespaces createExecuteTool itself wires up,
 * rendered into the default tool description. Models otherwise tend to guess
 * a filesystem API (`host.*`, `fs.*`) instead of discovering `state.*` /
 * `tools.*` via `codemode.search`.
 */
function connectorHints(
  options: CreateExecuteToolOptions
): Record<string, string> {
  const hints: Record<string, string> = {};
  if (options.tools) {
    const names = Object.entries(options.tools)
      .filter(([, t]) => typeof t.execute === "function")
      .map(([name]) => name);
    if (names.length > 0) {
      hints.tools =
        `your host tools as async functions — e.g. \`await tools.${names[0]}({ ... })\`. ` +
        `Available: ${names.join(", ")}`;
    }
  }
  if (options.state) {
    hints.state =
      "the workspace filesystem. Every method takes ONE object argument: " +
      "`state.readFile({ path })`, `state.writeFile({ path, content })`, " +
      "`state.readdir({ path })`, `state.glob({ pattern })`, …. " +
      "Each connector result must stay under the 1 MB durable-log limit; " +
      "fetch larger data in chunks and append each chunk to state.*.";
  }
  if (options.browser) {
    hints.cdp =
      "a live browser over the Chrome DevTools Protocol. " +
      'Target-scoped commands need no sessionId: `cdp.send({ method: "Target.createTarget", params: { url } })`. ' +
      "Page-scoped commands (Page.*, Runtime.*, DOM.*) need one: " +
      "`const { sessionId } = await cdp.attachToTarget({ targetId })`, then " +
      "`cdp.send({ method, params, sessionId })`. Discover commands with `cdp.spec()`";
  }
  return hints;
}

/** Character budget for a pending action's args in the transcript. */
const PENDING_ARGS_MAX_CHARS = 2_000;

/**
 * Truncate the `pending[].args` of a paused execution output for transcript /
 * model consumption. The full args stay on the runtime facet (used by the
 * actual resume); only the model-facing copy is bounded. Non-paused outputs
 * pass through unchanged.
 */
export function truncatePausedExecutionOutput(output: unknown): unknown {
  if (typeof output !== "object" || output === null) return output;
  const o = output as { status?: unknown; pending?: unknown };
  if (o.status !== "paused" || !Array.isArray(o.pending)) return output;
  return {
    ...o,
    pending: o.pending.map((action) => {
      if (typeof action !== "object" || action === null) return action;
      const a = action as { args?: unknown };
      if (!("args" in a)) return action;
      return {
        ...a,
        args: truncateResult(a.args, { maxChars: PENDING_ARGS_MAX_CHARS })
      };
    })
  };
}

/**
 * Create a code execution tool that lets the LLM write and run TypeScript
 * against your tools, the workspace filesystem, and (optionally) a live
 * browser — all inside a sandboxed Worker, recorded on a durable codemode
 * runtime (abort-and-replay, approvals, snippets).
 *
 * The model sees typed namespaces: `tools.*` for your AI SDK tools,
 * `state.*` for the filesystem (object args: `state.readFile({ path })`),
 * and `cdp.*` for the browser.
 *
 * Setup checklist:
 *
 * - `"worker_loaders": [{ "binding": "LOADER" }]` in wrangler.jsonc
 * - `"browser": { "binding": "BROWSER" }` in wrangler.jsonc (for `cdp.*`)
 * - export the runtime class from your worker entry:
 *   `export { CodemodeRuntime } from "@cloudflare/codemode"`
 *   (the `@cloudflare/codemode/vite` plugin does this automatically)
 *
 * @example One-liner — defaults from the agent
 * ```ts
 * getTools() {
 *   return {
 *     // state.* from this.workspace, cdp.* if env.BROWSER is bound,
 *     // executor from env.LOADER
 *     execute: createExecuteTool(this)
 *   };
 * }
 * ```
 *
 * @example Agent defaults plus overrides (e.g. custom tools.*)
 * ```ts
 * execute: createExecuteTool(this, { tools: myDomainTools })
 * ```
 *
 * @example Explicit options
 * ```ts
 * execute: createExecuteTool({
 *   ctx: this.ctx,
 *   tools: myDomainTools,                                  // tools.*
 *   state: createWorkspaceStateBackend(this.workspace),    // state.*
 *   browser: this.env.BROWSER,                             // cdp.*
 *   loader: this.env.LOADER
 * })
 * ```
 *
 * Use {@link createExecuteRuntime} to also get the runtime handle
 * (approvals, audit, snippets) and the connector set.
 */
export function createExecuteTool(
  source: CreateExecuteToolOptions | ExecuteToolAgent,
  overrides?: Partial<Omit<CreateExecuteToolOptions, "ctx">>
): Tool {
  return createExecuteRuntime(source, overrides).tool;
}
