/**
 * Test agent for the unified execute tool (Stage 3b): a real Think agent
 * whose execute tool is backed by createCodemodeRuntime — real facet, real
 * DynamicWorkerExecutor sandbox, real Workers RPC for connector calls.
 */
import { tool } from "ai";
import type { LanguageModel } from "ai";
import { z } from "zod";
import type { WorkspaceFsLike } from "@cloudflare/shell";
import { createWorkspaceStateBackend } from "@cloudflare/shell";
import { Think } from "../../think";
import {
  createExecuteRuntime,
  createExecuteTool,
  type ExecuteRuntime
} from "../../tools/execute";

// `result` is kept to RPC-serializable primitives so the DurableObjectStub
// method types don't collapse to `never` in tests.
type ExecuteOutput = {
  status: string;
  executionId?: string;
  result?: string | number | boolean | null;
  error?: string;
  pending?: Array<{ connector: string; method: string }>;
};

type ExecuteModelProjection = {
  rawCallCount: number;
  rawCallResultChars: number;
  modelHasCalls: boolean;
  modelStatus: string;
  modelResultPayloadChars: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function invoke(
  executeTool: { execute?: unknown },
  code: string
): Promise<ExecuteOutput> {
  const execute = executeTool.execute as (input: {
    code: string;
  }) => Promise<ExecuteOutput>;
  return execute({ code });
}

export class ThinkExecuteToolAgent extends Think {
  getModel(): LanguageModel {
    throw new Error("Model is not used in execute-tool tests");
  }

  #runtime(): ExecuteRuntime {
    return createExecuteRuntime({
      ctx: this.ctx,
      tools: {
        add: tool({
          description: "Add two numbers",
          inputSchema: z.object({ a: z.number(), b: z.number() }),
          execute: async ({ a, b }) => ({ sum: a + b })
        }),
        largePayload: tool({
          description: "Return a deterministic payload for audit-log tests",
          inputSchema: z.object({ size: z.number().int().nonnegative() }),
          execute: async ({ size }) => ({ payload: "x".repeat(size) })
        }),
        launchMissiles: tool({
          description: "Approval-gated — must be stripped from the sandbox",
          inputSchema: z.object({}),
          needsApproval: true,
          execute: async () => "boom"
        })
      },
      state: createWorkspaceStateBackend(
        this.workspace as unknown as WorkspaceFsLike
      ),
      loader: this.env.LOADER
    });
  }

  /** Run code on the explicit-options runtime (tools.* + state.*). */
  async runExecute(code: string): Promise<ExecuteOutput> {
    return invoke(this.#runtime().tool, code);
  }

  /** Run code through the `createExecuteTool(this)` one-liner. */
  async runOneLiner(code: string): Promise<ExecuteOutput> {
    return invoke(createExecuteTool(this), code);
  }

  /**
   * Compare the complete runtime result with the model projection exposed by
   * the same tool. The large nested result proves the replay log remains an
   * audit surface even though it is absent from model context.
   */
  async executeModelProjection(): Promise<ExecuteModelProjection> {
    const executeTool = this.#runtime().tool;
    if (!executeTool.execute || !executeTool.toModelOutput) {
      throw new Error("execute tool is missing execution or model projection");
    }
    const input = {
      code: `async () => {
        const result = await tools.largePayload({ size: 120000 });
        return { payloadChars: result.payload.length };
      }`
    };
    type ModelOutputOptions = Parameters<
      NonNullable<typeof executeTool.toModelOutput>
    >[0];
    const rawOutput = (await executeTool.execute(input, {
      toolCallId: "model-projection",
      messages: [],
      abortSignal: new AbortController().signal,
      context: {}
    })) as ModelOutputOptions["output"];
    if (!isRecord(rawOutput) || !Array.isArray(rawOutput.calls)) {
      throw new Error("raw execute output is missing its calls audit log");
    }
    const firstCall = rawOutput.calls[0];
    if (!isRecord(firstCall) || !isRecord(firstCall.result)) {
      throw new Error("raw execute output is missing the nested call result");
    }
    const rawPayload = firstCall.result.payload;
    if (typeof rawPayload !== "string") {
      throw new Error("nested call result is missing its payload");
    }

    const projected = await executeTool.toModelOutput({
      toolCallId: "model-projection",
      input,
      output: rawOutput
    });
    if (projected.type !== "json" || !isRecord(projected.value)) {
      throw new Error("execute model projection is not a JSON object");
    }
    const modelResult = projected.value.result;
    if (
      !isRecord(modelResult) ||
      typeof modelResult.payloadChars !== "number"
    ) {
      throw new Error("execute model projection lost the script result");
    }

    return {
      rawCallCount: rawOutput.calls.length,
      rawCallResultChars: rawPayload.length,
      modelHasCalls: "calls" in projected.value,
      modelStatus:
        typeof projected.value.status === "string"
          ? projected.value.status
          : "missing",
      modelResultPayloadChars: modelResult.payloadChars
    };
  }

  /** The sandbox type surface advertised by the `tools` connector. */
  async toolsConnectorTypes(): Promise<string> {
    const { connectors } = this.#runtime();
    const toolset = connectors.find((c) => c.name() === "tools");
    if (!toolset) throw new Error("tools connector missing");
    return toolset.getTypeScriptTypes();
  }

  /**
   * Audit trail via the agent-accessible handle — `createExecuteRuntime(this)`
   * (exercised by runOneLiner) assigns `this.codemode`.
   */
  async codemodeExecutionStatuses(): Promise<string[]> {
    if (!this.codemode) return [];
    return (await this.codemode.executions()).map((e) => e.status);
  }
}
