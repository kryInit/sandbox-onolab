export type PyodideWorkerRequest = PyodideWorkerInitializeRequest | PyodideWorkerExecuteRequest;
export type PyodideWorkerResponse = PyodideWorkerInitializeResponse | PyodideWorkerExecuteResponse;

export type PyodideWorkerInitializeRequest = {
    type: "initialize";
};

export type PyodideWorkerExecuteRequest = { type: "execute" } & PyodideWorkerExecuteRequestCore;
export type PyodideWorkerExecuteRequestCore = {
    requestId: string;
    sourceCode: string;
    argNames: string[];
    args: Record<string, unknown>;
};

export type PyodideWorkerInitializeResponse = { type: "initialize" } & PyodideWorkerInitializeResponseCore;
export type PyodideWorkerInitializeResponseCore = { finished: true } | { finished: false; status: SetupStatus };
export type SetupStatus =
    | {
          type: "initialize";
          target: string;
      }
    | {
          type: "load";
          target: string;
      };

export type PyodideWorkerExecuteResponse = {
    type: "execute";
    requestId: string;
    result: unknown;
};
