import { useEffect, useRef, useState } from "react";
import { PyodideWorkerHandler } from "./pyodide-worker/PyodideWorkerHandler.ts";
import { v6 as uuidv6 } from "uuid";
import { PyodideWorkerExecuteResponse, PyodideWorkerInitializeResponse, SetupStatus } from "./pyodide-worker/PyodideWorkerInterface.ts";

class ResponseCallbackMap {
    private map: Map<string, (response: any) => void> = new Map();

    set(requestId: string, resolve: (response: any) => void) {
        this.map.set(requestId, resolve);
    }

    get(requestId: string): ((response: any) => void) | undefined {
        return this.map.get(requestId);
    }

    delete(requestId: string) {
        this.map.delete(requestId);
    }
}

class PythonFunction<Key extends string, ReturnType> {
    constructor(
        public readonly argNames: Key[],
        public readonly sourceCode: string,
        private readonly workerHandler: PyodideWorkerHandler,
        private readonly responseCallbackMap: ResponseCallbackMap,
    ) {}

    private async getExecuteResponse<ReturnType>(requestId: string): Promise<ReturnType> {
        return new Promise<ReturnType>((resolve) => {
            this.responseCallbackMap.set(requestId, resolve);
        });
    }

    exec(args: { [K in Key]: unknown }): Promise<ReturnType> {
        const requestId = uuidv6();
        const responsePromise = this.getExecuteResponse(requestId) as Promise<ReturnType>;
        this.workerHandler.sendExecuteRequest({ requestId, argNames: this.argNames, args, sourceCode: this.sourceCode });
        return responsePromise;
    }
}

export type PythonRunnerStatus = { isAvailable: true } | { isAvailable: false; message: SetupStatus };

export class PythonRunner {
    constructor(
        private readonly workerHandler: PyodideWorkerHandler,
        public readonly status: PythonRunnerStatus,
        private readonly responseCallbackMap: ResponseCallbackMap,
    ) {}

    static createUninitializedPythonRunner = () => {
        return new PythonRunner({} as PyodideWorkerHandler, { isAvailable: false, message: { type: "initialize", target: "WebWorker" } }, new ResponseCallbackMap());
    };

    createFunction<Key extends string, ReturnType>(argNames: Key[], sourceCode: string): PythonFunction<Key, ReturnType> {
        return new PythonFunction(argNames, sourceCode, this.workerHandler, this.responseCallbackMap);
    }

    exec<Key extends string, ReturnType>(argNames: Key[], args: { [K in Key]: unknown }, sourceCode: string): Promise<ReturnType> {
        return this.createFunction<Key, ReturnType>(argNames, sourceCode).exec(args);
    }
}

// 若干実装が微妙かも
// statusによってhookを反応させたいので、initializeResponseHandlerはusePythonRunner内で設定する必要があるように思える
// 愚直に考えるとworkerやresponseCallbackMapはPythonRunner内で生成するべきな気はするが、PythonRunnerはhookの返り値なので複数回生成する必要がありそう & workerは一回しか生成したくなく、
// responseCallbackMapもexecuteResponseHandler内で使うので、ここで生成するのが妥当に思える
// 普通に頭がついてない気もする

export const usePythonRunner = (): PythonRunner => {
    const workerRef = useRef<PyodideWorkerHandler | undefined>(undefined);
    const responseCallbackMapRef = useRef<ResponseCallbackMap>(new ResponseCallbackMap());
    const [pythonRunner, setPythonRunner] = useState<PythonRunner>(PythonRunner.createUninitializedPythonRunner());

    useEffect(() => {
        const initializeResponseHandler = (response: PyodideWorkerInitializeResponse) => {
            if (workerRef.current === undefined) return;

            const status = response.finished ? ({ isAvailable: true } as const) : { isAvailable: false, message: response.status };
            setPythonRunner(new PythonRunner(workerRef.current, status, responseCallbackMapRef.current));
        };
        const executeResponseHandler = (response: PyodideWorkerExecuteResponse) => {
            const resolve = responseCallbackMapRef.current.get(response.requestId);
            if (resolve === undefined) return;

            resolve(response.result);
            responseCallbackMapRef.current.delete(response.requestId);
        };

        workerRef.current = new PyodideWorkerHandler(initializeResponseHandler, executeResponseHandler);
        workerRef.current.sendInitializeRequest();

        return () => workerRef.current?.terminate();
    }, []);

    return pythonRunner;
};
