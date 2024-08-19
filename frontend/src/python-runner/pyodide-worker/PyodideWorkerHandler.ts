import RawPyodideWorker from "./RawPyodideWorker.ts?worker";
import {
    PyodideWorkerExecuteRequest,
    PyodideWorkerExecuteRequestCore,
    PyodideWorkerExecuteResponse,
    PyodideWorkerInitializeRequest,
    PyodideWorkerInitializeResponse,
    PyodideWorkerResponse,
} from "./PyodideWorkerInterface.ts";

export type InitializeResponseHandler = (response: PyodideWorkerInitializeResponse) => void;
export type ExecuteResponseHandler = (response: PyodideWorkerExecuteResponse) => void;

export class PyodideWorkerHandler {
    private readonly worker: Worker;

    constructor(
        private initializeResponseHandler: InitializeResponseHandler = () => {},
        private executeResponseHandler: ExecuteResponseHandler = () => {},
    ) {
        this.worker = new RawPyodideWorker();
        this.worker.onmessage = (event: MessageEvent<PyodideWorkerResponse>) => {
            if (event.data.type === "initialize") this.initializeResponseHandler(event.data);
            else if (event.data.type === "execute") this.executeResponseHandler(event.data);
        };
    }

    sendInitializeRequest() {
        this.worker.postMessage({ type: "initialize" } satisfies PyodideWorkerInitializeRequest);
    }

    sendExecuteRequest(request: PyodideWorkerExecuteRequestCore) {
        this.worker.postMessage({ type: "execute", ...request } satisfies PyodideWorkerExecuteRequest);
    }

    terminate() {
        this.worker.terminate();
    }
}
