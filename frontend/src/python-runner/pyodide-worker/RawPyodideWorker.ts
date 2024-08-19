import { loadPyodide } from "pyodide";
import { Pyodide } from "../common.ts";
import { PyodideWorkerExecuteRequest, PyodideWorkerInitializeResponse, PyodideWorkerInitializeResponseCore, PyodideWorkerRequest } from "./PyodideWorkerInterface.ts";

const sendPyodideWorkerInitializeResponse = (response: PyodideWorkerInitializeResponseCore) =>
    self.postMessage({ type: "initialize", ...response } satisfies PyodideWorkerInitializeResponse);

const initializePyodideWithPostMessage = async (): Promise<Pyodide> => {
    sendPyodideWorkerInitializeResponse({ finished: false, status: { type: "load", target: "Pyodide" } });
    const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/" });

    sendPyodideWorkerInitializeResponse({ finished: false, status: { type: "load", target: "numpy" } });
    await pyodide.loadPackage("numpy");

    sendPyodideWorkerInitializeResponse({ finished: false, status: { type: "load", target: "matplotlib" } });
    await pyodide.loadPackage("matplotlib");

    sendPyodideWorkerInitializeResponse({ finished: true });
    return pyodide;
};

const execPythonCode = (pyodide: Pyodide, sourceCode: string, argNames: string[], args: Record<string, unknown>): unknown => {
    argNames.forEach((name) => {
        pyodide.globals.set(name, args[name]);
    });

    const result = pyodide.runPython(sourceCode);

    argNames.forEach((name) => {
        pyodide.globals.delete(name);
    });

    return result.toJs();
};
const execPythonCodeWithPostMessage = (pyodide: Pyodide, request: PyodideWorkerExecuteRequest) => {
    const { requestId, sourceCode, argNames, args } = request;
    const result = execPythonCode(pyodide, sourceCode, argNames, args);
    self.postMessage({ type: "execute", requestId, result });
};

let pyodide: Pyodide;

self.addEventListener("message", async (event: MessageEvent<PyodideWorkerRequest>) => {
    if (event.data.type === "initialize") pyodide = await initializePyodideWithPostMessage();
    if (event.data.type === "execute") execPythonCodeWithPostMessage(pyodide, event.data);
});

export default {};
