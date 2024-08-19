import "normalize.css";
import "./index.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Main from "./App.tsx";
import "./utils/arrayExtensions";
import "./utils/mapExtensions";

createRoot(document.getElementById("root")!).render(
    <StrictMode>
        <Main />
    </StrictMode>,
);
