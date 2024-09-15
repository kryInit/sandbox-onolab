import {SetupStatus} from "../python-runner/pyodide-worker/PyodideWorkerInterface.ts";
import {Text} from "@fluentui/react-components";
import {centeringProps} from "../utils/generalStyles.ts";
import {SelfBuildingSquareSpinner} from "react-epic-spinners";
import React from "react";

export type LoadingScreenProps = {
    message: SetupStatus | string;
}
export const LoadingScreen: React.FC<LoadingScreenProps> = (props) => {
    return (
        <div style={{ ...centeringProps, gap: "10px", backgroundColor: "gray" }} >
            <SelfBuildingSquareSpinner color={"black"}/>
            <Message message={props.message}/>
        </div>
    );
};

type MessageProps = {
    message: SetupStatus | string;
}
const Message: React.FC<MessageProps> = (props) => {
    const message = props.message
    if (typeof message === "string") return <Text>{message}</Text>;

    return (
        <Text>
            <Text style={{fontFamily: "Menlo"}}>{message.target}</Text>
            {message.type === "initialize" ? "を初期化しています..." : "を読み込んでいます..."}
        </Text>
    );
}