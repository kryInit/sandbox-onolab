import React, { useCallback, useEffect, useRef } from "react";

export type ReactiveDivProps = React.DetailedHTMLProps<React.HTMLAttributes<HTMLDivElement>, HTMLDivElement> & {
    resizeHandler(width: number, height: number): void;
};

export const ReactiveDiv: React.FC<ReactiveDivProps> = (rawProps) => {
    const divRef = useRef<HTMLDivElement | null>(null);

    const { resizeHandler, ...props } = rawProps;

    const resizeHandlerWrapper = useCallback(() => {
        if (divRef.current === null) return;
        const { clientWidth, clientHeight } = divRef.current;
        resizeHandler(clientWidth, clientHeight);
    }, []);

    useEffect(() => {
        window.addEventListener("resize", resizeHandlerWrapper);
        return () => window.removeEventListener("resize", resizeHandlerWrapper);
    }, []);

    useEffect(() => {
        resizeHandlerWrapper();
    }, [divRef.current]);

    return (
        <div {...props} ref={divRef}>
            {props.children}
        </div>
    );
};
