import React, { createContext, useContext, useEffect, useState, useRef, useMemo } from "react";
import { SelfBuildingSquareSpinner } from "react-epic-spinners";
import { Button, Checkbox, FluentProvider, Input, makeStyles, shorthands, Text, tokens, webLightTheme } from "@fluentui/react-components";
import { PythonRunner, usePythonRunner } from "./python-runner/PythonRunner.ts";
import { SetupStatus } from "./python-runner/pyodide-worker/PyodideWorkerInterface.ts";
import { ReactiveDiv } from "./components/ReactiveDiv.tsx";

import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

import { DataAttributes } from "./DataAttribute.ts";

const centeringProps = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    width: "100%",
} as const;

const fetchNpzFile = async (fileName: string): Promise<ArrayBuffer> => {
    const targetUrl = `${window.location.href}data/${fileName}`;
    const response = await fetch(targetUrl);
    if (!response.ok) {
        throw new Error("ファイルのダウンロードに失敗しました");
    }
    return await response.arrayBuffer();
};

const LoadingScreen = ({ message }: { message: SetupStatus | string }) => {
    const MessageComponent = () =>
        typeof message === "string" ? (
            <Text>{message}</Text>
        ) : (
            <Text>
                <Text style={{ fontFamily: "Menlo" }}>{message.target}</Text>
                {message.type === "initialize" ? "を初期化しています..." : "を読み込んでいます..."}
            </Text>
        );

    return (
        <div
            style={{
                ...centeringProps,
                gap: "10px",
                backgroundColor: "gray",
            }}
        >
            <SelfBuildingSquareSpinner color={"black"} />
            <MessageComponent />
        </div>
    );
};

const loadData = async (pythonRunner: PythonRunner, filePaths: string[]): Promise<number[][][] | null> => {
    if (!pythonRunner.status.isAvailable) return null;

    const npzBuffers = await Promise.all(filePaths.map(fetchNpzFile));

    return await pythonRunner.exec<"npzBuffers", number[][][]>(
        ["npzBuffers"],
        { npzBuffers },
        `
        import numpy as np
        import io

        ret = []
        for npzBuffer in npzBuffers:
            npz_file = io.BytesIO(npzBuffer.to_py())
            npz_data = np.load(npz_file, allow_pickle=True)
            ret.append([
                [], # npz_data['arr_0'].tolist(),
                [], # npz_data['arr_1'].tolist(),
                [], #npz_data['arr_2'].tolist(),
                npz_data['arr_3'].tolist(),
                npz_data['arr_4'].tolist()
            ])
        ret
    `,
    );
};

type ChartLine = {
    name: string;
    identifier: string;
    data: number[];
};
type ChartProps = {
    title: string;
    lines: ChartLine[];
    setRowAttrs: React.Dispatch<React.SetStateAction<DataRowAttribute[]>>;
    yLogScale?: boolean;
    tickInterval?: number;
};

const Chart: React.FC<ChartProps> = React.memo(({ yLogScale, title, tickInterval, lines, setRowAttrs }) => {
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    const len = lines.map((l) => l.data.length).max();

    const series = lines.map((l) => ({
        identifier: l.identifier,
        name: `Series ${l.name}`,
        data: l.data.map((y, j) => [j, y]).filter(([j]) => j < 1000 || j % Math.max(1, Math.floor(len / 1000)) === 0),
        type: "line",
        marker: {
            enabled: false,
        },
    }));

    const updateDimensions = (clientWidth: number, clientHeight: number) => {
        const maxWidth = clientWidth * 0.9;
        const maxHeight = clientHeight * 0.9;

        const width = Math.min(maxWidth, (maxHeight / 9) * 16);
        const height = Math.min(maxHeight, (maxWidth / 16) * 9);

        setDimensions({ width, height });
    };

    const options = {
        chart: {
            type: "line",
            width: dimensions.width,
            height: dimensions.height,
            zooming: {
                type: "x",
            },
        },
        title: {
            text: title,
        },
        xAxis: {
            type: "linear",
        },
        yAxis: {
            type: yLogScale ? "logarithmic" : undefined,
            title: {
                text: "Value",
            },
            tickInterval: tickInterval,
        },
        tooltip: {
            formatter: function (this: { x: number; y: number }) {
                return `n iters: ${this.x}, Y: ${this.y.toFixed(2)}`;
            },
        },
        series: series,
        plotOptions: {
            series: {
                turboThreshold: 0, // To handle large datasets
            },
            line: {
                events: {
                    mouseOver: function (this: { userOptions: { identifier: string } }) {
                        setRowAttrs((current) => current.map((x) => (x.identifier === this.userOptions.identifier ? { ...x, isFocused: true } : { ...x, isFocused: false })));
                    },
                    mouseOut: function (this: { userOptions: { identifier: string } }) {
                        setRowAttrs((current) => current.map((x) => ({ ...x, isFocused: false })));
                    },
                },
            },
        },
    };

    return (
        <ReactiveDiv resizeHandler={updateDimensions} style={{ ...centeringProps }}>
            <HighchartsReact highcharts={Highcharts} options={options} />
        </ReactiveDiv>
    );
});

type ResultChartProps = {
    rowAttrs: DataRowAttribute[];
    setRowAttrs: React.Dispatch<React.SetStateAction<DataRowAttribute[]>>;
    filterRowAttr: (rowAttr: DataRowAttribute) => boolean;
};

const ResultChart: React.FC<ResultChartProps> = ({ rowAttrs, setRowAttrs, filterRowAttr }) => {
    const startTime = useMemo(() => performance.now(), []);
    const [psnrChartLines, setPsnrChartLines] = useState<ChartLine[] | null>(null);
    const [objectiveChartLines, setObjectiveChartLines] = useState<ChartLine[] | null>(null);
    const pythonRunner = usePythonRunner();
    const [calculatedResults, setCalculatedResults] = useState<number[][][] | null>(null);

    useEffect(() => {
        const filePaths = rowAttrs.map((x) => x.filePath);
        loadData(pythonRunner, filePaths).then(setCalculatedResults);
    }, [pythonRunner]);

    useEffect(() => {
        if (calculatedResults === null) return;
        const psnrData = calculatedResults.map((xs) => xs[4]);
        const objectiveData = calculatedResults.map((xs) => xs[3]);

        const newPsnrChartLines = rowAttrs
            .zip(psnrData)
            .filter(([attr]) => filterRowAttr(attr))
            .map(([attr, xs]) => ({ name: String(attr.numId), identifier: attr.identifier, data: xs }));
        const newObjeciveChartLines = rowAttrs
            .zip(objectiveData)
            .filter(([attr]) => filterRowAttr(attr))
            .map(([attr, xs]) => ({ name: String(attr.numId), identifier: attr.identifier, data: xs }));
        setPsnrChartLines(newPsnrChartLines);
        setObjectiveChartLines(newObjeciveChartLines);
        const newRowAttrs = rowAttrs.zip(psnrData).map(([attr, xs]) => ({
            ...attr,
            isMatchedPlotCondition: filterRowAttr(attr),
            nIters: xs.length,
        }));
        setRowAttrs(newRowAttrs);
    }, [calculatedResults, filterRowAttr]);

    if (!pythonRunner.status.isAvailable) return <LoadingScreen message={pythonRunner.status.message} />;
    if (psnrChartLines === null || objectiveChartLines === null) return <LoadingScreen message={"計算中です..."} />;
    console.log(`elapsed: ${performance.now() - startTime}[ms]`);
    return (
        <div style={centeringProps}>
            <div style={{ height: "50%", width: "100%" }}>
                <Chart title={"PSNR"} lines={psnrChartLines} setRowAttrs={setRowAttrs} tickInterval={0.5} />;
            </div>
            <div style={{ height: "50%", width: "100%" }}>
                <Chart title={"log10 Objective"} lines={objectiveChartLines} setRowAttrs={setRowAttrs} yLogScale />;
            </div>
        </div>
    );
};

const useStyles = makeStyles({
    container: {
        display: "flex",
        flexDirection: "row",
        height: "100vh",
        // backgroundColor: tokens.colorNeutralBackground3,
    },
    summaryContainer: {
        display: "flex",
        flexDirection: "column",
        width: "320px",
        padding: tokens.spacingHorizontalM,
        overflowY: "auto",
        // backgroundColor: tokens.colorNeutralBackground6,
        border: `1px solid ${tokens.colorNeutralStroke1}`,
        boxShadow: tokens.shadow16,
        gap: tokens.spacingVerticalS,
    },
    detailsContainer: {
        flex: 1,
        padding: tokens.spacingHorizontalM,
        overflowY: "auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
    },
    card: {
        flexShrink: 0, // カードが縮まないようにする
        ...shorthands.padding(tokens.spacingHorizontalS, tokens.spacingVerticalS),
        borderRadius: tokens.borderRadiusMedium,
        backgroundColor: tokens.colorNeutralBackground1,
        boxShadow: tokens.shadow8,
        "& *": {
            fontFamily: `"M PLUS 1p"`,
            // fontSize: '1rem',
            lineHeight: 0.5,
            letterSpacing: 0,
        },
    },

    cardHeader: {
        display: "flex",
        alignItems: "center",
    },
    cardFooter: {
        display: "flex",
        justifyContent: "space-between",
    },

    icon: {
        marginRight: tokens.spacingHorizontalS,
    },

    gridContainer: {
        display: "grid",
        position: "relative",
        overflowY: "scroll",
        // margin: "auto",
        // padding: "auto",
        // paddingRight: '10px',
        // borderLeft: `1px solid black`,
        userSelect: "none",
        "& *": {
            fontFamily: `"M PLUS 1p"`,
        },
    },

    gridCell: {
        padding: "0px 4px",
        display: "flex",
        textAlign: "center",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "row",
        // backgroundColor: "#f3f2f1",
        border: `0.1px solid #d9d9d9`,
        borderRight: `1px solid black`,
    },

    gridCellLast: {
        // borderRight: 'none',
        borderRight: `0.01px solid #d9d9d9`,
    },

    focused: {
        // border: `2px solid #007bbb`,
        // borderRight: `2px solid #007bbb`,
        // outlineOffset: '-4px',
        position: "relative",
        "&::before": {
            content: '""',
            position: "absolute",
            top: "-1px",
            left: "-1px",
            right: "-1px",
            bottom: "-1px",
            border: `2px solid #007bbb`,
            pointerEvents: "none", // クリックイベントを無視
            zIndex: 1, // コンテンツより前面に描画
        },
    },

    selected: {
        backgroundColor: "#eaedf7",
    },

    rowHeaderCell: {
        padding: "0px 5px",
        backgroundColor: "#e1dfdd",
        fontWeight: "bold",
        borderBottom: `1.5px solid black`,
    },

    colHeaderCell: {
        padding: "0px 5px",
        backgroundColor: "#e1dfdd",
        fontWeight: "bold",
        borderRight: `1.5px solid black`,
        // textAlign: "left",
        // alignItems: "left",
        justifyContent: "right",
    },
    nonActiveCell: {
        backgroundColor: "#c0c0c0",
    },

    checkbox: {
        display: "flex",
        justifyContent: "center",
        position: "relative",
        "& > input": {
            position: "absolute",
            left: "auto",
        },
        "& *": {
            margin: "auto",
        },
    },
});

type CellPos = { row: number; col: number };
type CellRange = {
    tl: CellPos;
    br: CellPos;
};

const createCellRangeFrom2Point = (a: CellPos, b: CellPos): CellRange => {
    const tl = { row: Math.min(a.row, b.row), col: Math.min(a.col, b.col) };
    const br = { row: Math.max(a.row, b.row), col: Math.max(a.col, b.col) };
    return { tl, br };
};
const inRange = (pos: CellPos, range: CellRange): boolean => {
    const { tl, br } = range;
    return tl.row <= pos.row && pos.row <= br.row && tl.col <= pos.col && pos.col <= br.col;
};

type CommonCellProps = {
    nColumns?: number;
    focusedPos?: CellPos;
    selectedRange?: CellRange;
    onMouseDown?: (pos: CellPos) => void;
    onMouseEnter?: (pos: CellPos) => void;
    onMouseUp?: (pos: CellPos) => void;
};
const CommonCellPropsContext = createContext<CommonCellProps>({});

type CellProps = {
    cellPos: CellPos;
    rowHeader?: boolean;
    colHeader?: boolean;
    nonActive?: boolean;
    disableMouseEvents?: boolean;
    divProps?: React.DetailedHTMLProps<React.HTMLAttributes<HTMLDivElement>, HTMLDivElement>;
    children?: React.ReactNode;
};
const Cell: React.FC<CellProps> = ({ cellPos, rowHeader, colHeader, nonActive, disableMouseEvents, divProps, children }) => {
    const { onMouseDown, onMouseEnter, onMouseUp, focusedPos, selectedRange, nColumns } = useContext(CommonCellPropsContext);
    const styles = useStyles();
    const classNames = [styles.gridCell];

    const isFocused = focusedPos?.row === cellPos.row && focusedPos?.col === cellPos.col;
    const isInSelectedRange = selectedRange && inRange(cellPos, selectedRange);
    if (isFocused) classNames.push(styles.focused);
    if (isInSelectedRange) classNames.push(styles.selected);

    const isLast = nColumns !== undefined && cellPos.col === nColumns - 1;
    if (isLast) classNames.push(styles.gridCellLast);
    if (rowHeader) classNames.push(styles.rowHeaderCell);
    if (colHeader) classNames.push(styles.colHeaderCell);
    if (nonActive) classNames.push(styles.nonActiveCell);

    const onMouseDownWrapper = (_e: React.MouseEvent<HTMLDivElement>) => {
        if (disableMouseEvents) return;
        onMouseDown?.(cellPos);
    };
    const onMouseEnterWrapper = (_e: React.MouseEvent<HTMLDivElement>) => {
        if (disableMouseEvents) return;
        onMouseEnter?.(cellPos);
    };
    const onMouseUpWrapper = (_e: React.MouseEvent<HTMLDivElement>) => {
        if (disableMouseEvents) return;
        onMouseUp?.(cellPos);
    };

    const className = classNames.join(" ");
    return (
        <div className={className} onMouseDown={onMouseDownWrapper} onMouseEnter={onMouseEnterWrapper} onMouseUp={onMouseUpWrapper} {...divProps}>
            {children}
        </div>
    );
};

type GridRowProps = {
    rowIdx: number;
    cellContents: React.ReactNode[];
    isHeader?: boolean;
    disableMouseEvents?: boolean;
};
const GridRow: React.FC<GridRowProps> = ({ rowIdx, cellContents, isHeader, disableMouseEvents }) => {
    return (
        <>
            {cellContents.map((content, colIdx) => (
                <Cell key={colIdx} cellPos={{ row: rowIdx, col: colIdx }} rowHeader={isHeader} disableMouseEvents={disableMouseEvents}>
                    {content}
                </Cell>
            ))}
        </>
    );
};

type RowProps = {
    rowIdx: number;
    attr: DataRowAttribute;
};
const Row: React.FC<RowProps> = ({ rowIdx, attr }) => {
    const styles = useStyles();
    const isFocusedRow = attr.isFocused;

    const commonDivProps = {
        style: {
            backgroundColor: isFocusedRow ? "#f6bfbc" : undefined,
        },
    };
    const optionStr = attr.option?.filter ? "F" : "";

    let colIdx = 0;
    return (
        <>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps} disableMouseEvents colHeader>
                <Text style={{ fontFamily: "Menlo" }}>{attr.numId}</Text>
            </Cell>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps}>
                <Checkbox className={styles.checkbox} />
            </Cell>
            {/*<Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps} disableMouseEvents>*/}
            {/*    <Button style={{padding: 'auto', margin: 'auto', minWidth: "auto", width: "90%", height: "90%"}}/>*/}
            {/*</Cell>*/}
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps}>
                {attr.method}
            </Cell>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps}>
                {attr.nShots}
            </Cell>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps}>
                {attr.gamma1.toExponential(0)}
            </Cell>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps} nonActive={attr.gamma2 === undefined}>
                {attr.gamma2}
            </Cell>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps}>
                {attr.noiseSigma}
            </Cell>
            <Cell cellPos={{ row: rowIdx, col: colIdx++ }} divProps={commonDivProps} nonActive={optionStr === ""}>
                {optionStr}
            </Cell>
        </>
    );
};

type DataRowAttribute = {
    numId: number;
    identifier: string;
    filePath: string;
    method: string;
    nShots: number;
    nIters: number | undefined;
    gamma1: number;
    gamma2: number | undefined;
    noiseSigma: number;
    option?: {
        filter?: boolean;
        useNesterov?: boolean;
    };
    plotColor: string | undefined;

    isMatchedPlotCondition: boolean;
    forcePlotFlag: boolean;
    isFocused: boolean;
};

const initialFilteringConditionSource = "nShots == 24 && noiseSigma == 0 && (gamma2 == 10 || gamma2 == undefined)";
type FilteringConditionInputProps = {
    setFilterRowAttr: React.Dispatch<React.SetStateAction<(attr: DataRowAttribute) => boolean>>;
};
const FilteringConditionInput: React.FC<FilteringConditionInputProps> = ({ setFilterRowAttr }) => {
    const [filteringConditionSource, setFilteringConditionSource] = React.useState(initialFilteringConditionSource);
    const inputRef = React.useRef<HTMLInputElement>(null);

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        console.log("set");
        setFilteringConditionSource(event.target.value);
    };
    const setFilteringCondition = (strExpr: string) => {
        const source = `return ${strExpr}`;
        try {
            const rawFunc = Function("nShots", "gamma1", "gamma2", "noiseSigma", "method", "attr", source);
            const func = (attr: DataRowAttribute) => {
                return rawFunc(attr.nShots, attr.gamma1, attr.gamma2, attr.noiseSigma, attr.method, attr);
            };
            setFilterRowAttr(() => func);
        } catch {
            setFilteringConditionSource("nShots == 24 && noiseSigma == 0");
            setFilterRowAttr(() => (attr: DataRowAttribute) => attr.nShots === 24 && attr.noiseSigma === 0);
        }
    };
    const handleBlur = (event: React.FocusEvent<HTMLInputElement>) => {
        setFilteringCondition(event.target.value);
    };

    useEffect(() => {
        setFilteringCondition(initialFilteringConditionSource);
    }, []);

    return (
        <div style={{ height: "50px", width: "100%" }}>
            <div style={{ ...centeringProps, flexDirection: "row" }}>
                <Input
                    id="textBox"
                    value={filteringConditionSource}
                    style={{ width: "80%" }}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    onKeyDown={(event) => {
                        if (event.key === "Enter") {
                            inputRef.current!.blur();
                        }
                    }}
                    ref={inputRef}
                />
                <Button
                    onClick={() => {
                        setFilteringConditionSource(initialFilteringConditionSource);
                        setFilteringCondition(initialFilteringConditionSource);
                    }}
                    style={{ padding: "0", margin: "0", marginLeft: "10px", minWidth: "auto", width: "50px" }}
                >
                    reset
                </Button>
            </div>
        </div>
    );
};

// type ChartKindDrpoDownProps = {
//     setChartKind: React.Dispatch<React.SetStateAction<number>>;
// };
// const ChartKindDropdown: React.FC = () => {
//     const [selected, setSelected] = React.useState("");
//
//     const handleSelect = (event, data) => {
//         setSelected(data.optionValue);
//         console.log("Selected:", data.optionValue);
//     };
//
//     return (
//         <div>
//             <Label htmlFor="dropdown">Choose an option:</Label>
//             <Dropdown id="dropdown" placeholder="Select an option" value={selected} onOptionSelect={handleSelect}>
//                 <Option value="option1">objective</Option>
//                 <Option value="option2">PSNR</Option>
//             </Dropdown>
//         </div>
//     );
// };

type DataSummaryPanelProps = {
    rowAttrs: DataRowAttribute[];
    setRowAttrs: React.Dispatch<React.SetStateAction<DataRowAttribute[]>>;
    setFilterRowAttr: React.Dispatch<React.SetStateAction<(attr: DataRowAttribute) => boolean>>;
};
const DataSummaryPanel: React.FC<DataSummaryPanelProps> = ({ rowAttrs, setFilterRowAttr }) => {
    const styles = useStyles();
    const headers = ["ID", "Plot", "Method", "n shots", "γ1", "γ2", "noise σ", "ex"];
    const nHeaders = headers.length;

    const gridRef = useRef<HTMLDivElement>(null);
    const [focusedPos, setFocusedPos] = React.useState<{ row: number; col: number } | undefined>(undefined);
    const [selectedRange, setSelectedRange] = React.useState<CellRange | undefined>(undefined);
    const [isDragging, setIsDragging] = React.useState(false);

    const handleMouseDown = (pos: CellPos) => {
        // console.log("down: ", pos);
        setFocusedPos(pos);
        setIsDragging(true);
        setSelectedRange(undefined);
        // const key = `${row}-${col}`;
        // const newSelection = new Set([key]);
        // setSelection(newSelection);
    };

    const handleMouseEnter = (pos: CellPos) => {
        console.log("enter: ", pos);
        if (focusedPos === undefined) return;
        if (!isDragging) return;
        setSelectedRange(createCellRangeFrom2Point(focusedPos, pos));
        // const key = `${row}-${col}`;
        // setSelection((prev) => {
        //     const newSelection = new Set(prev);
        //     newSelection.add(key);
        //     return newSelection;
        // });
    };

    const handleMouseUpOnCell = () => {
        // console.log("up");
        // 選択が完了したら、コピー可能なデータを準備するなどの処理を行う
    };
    const handleMouseUpAlways = () => {
        // console.log("up always");
        setIsDragging(false);
    };
    const handleClickOutside = (event: MouseEvent) => {
        if (gridRef.current && !gridRef.current.contains(event.target as Node)) {
            setFocusedPos(undefined);
            setSelectedRange(undefined);
        }
    };

    useEffect(() => {
        addEventListener("mouseup", handleMouseUpAlways);
        return () => removeEventListener("mouseup", handleMouseUpAlways);
    }, []);
    useEffect(() => {
        addEventListener("mousedown", handleClickOutside);
        return () => removeEventListener("mousedown", handleClickOutside);
    }, []);

    // const handleCopy = () => {
    //     const selectedData = Array.from(selectedRange)
    //         .map((key) => {
    //             const [row, col] = key.split('-').map(Number);
    //             return `R${row + 1}C${col + 1}`; // サンプルデータ
    //         })
    //         .join('\n');
    //
    //     navigator.clipboard.writeText(selectedData).then(() => {
    //         alert('Copied to clipboard!');
    //     });
    // };

    const commonCellProps: CommonCellProps = {
        nColumns: nHeaders,
        focusedPos,
        selectedRange,
        onMouseDown: handleMouseDown,
        onMouseEnter: handleMouseEnter,
        onMouseUp: handleMouseUpOnCell,
    };

    return (
        <>
            {/*<ChartKindDropdown />*/}
            <FilteringConditionInput setFilterRowAttr={setFilterRowAttr} />
            <CommonCellPropsContext.Provider value={commonCellProps}>
                <div ref={gridRef} className={styles.gridContainer} style={{ gridTemplateColumns: `repeat(${nHeaders}, auto)` }}>
                    <GridRow rowIdx={-1} cellContents={headers} isHeader disableMouseEvents />
                    {rowAttrs.map((attr, rowIdx) => (
                        <Row key={rowIdx} attr={attr} rowIdx={rowIdx} />
                    ))}
                </div>
            </CommonCellPropsContext.Provider>
        </>
    );
};

const initialRowData = DataAttributes.sort((a, b) => {
    if (a.nShots !== b.nShots) return -(a.nShots - b.nShots);
    if (a.noiseSigma !== b.noiseSigma) return a.noiseSigma - b.noiseSigma;
    if (a.gamma2 !== b.gamma2) return (a.gamma2 ?? 0) - (b.gamma2 ?? 0);
    if (a.option?.useNesterov !== b.option?.useNesterov) return (a.option?.useNesterov ? 1 : 0) - (b.option?.useNesterov ? 1 : 0);
    if (a.option?.filter !== b.option?.filter) return (a.option?.filter ? 1 : 0) - (b.option?.filter ? 1 : 0);
    return 0;
}).map(
    (attr, idx): DataRowAttribute => ({
        numId: idx,
        identifier: attr.filePath,
        filePath: attr.filePath,
        method: (attr.method === "gradient" ? "GD" : "PDS") + (attr.option?.useNesterov ? "+NA" : ""),
        nShots: attr.nShots,
        nIters: undefined,
        gamma1: attr.gamma1,
        gamma2: attr.gamma2 ?? undefined,
        noiseSigma: attr.noiseSigma,
        option: attr.option,
        plotColor: undefined,
        isMatchedPlotCondition: false,
        forcePlotFlag: false,
        isFocused: false,
    }),
);

const App: React.FC = () => {
    const [rowAttrs, setRowAttrs] = useState<DataRowAttribute[]>(initialRowData);
    const [filterRowAttr, setFilterRowAttr] = useState<(attr: DataRowAttribute) => boolean>(() => (rowAttr: DataRowAttribute): boolean => {
        return rowAttr.nShots === 10 && rowAttr.noiseSigma === 0;
    });

    return (
        <FluentProvider theme={webLightTheme}>
            <div style={{ width: "100vw", height: "100vh" }}>
                <div style={{ ...centeringProps, flexDirection: "row" }}>
                    <div style={{ ...centeringProps, width: "35%", height: "100%" }}>
                        <DataSummaryPanel rowAttrs={rowAttrs} setRowAttrs={setRowAttrs} setFilterRowAttr={setFilterRowAttr} />
                    </div>
                    <div style={{ width: "65%", height: "100%" }}>
                        <ResultChart rowAttrs={rowAttrs} setRowAttrs={setRowAttrs} filterRowAttr={filterRowAttr} />
                    </div>
                </div>
            </div>
        </FluentProvider>
    );
};

export default App;
