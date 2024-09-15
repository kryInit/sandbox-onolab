export type DataAttribute = {
    filePath: string;
    nShots: number;
    gamma1: number;
    noiseSigma: number;
    option?: {
        filter?: boolean;
        useNesterov?: boolean;
    };
} & (
    | {
          method: "pds";
          gamma2: number;
      }
    | {
          method: "gradient";
          gamma2: null;
      }
);
const createDataAttributeFromFileName = (fileName: string): DataAttribute => {
    // "2024-08-13_00-05-55,nshots=24,gamma1=1e-05,gamma2=1,niters=100000,sigma=1.npz",形式のデータを解析する
    const rawContent = fileName.split(",");

    const { nShots, gamma1, gamma2, noiseSigma } = (() => {
        if (rawContent.length === 4) {
            // "2024-07-30_16-04-10,nshots=10,gamma1=1e-05,gamma2=None.npz"形式のデータを解析する
            const [_rawTime, rawNShots, rawGamma1, rawGamma2] = rawContent;
            const nShots = Number(rawNShots.split("=")[1]);
            const gamma1 = Number(rawGamma1.split("=")[1]);
            const strGamma2 = rawGamma2.split("=")[1].slice(0, -4);
            const gamma2 = strGamma2 === "None" ? null : Number(strGamma2);
            const noiseSigma = 0;
            return { nShots, gamma1, gamma2, noiseSigma };
        }
        if (rawContent.length === 6) {
            const [_rawTime, rawNShots, rawGamma1, rawGamma2, _rawNIters, rawSigma] = rawContent;
            const nShots = Number(rawNShots.split("=")[1]);
            const gamma1 = Number(rawGamma1.split("=")[1]);
            const gamma2 = rawGamma2.split("=")[1] === "None" ? null : Number(rawGamma2.split("=")[1]);
            const noiseSigma = Number(rawSigma.split("=")[1].split(".")[0]);

            return { nShots, gamma1, gamma2, noiseSigma };
        }
        if (rawContent.length === 8) {
            const [_rawTime, _method, rawNShots, rawGamma1, rawGamma2, _rawNIters, rawSigma] = rawContent;
            const nShots = Number(rawNShots.split("=")[1]);
            const gamma1 = Number(rawGamma1.split("=")[1]);
            const gamma2 = rawGamma2.split("=")[1] === "None" ? null : Number(rawGamma2.split("=")[1]);
            const noiseSigma = Number(rawSigma.split("=")[1].split(".")[0]);

            return { nShots, gamma1, gamma2, noiseSigma };
        }
        throw new Error("unimplemented");
    })();

    if (gamma2 === null) return { filePath: fileName, nShots, gamma1, gamma2, noiseSigma, method: "gradient" };
    return { filePath: fileName, nShots, gamma1, gamma2, noiseSigma, method: "pds" };
};

export const DataAttributes: DataAttribute[] = [
    {
        ...createDataAttributeFromFileName("2024-08-12_18-16-59,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=1.npz"),
        option: {
            filter: true,
        },
    },
    {
        ...createDataAttributeFromFileName("2024-08-13_00-05-55,nshots=24,gamma1=1e-05,gamma2=1,niters=100000,sigma=1.npz"),
        option: {
            filter: true,
        },
    },
    // PDS + NA はノイズになるので一旦削除
    // {
    //     ...createDataAttributeFromFileName('2024-08-18_23-33-51,pds_nesterov,nshots=24,gamma1=1e-05,gamma2=10,niters=10000,sigma=1,.npz'),
    //     option: {
    //         useNesterov: true
    //     },
    // },
    // {
    //     ...createDataAttributeFromFileName('2024-08-18_22-59-21,pds_nesterov,nshots=24,gamma1=1e-05,gamma2=10,niters=10000,sigma=0,.npz'),
    //     option: {
    //         useNesterov: true
    //     },
    // },
    {
        ...createDataAttributeFromFileName("2024-08-18_06-16-28,gd_nesterov,nshots=24,gamma1=1e-05,gamma2=None,niters=419,sigma=1,.npz"),
        option: {
            useNesterov: true,
        },
    },
    {
        ...createDataAttributeFromFileName("2024-08-18_05-35-06,gd_nesterov,nshots=24,gamma1=1e-05,gamma2=None,niters=542,sigma=0,.npz"),
        option: {
            useNesterov: true,
        },
    },
    ...[
        "2024-08-06_20-03-13,nshots=24,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=0.npz",
        "2024-08-06_21-48-41,nshots=24,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=0.npz",
        "2024-08-06_23-35-13,nshots=24,gamma1=1e-05,gamma2=0.001,niters=30000,sigma=0.npz",
        "2024-08-08_04-32-59,nshots=10,gamma1=1e-05,gamma2=None,niters=30000,sigma=0.npz",
        "2024-08-08_05-23-29,nshots=10,gamma1=1e-05,gamma2=0.002,niters=30000,sigma=0.npz",
        "2024-08-08_06-14-00,nshots=10,gamma1=1e-05,gamma2=0.003,niters=30000,sigma=0.npz",
        "2024-08-08_07-04-55,nshots=10,gamma1=1e-05,gamma2=0.001,niters=30000,sigma=0.npz",

        "2024-08-13_06-10-57,nshots=24,gamma1=1e-05,gamma2=10,niters=100000,sigma=1.npz",
        "2024-08-13_17-47-10,nshots=24,gamma1=1e-05,gamma2=100,niters=100000,sigma=1.npz",
        "2024-08-13_11-58-54,nshots=24,gamma1=1e-05,gamma2=0.1,niters=100000,sigma=1.npz",
        "2024-08-13_23-35-23,nshots=24,gamma1=1e-05,gamma2=0.01,niters=100000,sigma=1.npz",
        "2024-08-14_02-38-12,nshots=24,gamma1=1e-05,gamma2=None,niters=24000,sigma=1.npz",
        "2024-08-16_17-40-06,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=2.npz",
        "2024-08-16_23-27-41,nshots=24,gamma1=1e-05,gamma2=10,niters=100000,sigma=2.npz",
        "2024-08-17_05-02-42,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=5.npz",
        "2024-08-17_10-52-22,nshots=24,gamma1=1e-05,gamma2=10,niters=100000,sigma=5.npz",
        "2024-08-17_16-34-37,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=3.npz",
        "2024-08-17_22-25-07,nshots=24,gamma1=1e-05,gamma2=10,niters=100000,sigma=3.npz",

        "2024-08-19_11-04-32,pds,nshots=24,gamma1=1e-05,gamma2=10,niters=100000,sigma=0,.npz",
        "2024-08-19_05-15-21,gradient,nshots=24,gamma1=1e-05,gamma2=None,niters=100000,sigma=0,.npz",
    ].map(createDataAttributeFromFileName),
];
