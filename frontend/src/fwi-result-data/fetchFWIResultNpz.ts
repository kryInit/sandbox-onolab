export const fetchFWIResultNpz = async (fileName: string): Promise<ArrayBuffer> => {
    const targetUrl = `${window.location.href}data/${fileName}`;
    const response = await fetch(targetUrl);
    if (!response.ok) {
        throw new Error("ファイルのダウンロードに失敗しました");
    }
    return await response.arrayBuffer();
};