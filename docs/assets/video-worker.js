// 此文件为 Web Worker 示例（可选扩展）
self.onmessage = async (e) => {
    const { videoData, roi, format } = e.data;
    const { createFFmpeg, fetchFile } = self.FFmpeg;
    const ffmpeg = createFFmpeg({ log: true });
    
    await ffmpeg.load();
    ffmpeg.FS('writeFile', 'input.mp4', videoData);
    
    await ffmpeg.run(
        '-i', 'input.mp4',
        '-vf', `crop=${roi.width}:${roi.height}:${roi.x}:${roi.y}`,
        '-c:v', 'libx264',
        `output.${format}`
    );
    
    const output = ffmpeg.FS('readFile', `output.${format}`);
    self.postMessage({ output });
};