const { createFFmpeg, fetchFile } = FFmpeg;
const ffmpeg = createFFmpeg({ 
    log: true,
    progress: ({ ratio }) => {
        document.getElementById('progressBar').value = ratio * 100;
    }
});

// ROI 坐标转换（Canvas ↔ 原始视频）
function convertROICoordinates(canvasROI, videoWidth, videoHeight) {
    const canvas = document.getElementById('previewCanvas');
    const scaleX = videoWidth / canvas.width;
    const scaleY = videoHeight / canvas.height;
    return {
        x: Math.round(canvasROI.x * scaleX),
        y: Math.round(canvasROI.y * scaleY),
        width: Math.round(canvasROI.width * scaleX),
        height: Math.round(canvasROI.height * scaleY)
    };
}

// 视频处理
document.getElementById('processBtn').addEventListener('click', async () => {
    const videoFile = document.getElementById('videoInput').files[0];
    const format = document.getElementById('formatSelect').value;
    
    await ffmpeg.load();
    ffmpeg.FS('writeFile', 'input.mp4', await fetchFile(videoFile));
    
    // 动态配置编码参数
    const encoderParams = {
        'mp4': ['-c:v', 'libx264', '-preset', 'fast'],
        'avi': ['-c:v', 'mpeg4', '-qscale:v', '3']
    };
    
    await ffmpeg.run(
        '-i', 'input.mp4',
        '-vf', `crop=${roi.width}:${roi.height}:${roi.x}:${roi.y}`,
        ...encoderParams[format],
        `output.${format}`
    );

    // 生成下载链接
    const output = ffmpeg.FS('readFile', `output.${format}`);
    const blob = new Blob([output.buffer], { type: `video/${format}` });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `cropped.${format}`;
    a.click();
});