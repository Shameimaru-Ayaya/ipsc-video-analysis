const videoInput = document.getElementById('videoInput');
const previewCanvas = document.getElementById('previewCanvas');
const processBtn = document.getElementById('processBtn');
const formatSelect = document.getElementById('formatSelect');
const progressBar = document.getElementById('progress');

let video = document.createElement('video');
let ctx = previewCanvas.getContext('2d');
let selectedROI = null;

// 初始化 FFmpeg
const { createFFmpeg, fetchFile } = FFmpeg;
const ffmpeg = createFFmpeg({ log: true, progress: ({ ratio }) => {
    progressBar.value = ratio * 100;
}});

// 加载视频文件
videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    
    video.src = url;
    video.addEventListener('loadedmetadata', () => {
        previewCanvas.width = video.videoWidth;
        previewCanvas.height = video.videoHeight;
        drawVideoFrame();
    });
});

// 绘制视频帧
function drawVideoFrame() {
    ctx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
    if (!video.paused) requestAnimationFrame(drawVideoFrame);
}

// ROI 选择交互
let isDrawing = false;
let startX, startY;

previewCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = previewCanvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
});

previewCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = previewCanvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // 绘制 ROI 框
    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    ctx.drawImage(video, 0, 0);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, endX - startX, endY - startY);
});

previewCanvas.addEventListener('mouseup', (e) => {
    isDrawing = false;
    const rect = previewCanvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // 计算实际 ROI 坐标
    selectedROI = {
        x: Math.min(startX, endX),
        y: Math.min(startY, endY),
        width: Math.abs(endX - startX),
        height: Math.abs(endY - startY)
    };
    processBtn.disabled = false;
});

// 处理视频
processBtn.addEventListener('click', async () => {
    if (!selectedROI) return;
    
    processBtn.disabled = true;
    await ffmpeg.load();
    
    // 写入输入文件
    const videoData = await fetchFile(videoInput.files[0]);
    ffmpeg.FS('writeFile', 'input.mp4', videoData);
    
    // 执行裁剪命令
    await ffmpeg.run(
        '-i', 'input.mp4',
        '-vf', `crop=${selectedROI.width}:${selectedROI.height}:${selectedROI.x}:${selectedROI.y}`,
        '-c:v', 'libx264',
        `output.${formatSelect.value}`
    );
    
    // 生成下载链接
    const output = ffmpeg.FS('readFile', `output.${formatSelect.value}`);
    const blob = new Blob([output.buffer], { type: `video/${formatSelect.value}` });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `cropped_video.${formatSelect.value}`;
    a.click();
    
    processBtn.disabled = false;
    progressBar.value = 0;
});