# ComfyUI Remotion Captions

独立的 Remotion 字幕渲染子项目（无需依赖 reference），提供 TikTok 风格逐词高亮字幕渲染。

## 依赖
- Node.js 18+（已安装 `node` / `npx`）
- 首次运行会在本目录执行 `npm install`

## 项目结构
- `src/index.ts`, `src/root.tsx`：Remotion 入口与 Composition 定义
- `src/tiktok/*`：TikTok 风格字幕组件
- `tools/convert-srt-to-json.mjs`：SRT 转 Caption[] JSON 脚本
- `remotion_captions_node.py`：ComfyUI 自定义节点

## 节点使用
在 ComfyUI 中选择：`字幕/Remotion / 🎬 Render TikTok Captions (Remotion)`

输入：
- `video_path`：原视频路径（mp4/mov/webm）
- `srt_path`：SRT 字幕路径
- 可选：
  - `output_path`：输出视频路径（默认写入项目 `output/remotion_captions.mp4`）
  - `switchEveryMs`：切页窗口毫秒数（默认 1200）
  - `highlightColor`：高亮颜色（默认 `#39E508`）

输出：
- `video`：渲染成品视频路径
- `log`：渲染日志

## 渲染说明
节点内部流程：
1. 检查 Node 环境，首次运行自动安装依赖
2. 将视频拷贝到临时 public 目录为 `video.mp4`
3. 调用 `tools/convert-srt-to-json.mjs` 生成 `video.json`
4. 调用 `npx remotion render src/index.ts CaptionedVideo` 进行渲染

如需自定义样式，可调整：
- `src/tiktok/page.tsx`：字体、描边、文本位置
- `src/root.tsx`：Composition 尺寸（默认 1080×1920）与默认 props
