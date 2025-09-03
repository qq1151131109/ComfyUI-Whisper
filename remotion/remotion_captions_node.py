import json
import os
import shutil
import subprocess
import tempfile
from typing import Tuple


class RenderRemotionCaptionsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"multiline": False, "default": ""}),
                "srt_path": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "output_path": ("STRING", {"multiline": False, "default": os.path.abspath(os.path.join(os.getcwd(), "output", "remotion_captions.mp4"))}),
                "switchEveryMs": ("INT", {"default": 1200, "min": 100, "max": 10000, "step": 50}),
                "highlightColor": ("STRING", {"default": "#39E508"}),
                "style": (["neon", "boxed", "pop", "karaoke"], {"default": "neon"}),
                "timeout_ms": ("INT", {"default": 180000, "min": 10000, "max": 1200000, "step": 1000}),
                "concurrency": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video", "log")
    FUNCTION = "render"
    CATEGORY = "字幕/Remotion"

    def _ensure_node_dependencies(self, project_dir: str) -> str:
        node_modules = os.path.join(project_dir, "node_modules")
        log_msgs = []
        if not os.path.isdir(node_modules):
            log_msgs.append("Installing Node dependencies (first run)...")
            try:
                subprocess.run(
                    ["npm", "install", "--silent", "--no-fund", "--no-audit"],
                    cwd=project_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                log_msgs.append("npm install completed.")
            except Exception as e:
                log_msgs.append(f"npm install failed: {e}")
                raise
        return "\n".join(log_msgs)

    def _check_node(self) -> None:
        try:
            subprocess.run(["node", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            subprocess.run(["npx", "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception:
            raise RuntimeError("Node.js / npx 未安装或不可用，请先安装 Node.js 后重试。")

    def render(
        self,
        video_path: str,
        srt_path: str,
        output_path: str = None,
        switchEveryMs: int = 1200,
        highlightColor: str = "#39E508",
        style: str = "neon",
        timeout_ms: int = 180000,
        concurrency: int = 2,
    ) -> Tuple[str, str]:
        logs = []

        if not video_path or not os.path.isfile(video_path):
            raise ValueError(f"无效的视频路径: {video_path}")
        if not srt_path or not os.path.isfile(srt_path):
            raise ValueError(f"无效的SRT路径: {srt_path}")

        project_dir = os.path.dirname(__file__)
        entry_file = os.path.join(project_dir, "src", "index.ts")
        if not os.path.isfile(entry_file):
            raise RuntimeError("Remotion 项目入口不存在，安装可能不完整。")

        output_path = output_path or os.path.abspath(os.path.join(os.getcwd(), "output", "remotion_captions.mp4"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self._check_node()
        logs.append(self._ensure_node_dependencies(project_dir))

        public_dir = tempfile.mkdtemp(prefix="remotion_public_")
        try:
            pub_video = os.path.join(public_dir, "video.mp4")
            shutil.copy2(video_path, pub_video)

            json_out = os.path.join(public_dir, "video.json")
            convert_script = os.path.join(project_dir, "tools", "convert-srt-to-json.mjs")
            if not os.path.isfile(convert_script):
                raise RuntimeError("SRT 转换脚本缺失。")

            conv = subprocess.run(
                ["node", convert_script, srt_path, json_out],
                cwd=project_dir,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logs.append(conv.stdout or "")
            if conv.returncode != 0:
                raise RuntimeError("SRT 转换失败，见日志。")

            props = {
                "src": "video.mp4",
                "switchEveryMs": int(switchEveryMs),
                "highlightColor": str(highlightColor),
                "style": style,
            }
            props_str = json.dumps(props)

            cmd = [
                "npx",
                "--yes",
                "remotion",
                "render",
                "src/index.ts",
                "CaptionedVideo",
                output_path,
                f"--props={props_str}",
                f"--public-dir={public_dir}",
                f"--timeout={int(timeout_ms)}",
                f"--concurrency={int(concurrency)}",
            ]
            proc = subprocess.run(
                cmd,
                cwd=project_dir,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logs.append(proc.stdout or "")
            if proc.returncode != 0:
                raise RuntimeError("Remotion 渲染失败，见日志。")

            return (output_path, "\n".join([m for m in logs if m]))
        finally:
            try:
                shutil.rmtree(public_dir)
            except Exception:
                pass
