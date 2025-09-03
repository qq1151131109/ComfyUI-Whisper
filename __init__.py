from .apply_whisper import ApplyWhisperNode
from .add_subtitles_to_frames import AddSubtitlesToFramesNode
from .add_subtitles_to_background import AddSubtitlesToBackgroundNode
from .resize_cropped_subtitles import ResizeCroppedSubtitlesNode

# 新增的WhisperX强制对齐和增强字幕节点
try:
    from .apply_whisperx_alignment import ApplyWhisperXAlignmentNode
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("⚠️ WhisperX依赖未安装，WhisperX强制对齐节点不可用")

try:
    from .add_subtitles_enhanced import AddSubtitlesEnhancedNode
    ENHANCED_SUBTITLES_AVAILABLE = True
except ImportError:
    ENHANCED_SUBTITLES_AVAILABLE = False
    print("⚠️ 增强字幕节点加载失败")

# Remotion 渲染节点（合并自 ComfyUI-RemotionCaptions）
try:
    from .remotion.remotion_captions_node import RenderRemotionCaptionsNode
    REMOTION_AVAILABLE = True
except Exception as e:
    REMOTION_AVAILABLE = False
    print(f"⚠️ Remotion 字幕渲染节点加载失败: {e}")

NODE_CLASS_MAPPINGS = { 
    "Apply Whisper" : ApplyWhisperNode,
    "Add Subtitles To Frames": AddSubtitlesToFramesNode,
    "Add Subtitles To Background": AddSubtitlesToBackgroundNode,
    "Resize Cropped Subtitles": ResizeCroppedSubtitlesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "Apply Whisper" : "Apply Whisper", 
     "Add Subtitles To Frames": "Add Subtitles To Frames",
     "Add Subtitles To Background": "Add Subtitles To Background",
     "Resize Cropped Subtitles": "Resize Cropped Subtitles"
}

# 注册WhisperX强制对齐节点
if WHISPERX_AVAILABLE:
    NODE_CLASS_MAPPINGS["Apply WhisperX Alignment"] = ApplyWhisperXAlignmentNode
    NODE_DISPLAY_NAME_MAPPINGS["Apply WhisperX Alignment"] = "🎯 WhisperX 强制对齐 (Forced Alignment)"

# 注册增强字幕节点
if ENHANCED_SUBTITLES_AVAILABLE:
    NODE_CLASS_MAPPINGS["Add Subtitles Enhanced"] = AddSubtitlesEnhancedNode
    NODE_DISPLAY_NAME_MAPPINGS["Add Subtitles Enhanced"] = "🎨 Add Subtitles (Enhanced)"

# 注册 Remotion 节点
if REMOTION_AVAILABLE:
    NODE_CLASS_MAPPINGS["Render Remotion Captions"] = RenderRemotionCaptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["Render Remotion Captions"] = "🎬 Render TikTok Captions (Remotion)"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']