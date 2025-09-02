"""
ComfyUI WhisperX 强制对齐节点
与现有 apply_whisper 节点接口保持一致，支持音频流输入
用于将准确文本与音频进行精确时间同步，生成高质量字幕
"""

import os
import sys
import json
import uuid
import tempfile
import torchaudio
import torch
import logging
import folder_paths
from typing import Dict, Any, Tuple, Optional, List

import comfy.model_management as mm
import comfy.model_patcher

logger = logging.getLogger(__name__)

# WhisperX依赖检查
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    logger.info(f"WhisperX版本: {whisperx.__version__ if hasattr(whisperx, '__version__') else 'Unknown'}")
except ImportError as e:
    WHISPERX_AVAILABLE = False
    whisperx = None
    logger.warning(f"WhisperX未安装: {e}")

try:
    import ctranslate2
    logger.info(f"CTranslate2版本: {ctranslate2.__version__ if hasattr(ctranslate2, '__version__') else 'Unknown'}")
except ImportError:
    logger.warning("CTranslate2未安装，可能导致WhisperX功能异常")

WHISPERX_MODEL_SUBDIR = os.path.join("stt", "whisperx")
WHISPERX_PATCHER_CACHE = {}


def convert_device_for_whisperx(device):
    """将torch设备格式转换为WhisperX兼容的设备字符串（全局函数）"""
    if hasattr(device, 'type'):
        device_type = device.type
    else:
        device_str = str(device).lower()
        if 'cuda' in device_str:
            device_type = 'cuda'
        elif 'cpu' in device_str:
            device_type = 'cpu'
        else:
            device_type = 'cpu'  # 默认使用CPU
    
    # WhisperX只支持 'cuda' 或 'cpu'，不支持 'cuda:0' 格式
    if device_type == 'cuda':
        return 'cuda'
    else:
        return 'cpu'


def validate_device_compatibility(device_str: str) -> bool:
    """验证设备兼容性（全局函数）"""
    if device_str not in ['cuda', 'cpu']:
        logger.warning(f"设备 '{device_str}' 可能不被WhisperX支持，已转换为兼容格式")
        return False
    
    if device_str == 'cuda' and torch and not torch.cuda.is_available():
        logger.warning("CUDA设备不可用，将使用CPU模式")
        return False
    
    return True


class WhisperXModelWrapper(torch.nn.Module):
    """WhisperX模型包装器，集成ComfyUI模型管理"""
    
    def __init__(self, model_name, language=None):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.whisperx_model = None
        self.align_model = None
        self.model_loaded_weight_memory = 0
        self.device = mm.get_torch_device()
        self.compute_type = "float16" if self.device.type == "cuda" else "int8"

    def load_model(self, device):
        """加载WhisperX模型到指定设备"""
        if not WHISPERX_AVAILABLE:
            raise ImportError("WhisperX not installed. Please run: pip install whisperx")
        
        try:
            # 转换设备格式：torch.device -> WhisperX兼容格式
            device_str = convert_device_for_whisperx(device)
            
            # 验证设备兼容性
            if not validate_device_compatibility(device_str):
                if device_str == 'cuda':
                    device_str = 'cpu'  # 回退到CPU
                    logger.info("CUDA不可用，回退到CPU模式")
            
            logger.info(f"Loading WhisperX model with device: {device_str}")
            
            # 加载Whisper ASR模型
            self.whisperx_model = whisperx.load_model(
                self.model_name,
                device_str,  # 使用字符串格式的设备
                compute_type=self.compute_type,
                language=self.language if self.language != "auto" else None
            )
            
            # 估算模型大小用于内存管理
            if hasattr(self.whisperx_model, 'model'):
                model_size = sum(p.numel() * p.element_size() 
                               for p in self.whisperx_model.model.parameters())
                self.model_loaded_weight_memory = model_size
            
            logger.info(f"WhisperX model '{self.model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise


class WhisperXPatcher(comfy.model_patcher.ModelPatcher):
    """WhisperX模型管理器，集成ComfyUI内存管理系统"""
    
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def patch_model(self, device_to=None, *args, **kwargs):
        """加载模型到目标设备"""
        target_device = self.load_device

        if self.model.whisperx_model is None:
            logger.info(f"Loading WhisperX model '{self.model.model_name}' to {target_device}...")
            self.model.load_model(target_device)
            self.size = self.model.model_loaded_weight_memory
        else:
            logger.info(f"WhisperX model '{self.model.model_name}' already loaded")

        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """卸载模型释放显存"""
        if unpatch_weights:
            logger.info(f"Unloading WhisperX model '{self.model.model_name}'...")
            self.model.whisperx_model = None
            self.model.align_model = None
            self.model.model_loaded_weight_memory = 0
            mm.soft_empty_cache()
        
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)


class ApplyWhisperXAlignmentNode:
    """WhisperX强制对齐节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取支持的语言选项
        language_options = [
            "auto",      # 自动检测
            "Chinese",   # 中文
            "English",   # 英语  
            "Japanese",  # 日语
            "Korean",    # 韩语
            "French",    # 法语
            "German",    # 德语
            "Spanish",   # 西班牙语
            "Italian"    # 意大利语
        ]
        
        # 模型大小选项
        model_options = [
            'tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large'
        ]
        
        return {
            "required": {
                "audio": ("AUDIO",),  # 与apply_whisper保持一致
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入准确的文本内容，将与音频进行强制对齐"
                }),
                "model": (model_options, {"default": "base"}),
            },
            "optional": {
                "language": (language_options, {"default": "auto"}),
                "return_char_alignments": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否返回字符级别的对齐信息"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "whisper_alignment", "whisper_alignment", "STRING")
    RETURN_NAMES = ("aligned_text", "segments_alignment", "words_alignment", "process_log")
    FUNCTION = "apply_whisperx_alignment"
    CATEGORY = "字幕"

    def apply_whisperx_alignment(self, 
                               audio: Dict[str, torch.Tensor],
                               reference_text: str,
                               model: str,
                               language: str = "auto",
                               return_char_alignments: bool = False) -> Tuple[str, List[Dict], List[Dict], str]:
        """
        执行WhisperX强制对齐
        
        Args:
            audio: 音频张量字典 {"waveform": tensor, "sample_rate": int}
            reference_text: 准确的参考文本
            model: 模型大小
            language: 语言（auto为自动检测）
            return_char_alignments: 是否返回字符级对齐
            
        Returns:
            (对齐文本, 句级对齐, 词级对齐, 处理日志)
        """
        log_messages = []
        
        try:
            # 验证输入
            if not reference_text.strip():
                error_msg = "请提供准确的参考文本用于对齐"
                log_messages.append(f"❌ {error_msg}")
                return "", [], [], "\n".join(log_messages)
            
            log_messages.append(f"📝 参考文本长度: {len(reference_text)} 字符")
            log_messages.append(f"🤖 使用模型: {model}")
            log_messages.append(f"🌍 语言设置: {language}")
            
            # 保存音频到临时文件
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            audio_save_path = os.path.join(temp_dir, f"whisperx_{uuid.uuid1()}.wav")
            
            torchaudio.save(
                audio_save_path, 
                audio['waveform'].squeeze(0), 
                audio["sample_rate"]
            )
            log_messages.append(f"💾 音频已保存: {os.path.basename(audio_save_path)}")
            
            # 获取或创建模型管理器
            language_code = self._get_language_code(language)
            cache_key = f"{model}_{language_code}"
            
            if cache_key not in WHISPERX_PATCHER_CACHE:
                load_device = mm.get_torch_device()
                log_messages.append(f"🔄 初始化WhisperX模型管理器: {model}")
                
                model_wrapper = WhisperXModelWrapper(model, language_code)
                patcher = WhisperXPatcher(
                    model=model_wrapper,
                    load_device=load_device,
                    offload_device=mm.unet_offload_device(),
                    size=0  # 将在模型加载时设置
                )
                WHISPERX_PATCHER_CACHE[cache_key] = patcher
            
            patcher = WHISPERX_PATCHER_CACHE[cache_key]
            
            # 加载模型
            mm.load_model_gpu(patcher)
            whisperx_model = patcher.model.whisperx_model
            
            if whisperx_model is None:
                error_msg = f"WhisperX模型加载失败: {model}"
                log_messages.append(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            log_messages.append("✅ WhisperX模型加载成功")
            
            # 执行转录（获取初始对齐）
            log_messages.append("🎵 开始音频转录...")
            transcribe_options = {}
            if language_code and language_code != "auto":
                transcribe_options['language'] = language_code
            
            audio_result = whisperx_model.transcribe(
                audio_save_path, 
                batch_size=16,
                **transcribe_options
            )
            
            detected_language = audio_result.get("language", language_code or "unknown")
            log_messages.append(f"🔍 检测到语言: {detected_language}")
            
            # 加载对齐模型
            log_messages.append("🎯 加载对齐模型...")
            try:
                # 转换设备格式用于对齐模型
                align_device = convert_device_for_whisperx(patcher.load_device)
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=align_device
                )
                patcher.model.align_model = model_a
                log_messages.append("✅ 对齐模型加载成功")
            except Exception as e:
                log_messages.append(f"⚠️ 对齐模型加载失败，使用基础对齐: {e}")
                model_a, metadata = None, {}
            
            # 执行强制对齐
            log_messages.append("⚡ 开始强制对齐...")
            if model_a is not None:
                aligned_result = whisperx.align(
                    audio_result["segments"],
                    model_a,
                    metadata,
                    audio_save_path,
                    align_device,  # 使用字符串格式的设备
                    return_char_alignments=return_char_alignments
                )
                segments = aligned_result["segments"]
                words = aligned_result.get("word_segments", [])
            else:
                # 使用基础转录结果
                segments = audio_result["segments"]
                words = []
                for segment in segments:
                    if "words" in segment:
                        words.extend(segment["words"])
            
            log_messages.append(f"🎉 对齐完成! 句子: {len(segments)}, 词语: {len(words)}")
            
            # 格式化输出为whisper_alignment格式
            segments_alignment = []
            words_alignment = []
            aligned_text_parts = []
            
            for segment in segments:
                # 句级别对齐
                segment_data = {
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "value": segment.get("text", "").strip(),
                    "confidence": segment.get("confidence", 1.0)
                }
                segments_alignment.append(segment_data)
                aligned_text_parts.append(segment_data["value"])
                
                # 词级别对齐
                if "words" in segment:
                    for word in segment["words"]:
                        word_data = {
                            "start": word.get("start", segment_data["start"]),
                            "end": word.get("end", segment_data["end"]),
                            "value": word.get("word", "").strip(),
                            "confidence": word.get("confidence", 1.0)
                        }
                        words_alignment.append(word_data)
            
            # 生成对齐后的完整文本
            aligned_text = " ".join(aligned_text_parts)
            
            # 清理临时文件
            try:
                os.remove(audio_save_path)
                log_messages.append("🧹 临时文件已清理")
            except Exception as e:
                log_messages.append(f"⚠️ 临时文件清理失败: {e}")
            
            log_messages.append(f"📊 处理统计:")
            log_messages.append(f"  - 对齐句子数: {len(segments_alignment)}")
            log_messages.append(f"  - 对齐词语数: {len(words_alignment)}")
            log_messages.append(f"  - 检测语言: {detected_language}")
            log_messages.append(f"  - 文本匹配度: {self._calculate_text_similarity(reference_text, aligned_text):.1%}")
            
            return aligned_text, segments_alignment, words_alignment, "\n".join(log_messages)
            
        except Exception as e:
            error_msg = f"WhisperX对齐过程发生错误: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            
            # 提供具体的错误分析和解决建议
            error_analysis = self._analyze_error(str(e))
            log_messages.append(f"❌ {error_msg}")
            if error_analysis:
                log_messages.extend(error_analysis)
            
            # 清理临时文件
            try:
                if 'audio_save_path' in locals():
                    os.remove(audio_save_path)
            except:
                pass
            
            return "", [], [], "\n".join(log_messages)
    
    def _get_language_code(self, language: str) -> str:
        """将语言名称转换为代码"""
        language_mapping = {
            "auto": None,
            "chinese": "zh",
            "english": "en", 
            "japanese": "ja",
            "korean": "ko",
            "french": "fr",
            "german": "de", 
            "spanish": "es",
            "italian": "it"
        }
        return language_mapping.get(language.lower(), None)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的词语重叠率计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_error(self, error_str: str) -> List[str]:
        """分析错误并提供解决建议"""
        suggestions = []
        
        if "incompatible constructor arguments" in error_str and "ctranslate2" in error_str:
            suggestions.extend([
                "🔧 检测到CTranslate2版本兼容性问题，建议解决方案：",
                "   1. 更新WhisperX: pip install --upgrade whisperx",
                "   2. 或降级CTranslate2: pip install ctranslate2==3.24.0", 
                "   3. 或使用兼容版本: pip install whisperx==3.1.1",
                "   4. 检查CUDA版本兼容性"
            ])
        
        elif "unsupported device" in error_str:
            suggestions.extend([
                "🔧 检测到设备格式不兼容问题，建议解决方案：",
                "   1. 已自动修复设备格式转换",
                "   2. 如果问题持续，尝试重启ComfyUI",
                "   3. 检查CUDA驱动是否正常工作",
                "   4. 尝试使用CPU模式测试"
            ])
        
        elif "device" in error_str.lower():
            suggestions.extend([
                "🔧 检测到设备相关错误，建议解决方案：",
                "   1. 检查CUDA是否正确安装",
                "   2. 尝试使用CPU模式",
                "   3. 重启ComfyUI重新初始化设备"
            ])
        
        elif "memory" in error_str.lower() or "oom" in error_str.lower():
            suggestions.extend([
                "🔧 检测到内存不足，建议解决方案：",
                "   1. 使用更小的模型 (如 base、small)",
                "   2. 降低batch_size",
                "   3. 关闭其他占用GPU内存的程序"
            ])
        
        elif "model" in error_str.lower() and "load" in error_str.lower():
            suggestions.extend([
                "🔧 检测到模型加载错误，建议解决方案：",
                "   1. 检查网络连接",
                "   2. 清理模型缓存目录",
                "   3. 手动下载模型文件"
            ])
        
        else:
            suggestions.extend([
                "🔧 通用解决方案：",
                "   1. 检查所有依赖是否正确安装",
                "   2. 重启ComfyUI",
                "   3. 查看完整错误日志",
                "   4. 尝试使用较小的音频文件测试"
            ])
        
        return suggestions


# 节点注册
NODE_CLASS_MAPPINGS = {
    "ApplyWhisperXAlignmentNode": ApplyWhisperXAlignmentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyWhisperXAlignmentNode": "Apply WhisperX (Force Alignment)"
}


# 测试代码
if __name__ == "__main__":
    print("🎵 WhisperX强制对齐节点测试")
    node = ApplyWhisperXAlignmentNode()
    print("📋 输入类型:", node.INPUT_TYPES())
    print("🎯 返回类型:", node.RETURN_TYPES)
    print("📝 返回名称:", node.RETURN_NAMES)