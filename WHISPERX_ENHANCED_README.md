# WhisperX强制对齐 & 增强字幕节点

## 🎯 功能概述

本项目在原有ComfyUI-Whisper的基础上新增了两个强大的节点：

1. **🎯 WhisperX 强制对齐 (Forced Alignment)** - 真正的强制对齐节点
2. **🎨 Add Subtitles (Enhanced)** - 增强字幕添加节点

### 💡 什么是强制对齐 (Forced Alignment)?

**强制对齐**是一种将已知的准确文本与音频进行精确时间同步的技术：

- **输入**：准确的参考文本 + 对应的音频文件
- **输出**：文本中每个词/句在音频中的精确时间戳
- **要求**：参考文本必须与音频内容完全匹配
- **用途**：配音同步、字幕校时、朗读对齐、有声书制作等

**与普通ASR的区别**：
- **ASR转录**：音频 → 识别出文本（可能有错误）
- **强制对齐**：准确文本 + 音频 → 精确的时间同步

### 📁 节点分类
所有字幕相关节点现在统一归类在 **"字幕"** 目录下，包括：
- Apply Whisper - 基础语音转录（ASR）
- 🎯 WhisperX 强制对齐 - 真正的强制对齐
- Add Subtitles To Frames - 基础字幕添加  
- Add Subtitles To Background - 背景字幕
- 🎨 Add Subtitles (Enhanced) - 增强字幕
- Resize Cropped Subtitles - 字幕尺寸调整

## 📦 安装依赖

### 基础依赖（已有）
```bash
pip install openai-whisper pillow
```

### WhisperX强制对齐依赖（可选）
```bash
pip install whisperx>=3.1.0 torch torchaudio librosa
```

## 🎵 WhisperX强制对齐节点

### 功能特点
- **精确时间对齐**：将已知文本与音频进行强制对齐，获得精确时间戳
- **接口标准化**：与原有apply_whisper节点接口保持一致，接受AUDIO输入
- **内存优化**：集成ComfyUI的ModelPatcher系统，支持自动显存管理
- **多语言支持**：支持中文、英文、日文、韩文等多种语言

### 输入参数
| 参数 | 类型 | 说明 |
|-----|------|------|
| audio | AUDIO | 音频数据（来自上游节点） |
| reference_text | STRING | 准确的参考文本内容 |
| model | LIST | 模型大小选择 |
| language | LIST | 语言选择（可选） |
| return_char_alignments | BOOLEAN | 是否返回字符级对齐（可选） |

### 输出结果
| 输出 | 类型 | 说明 |
|-----|------|------|
| aligned_text | STRING | 对齐后的文本 |
| segments_alignment | whisper_alignment | 句级别对齐数据 |
| words_alignment | whisper_alignment | 词级别对齐数据 |
| process_log | STRING | 处理过程日志 |

### 使用场景
- **配音对齐**：将配音文本与音频精确对齐
- **字幕校时**：修正现有字幕的时间戳
- **朗读同步**：朗读文本与音频的时间同步
- **歌词对齐**：歌词与歌曲的精确同步

## 🎨 增强字幕添加节点

### 功能特点
- **丰富样式**：支持描边、阴影、背景、渐变等高级效果
- **智能布局**：自动换行、文字对齐、位置居中
- **动画效果**：支持淡入淡出动画
- **兼容性好**：与原有add_subtitles_to_frames节点接口兼容

### 主要样式选项

#### 基础样式
- **字体**：支持多种字体文件
- **大小**：可调节字体大小（20-500px）
- **颜色**：支持颜色名称、十六进制、RGB格式

#### 高级效果
- **描边**：可设置描边颜色和宽度
- **阴影**：支持阴影颜色和偏移
- **背景**：半透明背景框
- **渐变**：多色渐变文字效果

#### 布局控制
- **对齐方式**：左对齐、居中、右对齐
- **行距**：可调节行间距
- **最大宽度**：自动换行控制
- **位置**：X/Y坐标定位

#### 动画效果
- **淡入时间**：字幕出现时的淡入效果
- **淡出时间**：字幕消失时的淡出效果

## 🚀 快速开始

### 1. 基础工作流
```
音频加载 → WhisperX强制对齐 → 增强字幕添加 → 视频保存
```

### 2. 高级工作流
```
音频加载 ┐
         ├→ WhisperX强制对齐 → 增强字幕添加 → 视频保存
视频加载 ┘                    ↓
                        字幕样式预览
```

### 3. 对比工作流
```
音频加载 ┐
         ├→ 普通Whisper转录 → 基础字幕添加
         └→ WhisperX强制对齐 → 增强字幕添加
```

## 📋 使用示例

### 配音对齐示例
1. 加载配音音频文件
2. 在reference_text中输入准确的台词文本
3. 选择合适的语言和模型
4. 获得精确的时间戳数据
5. 使用增强字幕节点渲染字幕

### 字幕美化示例  
1. 使用现有的对齐数据
2. 设置字体样式（字体、大小、颜色）
3. 添加描边和阴影效果
4. 设置半透明背景
5. 调整淡入淡出动画

## ⚡ 性能优化

### 内存管理
- 自动显存管理，避免OOM
- 模型缓存机制，提升重复使用效率
- 临时文件自动清理

### 处理速度
- 批量处理支持
- GPU加速计算
- 智能依赖检查

## 🔧 故障排除

### 常见问题

**1. WhisperX节点不可用**
```
解决方案：安装WhisperX依赖
pip install whisperx>=3.1.0
```

**2. 对齐效果不好**
```
解决方案：
- 确保参考文本与音频内容完全一致
- 选择合适的语言设置（不要使用auto）
- 尝试使用更大的模型（如large-v3）
```

**3. 字幕显示不正常**
```
解决方案：
- 检查字体文件是否存在
- 调整字体大小和位置参数
- 确认视频帧率设置正确
```

**4. 内存不足**
```
解决方案：
- 使用较小的模型（如base、small）
- 降低视频分辨率
- 分段处理长视频
```

**5. CTranslate2版本兼容性错误**
```
错误信息：incompatible constructor arguments
解决方案：
- 方案1: pip install whisperx==3.1.1 ctranslate2==3.24.0
- 方案2: pip install --upgrade whisperx
- 方案3: 使用虚拟环境隔离依赖冲突
```

**6. 设备格式不支持错误**
```
错误信息：unsupported device cuda:0
解决方案：
- ✅ 已自动修复：节点会自动将 cuda:0 转换为 cuda
- 如果问题持续，检查CUDA驱动是否正常
- 可以尝试重启ComfyUI重新初始化
```

**7. 模型结构兼容性错误**
```
错误信息：'WhisperModel' object has no attribute 'parameters'
解决方案：
- ✅ 已自动修复：节点会自动使用安全的模型大小估算
- 如果问题持续，重启ComfyUI重新加载修复后的代码
- 检查WhisperX版本是否与其他依赖兼容
```

**8. 版本兼容性警告**
```
警告信息：Model was trained with pyannote.audio 0.0.1, yours is 3.x.x
解决方案：
- ⚠️ 这些是警告，不影响功能使用
- 如需消除警告：pip install pyannote.audio==3.1.0
- 或忽略警告，继续正常使用
```

**9. 其他设备相关错误**
```
错误信息：device相关错误  
解决方案：
- 检查CUDA安装状态
- 重启ComfyUI重新初始化设备
- 尝试使用CPU模式（设置环境变量CUDA_VISIBLE_DEVICES=""）
```

### 日志分析
节点会输出详细的处理日志，包括：
- 对齐统计信息
- 渲染过程状态
- 错误和警告信息
- 性能指标数据

## 🎁 示例工作流

项目提供了完整的示例工作流：
- `whisperx_force_alignment_workflow.json` - 基础强制对齐流程
- 展示了从音频加载到字幕渲染的完整流程
- 包含样式配置和效果预览

## 📝 更新日志

### v1.0 (2024-01-20)
- ✅ 新增WhisperX强制对齐节点
- ✅ 新增增强字幕添加节点  
- ✅ 集成ComfyUI模型管理系统
- ✅ 支持多种字幕样式和动画效果
- ✅ 提供完整示例工作流

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境
```bash
git clone <repo>
cd ComfyUI-Whisper
pip install -e .
```

### 测试节点
```bash
python apply_whisperx_alignment.py
python add_subtitles_enhanced.py
```

## 📄 许可证

本项目基于原ComfyUI-Whisper项目协议开源。

---

**🎉 享受高质量的字幕制作体验！**