"""
ComfyUI 增强字幕添加节点
整合 comfy_add_subtitles 的高级样式功能
与现有 add_subtitles_to_frames 节点接口兼容，但功能更强大
"""

from PIL import ImageDraw, ImageFont, Image, ImageFilter
from .utils import tensor2pil, pil2tensor
import math
import os
import json
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


class SubtitleStyle:
    """字幕样式配置类"""
    
    def __init__(self, 
                 font_color: str = "white",
                 font_family: str = "Roboto-Regular.ttf", 
                 font_size: int = 100,
                 outline_color: str = "black",
                 outline_width: int = 3,
                 shadow_color: str = "black", 
                 shadow_offset: Tuple[int, int] = (2, 2),
                 background_color: str = None,
                 background_opacity: float = 0.7,
                 gradient_colors: List[str] = None,
                 text_align: str = "center"):
        self.font_color = font_color
        self.font_family = font_family
        self.font_size = font_size
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset
        self.background_color = background_color
        self.background_opacity = background_opacity
        self.gradient_colors = gradient_colors or []
        self.text_align = text_align


class AddSubtitlesEnhancedNode:
    """增强版字幕添加节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用字体
        try:
            available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith('.ttf')]
        except:
            available_fonts = ["Roboto-Regular.ttf"]
        
        return {
            "required": { 
                "images": ("IMAGE",),
                "alignment": ("whisper_alignment",),
                "video_fps": ("FLOAT", {
                    "default": 24.0,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                # 基础样式
                "font_family": (available_fonts, {"default": "Roboto-Regular.ttf"}),
                "font_size": ("INT", {
                    "default": 100,
                    "min": 20,
                    "max": 500,
                    "step": 5
                }),
                "font_color": ("STRING", {"default": "white"}),
                
                # 位置控制
                "x_position": ("INT", {
                    "default": 100,
                    "step": 50
                }),
                "y_position": ("INT", {
                    "default": 100, 
                    "step": 50
                }),
                "center_x": ("BOOLEAN", {"default": True}),
                "center_y": ("BOOLEAN", {"default": True}),
                
                # 高级样式
                "outline_color": ("STRING", {"default": "black"}),
                "outline_width": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 20,
                    "step": 1
                }),
                "shadow_color": ("STRING", {"default": "gray"}),
                "shadow_offset_x": ("INT", {
                    "default": 2,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "shadow_offset_y": ("INT", {
                    "default": 2,
                    "min": -20, 
                    "max": 20,
                    "step": 1
                }),
                
                # 背景样式
                "background_color": ("STRING", {"default": ""}),
                "background_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "background_padding": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 5
                }),
                
                # 渐变效果
                "gradient_colors": ("STRING", {
                    "default": "",
                    "placeholder": "例如: #FF0000,#00FF00,#0000FF"
                }),
                
                # 文字效果
                "text_align": (["left", "center", "right"], {"default": "center"}),
                "line_spacing": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1
                }),
                "max_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2000,
                    "tooltip": "最大文字宽度，0为不限制"
                }),
                
                # 动画效果
                "fade_in_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "淡入时间（秒）"
                }),
                "fade_out_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "淡出时间（秒）"
                }),
                
                # 高级选项
                "enable_emoji": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用emoji表情渲染"
                }),
                "text_case": (["none", "upper", "lower", "title"], {
                    "default": "none",
                    "tooltip": "文字大小写转换"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "subtitle_coord", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "cropped_subtitles", "subtitle_coord", "render_log")
    FUNCTION = "add_subtitles_enhanced"
    CATEGORY = "字幕"

    def add_subtitles_enhanced(self, 
                             images,
                             alignment: List[Dict],
                             video_fps: float = 24.0,
                             font_family: str = "Roboto-Regular.ttf",
                             font_size: int = 100,
                             font_color: str = "white",
                             x_position: int = 100,
                             y_position: int = 100,
                             center_x: bool = True,
                             center_y: bool = True,
                             outline_color: str = "black",
                             outline_width: int = 3,
                             shadow_color: str = "gray",
                             shadow_offset_x: int = 2,
                             shadow_offset_y: int = 2,
                             background_color: str = "",
                             background_opacity: float = 0.7,
                             background_padding: int = 20,
                             gradient_colors: str = "",
                             text_align: str = "center",
                             line_spacing: float = 1.2,
                             max_width: int = 0,
                             fade_in_duration: float = 0.0,
                             fade_out_duration: float = 0.0,
                             enable_emoji: bool = True,
                             text_case: str = "none") -> Tuple:
        """
        增强版字幕添加功能
        """
        log_messages = []
        
        try:
            # 转换图像格式
            pil_images = tensor2pil(images)
            log_messages.append(f"📷 处理图像数量: {len(pil_images)}")
            log_messages.append(f"🎬 视频帧率: {video_fps} fps")
            log_messages.append(f"🎯 对齐数据: {len(alignment)} 个片段")
            
            # 创建字幕样式
            style = SubtitleStyle(
                font_color=font_color,
                font_family=font_family,
                font_size=font_size,
                outline_color=outline_color,
                outline_width=outline_width,
                shadow_color=shadow_color,
                shadow_offset=(shadow_offset_x, shadow_offset_y),
                background_color=background_color if background_color else None,
                background_opacity=background_opacity,
                gradient_colors=self._parse_gradient_colors(gradient_colors),
                text_align=text_align
            )
            
            # 加载字体
            try:
                font_path = os.path.join(FONT_DIR, font_family)
                font = ImageFont.truetype(font_path, font_size)
                log_messages.append(f"🔤 字体加载成功: {font_family}")
            except Exception as e:
                # 使用默认字体作为后备
                font = ImageFont.load_default()
                log_messages.append(f"⚠️ 字体加载失败，使用默认字体: {e}")
            
            # 初始化输出列表
            pil_images_with_text = []
            cropped_pil_images_with_text = []
            pil_images_masks = []
            subtitle_coord = []
            
            # 处理空对齐情况
            if len(alignment) == 0:
                log_messages.append("⚠️ 无对齐数据，返回原始图像")
                return self._return_original_images(pil_images, log_messages)
            
            # 处理每一帧
            last_frame_no = 0
            processed_segments = 0
            
            for i, alignment_obj in enumerate(alignment):
                start_time = alignment_obj.get("start", 0.0)
                end_time = alignment_obj.get("end", 0.0) 
                text = alignment_obj.get("value", "").strip()
                
                if not text:
                    continue
                
                # 应用文字大小写转换
                text = self._apply_text_case(text, text_case)
                
                start_frame_no = math.floor(start_time * video_fps)
                end_frame_no = math.floor(end_time * video_fps)
                
                # 确保帧数在有效范围内
                start_frame_no = max(0, min(start_frame_no, len(pil_images) - 1))
                end_frame_no = max(start_frame_no, min(end_frame_no, len(pil_images)))
                
                # 添加无字幕帧
                for frame_idx in range(last_frame_no, start_frame_no):
                    if frame_idx < len(pil_images):
                        self._add_empty_frame(
                            pil_images[frame_idx],
                            pil_images_with_text,
                            pil_images_masks,
                            cropped_pil_images_with_text,
                            subtitle_coord
                        )
                
                # 添加有字幕帧
                for frame_idx in range(start_frame_no, end_frame_no):
                    if frame_idx < len(pil_images):
                        # 计算动画透明度
                        frame_time = frame_idx / video_fps
                        alpha = self._calculate_alpha(
                            frame_time, start_time, end_time,
                            fade_in_duration, fade_out_duration
                        )
                        
                        # 渲染字幕
                        self._render_subtitle_frame(
                            pil_images[frame_idx],
                            text,
                            font,
                            style,
                            x_position, y_position,
                            center_x, center_y,
                            max_width,
                            line_spacing,
                            alpha,
                            pil_images_with_text,
                            pil_images_masks,
                            cropped_pil_images_with_text,
                            subtitle_coord
                        )
                
                last_frame_no = end_frame_no
                processed_segments += 1
                
                log_messages.append(
                    f"✅ 片段 {i+1}: '{text[:20]}...' "
                    f"({start_time:.1f}s-{end_time:.1f}s, 帧{start_frame_no}-{end_frame_no})"
                )
            
            # 处理剩余帧
            for frame_idx in range(last_frame_no, len(pil_images)):
                self._add_empty_frame(
                    pil_images[frame_idx],
                    pil_images_with_text,
                    pil_images_masks, 
                    cropped_pil_images_with_text,
                    subtitle_coord
                )
            
            log_messages.append(f"🎉 字幕渲染完成!")
            log_messages.append(f"📊 统计信息:")
            log_messages.append(f"  - 处理片段数: {processed_segments}")
            log_messages.append(f"  - 总帧数: {len(pil_images_with_text)}")
            log_messages.append(f"  - 字幕帧数: {sum(1 for coord in subtitle_coord if coord != (0,0,0,0))}")
            
            # 转换回张量格式
            output_images = pil2tensor(pil_images_with_text)
            output_masks = pil2tensor([img.convert("L") for img in pil_images_masks])
            cropped_images = pil2tensor(cropped_pil_images_with_text)
            
            return (output_images, output_masks, cropped_images, subtitle_coord, "\n".join(log_messages))
            
        except Exception as e:
            error_msg = f"字幕渲染过程发生错误: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            log_messages.append(f"❌ {error_msg}")
            
            # 返回原始图像
            return self._return_original_images(pil_images, log_messages)
    
    def _parse_gradient_colors(self, gradient_str: str) -> List[str]:
        """解析渐变颜色字符串"""
        if not gradient_str.strip():
            return []
        
        try:
            colors = [color.strip() for color in gradient_str.split(',')]
            return [color for color in colors if color]
        except:
            return []
    
    def _apply_text_case(self, text: str, text_case: str) -> str:
        """应用文字大小写转换"""
        if text_case == "upper":
            return text.upper()
        elif text_case == "lower":
            return text.lower()
        elif text_case == "title":
            return text.title()
        return text
    
    def _calculate_alpha(self, frame_time: float, start_time: float, end_time: float,
                        fade_in_duration: float, fade_out_duration: float) -> float:
        """计算帧的透明度（用于淡入淡出效果）"""
        duration = end_time - start_time
        relative_time = frame_time - start_time
        
        alpha = 1.0
        
        # 淡入效果
        if fade_in_duration > 0 and relative_time < fade_in_duration:
            alpha *= relative_time / fade_in_duration
        
        # 淡出效果  
        if fade_out_duration > 0 and relative_time > (duration - fade_out_duration):
            remaining_time = end_time - frame_time
            alpha *= remaining_time / fade_out_duration
        
        return max(0.0, min(1.0, alpha))
    
    def _add_empty_frame(self, img, images_with_text, masks, cropped_images, coords):
        """添加无字幕的帧"""
        img = img.convert("RGB")
        width, height = img.size
        
        images_with_text.append(img)
        
        # 创建黑色蒙版
        black_mask = Image.new('RGB', (width, height), 'black')
        masks.append(black_mask)
        
        # 创建最小尺寸的裁剪图像
        small_black = Image.new('RGB', (1, 1), 'black')
        cropped_images.append(small_black)
        
        coords.append((0, 0, 0, 0))
    
    def _render_subtitle_frame(self, img, text, font, style, x_pos, y_pos,
                              center_x, center_y, max_width, line_spacing, alpha,
                              images_with_text, masks, cropped_images, coords):
        """渲染单个字幕帧"""
        img = img.convert("RGBA")
        width, height = img.size
        
        # 创建文字层
        text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        
        # 处理换行（如果指定了最大宽度）
        lines = self._wrap_text(text, font, max_width) if max_width > 0 else [text]
        
        # 计算文字总体尺寸
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        
        total_height = sum(line_heights) + (len(lines) - 1) * int(line_heights[0] * (line_spacing - 1))
        max_line_width = max(line_widths) if line_widths else 0
        
        # 计算起始位置
        if center_x:
            start_x = (width - max_line_width) // 2
        else:
            start_x = x_pos
            
        if center_y:
            start_y = (height - total_height) // 2
        else:
            start_y = y_pos
        
        # 绘制背景（如果有）
        if style.background_color:
            bg_alpha = int(255 * style.background_opacity * alpha)
            bg_color = self._parse_color(style.background_color) + (bg_alpha,)
            
            # 计算背景矩形
            padding = 20  # 背景内边距
            bg_left = start_x - padding
            bg_top = start_y - padding
            bg_right = start_x + max_line_width + padding
            bg_bottom = start_y + total_height + padding
            
            draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_color)
        
        # 绘制每行文字
        current_y = start_y
        text_coords = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                current_y += int(line_heights[0] * line_spacing)
                continue
                
            # 计算该行的x位置
            line_width = line_widths[i]
            if style.text_align == "center":
                line_x = start_x + (max_line_width - line_width) // 2
            elif style.text_align == "right":
                line_x = start_x + max_line_width - line_width
            else:  # left
                line_x = start_x
            
            # 绘制阴影
            if style.shadow_color and style.shadow_offset != (0, 0):
                shadow_x = line_x + style.shadow_offset[0]
                shadow_y = current_y + style.shadow_offset[1]
                shadow_color = self._parse_color(style.shadow_color) + (int(255 * alpha),)
                draw.text((shadow_x, shadow_y), line, font=font, fill=shadow_color)
            
            # 绘制描边
            if style.outline_width > 0:
                outline_color = self._parse_color(style.outline_color) + (int(255 * alpha),)
                for dx in range(-style.outline_width, style.outline_width + 1):
                    for dy in range(-style.outline_width, style.outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((line_x + dx, current_y + dy), line, 
                                    font=font, fill=outline_color)
            
            # 绘制主文字
            text_color = self._parse_color(style.font_color) + (int(255 * alpha),)
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            
            # 记录坐标
            bbox = draw.textbbox((line_x, current_y), line, font=font)
            text_coords.append(bbox)
            
            current_y += int(line_heights[i] * line_spacing)
        
        # 合并图层
        result_img = Image.alpha_composite(img, text_layer)
        result_img = result_img.convert("RGB")
        
        images_with_text.append(result_img)
        
        # 创建蒙版
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # 在蒙版上绘制文字区域
        for bbox in text_coords:
            mask_draw.rectangle(bbox, fill=255)
        
        masks.append(mask.convert("RGB"))
        
        # 创建裁剪的字幕图像
        if text_coords:
            # 计算所有文字的边界框
            min_x = min(bbox[0] for bbox in text_coords)
            min_y = min(bbox[1] for bbox in text_coords) 
            max_x = max(bbox[2] for bbox in text_coords)
            max_y = max(bbox[3] for bbox in text_coords)
            
            # 裁剪字幕区域
            cropped_subtitle = result_img.crop((min_x, min_y, max_x, max_y))
            cropped_images.append(cropped_subtitle)
            coords.append((min_x, min_y, max_x, max_y))
        else:
            # 无文字时添加小黑图
            small_black = Image.new('RGB', (1, 1), 'black')
            cropped_images.append(small_black)
            coords.append((0, 0, 0, 0))
    
    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """文字换行处理"""
        if max_width <= 0:
            return [text]
        
        words = text.split()
        lines = []
        current_line = []
        
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width or not current_line:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """解析颜色字符串为RGB元组"""
        color_str = color_str.strip().lower()
        
        # 预定义颜色
        color_map = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }
        
        if color_str in color_map:
            return color_map[color_str]
        
        # 处理十六进制颜色
        if color_str.startswith('#'):
            try:
                hex_str = color_str[1:]
                if len(hex_str) == 6:
                    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                elif len(hex_str) == 3:
                    return tuple(int(hex_str[i]*2, 16) for i in range(3))
            except ValueError:
                pass
        
        # 处理RGB格式 rgb(255,255,255)
        if color_str.startswith('rgb(') and color_str.endswith(')'):
            try:
                rgb_str = color_str[4:-1]
                return tuple(int(x.strip()) for x in rgb_str.split(','))
            except ValueError:
                pass
        
        # 默认返回白色
        return (255, 255, 255)
    
    def _return_original_images(self, pil_images, log_messages):
        """返回原始图像（错误处理时使用）"""
        # 创建空蒙版和坐标
        width, height = pil_images[0].size if pil_images else (1, 1)
        black_masks = [Image.new('RGB', (width, height), 'black') for _ in pil_images]
        small_blacks = [Image.new('RGB', (1, 1), 'black') for _ in pil_images]
        empty_coords = [(0, 0, 0, 0) for _ in pil_images]
        
        output_images = pil2tensor(pil_images)
        output_masks = pil2tensor([img.convert("L") for img in black_masks])
        cropped_images = pil2tensor(small_blacks)
        
        return (output_images, output_masks, cropped_images, empty_coords, "\n".join(log_messages))


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AddSubtitlesEnhancedNode": AddSubtitlesEnhancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddSubtitlesEnhancedNode": "Add Subtitles (Enhanced)"
}


# 测试代码
if __name__ == "__main__":
    print("🎨 增强字幕添加节点测试")
    node = AddSubtitlesEnhancedNode()
    print("📋 输入类型:", node.INPUT_TYPES())
    print("🎯 返回类型:", node.RETURN_TYPES)
    print("📝 返回名称:", node.RETURN_NAMES)