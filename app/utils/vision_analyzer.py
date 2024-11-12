# 修改后的 vision_analyzer.py

import json
from typing import List, Union, Dict
import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import asyncio
from tenacity import retry, stop_after_attempt, RetryError, retry_if_exception_type, wait_exponential
from abc import ABC, abstractmethod
import PIL.Image
import traceback

class BaseVisionAnalyzer(ABC):
    """视觉分析器基类"""
    
    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("必须提供API密钥")
            
        self.model_name = model_name
        self.api_key = api_key
        self._configure_client()
    
    @abstractmethod
    def _configure_client(self):
        """配置API客户端"""
        pass
        
    @abstractmethod
    async def _generate_content_with_retry(self, prompt: str, image_batch: List[PIL.Image.Image]) -> str:
        """带重试机制的内容生成"""
        pass
        
    @abstractmethod
    def process_response(self, response: any) -> str:
        """处理API响应"""
        pass

    async def analyze_images(self,
                           images: Union[List[str], List[PIL.Image.Image]],
                           prompt: str,
                           batch_size: int = 5) -> List[Dict]:
        """批量分析多张图片的通用实现"""
        try:
            # 加载图片
            if isinstance(images[0], str):
                logger.info("正在加载图片...")
                images = self.load_images(images)

            # 验证每个图片对象
            valid_images = []
            for i, img in enumerate(images):
                if not isinstance(img, PIL.Image.Image):
                    logger.error(f"无效的图片对象，索引 {i}: {type(img)}")
                    continue
                valid_images.append(img)

            if not valid_images:
                raise ValueError("没有有效的图片对象")

            images = valid_images
            results = []
            total_batches = (len(images) + batch_size - 1) // batch_size

            with tqdm(total=total_batches, desc="分析进度") as pbar:
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size]
                    retry_count = 0

                    while retry_count < 3:
                        try:
                            # 在每个批次处理前添加小延迟
                            if i > 0:
                                await asyncio.sleep(2)

                            response = await self._generate_content_with_retry(prompt, batch)
                            processed_response = self.process_response(response)
                            
                            results.append({
                                'batch_index': i // batch_size,
                                'images_processed': len(batch),
                                'response': processed_response,
                                'model_used': self.model_name
                            })
                            break

                        except Exception as e:
                            retry_count += 1
                            error_msg = f"批次 {i // batch_size} 处理出错: {str(e)}"
                            logger.error(error_msg)

                            if retry_count >= 3:
                                results.append({
                                    'batch_index': i // batch_size,
                                    'images_processed': len(batch),
                                    'error': error_msg,
                                    'model_used': self.model_name
                                })
                            else:
                                await asyncio.sleep(60)

                    pbar.update(1)

            return results

        except Exception as e:
            error_msg = f"图片分析过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def load_images(self, image_paths: List[str]) -> List[PIL.Image.Image]:
        """加载多张图片的通用实现"""
        images = []
        failed_images = []

        for img_path in image_paths:
            try:
                if not os.path.exists(img_path):
                    logger.error(f"图片文件不存在: {img_path}")
                    failed_images.append(img_path)
                    continue

                img = PIL.Image.open(img_path)
                img.load()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)

            except Exception as e:
                logger.error(f"无法加载图片 {img_path}: {str(e)}")
                failed_images.append(img_path)

        if failed_images:
            logger.warning(f"以下图片加载失败:\n{json.dumps(failed_images, indent=2, ensure_ascii=False)}")

        if not images:
            raise ValueError("没有成功加载任何图片")

        return images

class GeminiVisionAnalyzer(BaseVisionAnalyzer):
    """Gemini 视觉分析器实现"""
    
    def _configure_client(self):
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        genai.configure(api_key=self.api_key)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.model = genai.GenerativeModel(self.model_name, safety_settings=safety_settings)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_content_with_retry(self, prompt: str, image_batch: List[PIL.Image.Image]) -> str:
        try:
            response = await self.model.generate_content_async([prompt, *image_batch])
            return response
        except Exception as e:
            logger.error(f"Gemini生成内容失败: {str(e)}")
            raise

    def process_response(self, response: any) -> str:
        if not response or not response.text:
            raise ValueError("Invalid response from Gemini API")
        return response.text.strip()

class GPTVisionAnalyzer(BaseVisionAnalyzer):
    """GPT-4 Vision 分析器实现"""
    
    def _configure_client(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_content_with_retry(self, prompt: str, image_batch: List[PIL.Image.Image]) -> str:
        try:
            import base64
            from io import BytesIO
            
            # 准备消息内容
            messages = []
            messages.append({"type": "text", "text": prompt})
            
            # 添加所有图片
            for image in image_batch:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })

            # 调用API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": messages}]
            )
            
            return response
        except Exception as e:
            logger.error(f"GPT-4 Vision生成内容失败: {str(e)}")
            raise

    def process_response(self, response: any) -> str:
        if not response or not response.choices:
            raise ValueError("Invalid response from GPT-4 Vision API")
        return response.choices[0].message.content.strip()

def create_vision_analyzer(model_type: str, model_name: str, api_key: str) -> BaseVisionAnalyzer:
    """工厂函数：创建视觉分析器实例"""
    analyzers = {
        'gemini': GeminiVisionAnalyzer,
        'gpt': GPTVisionAnalyzer,
        # 在这里添加新的模型支持
    }
    
    analyzer_class = analyzers.get(model_type.lower())
    if not analyzer_class:
        raise ValueError(f"不支持的模型类型: {model_type}")
        
    return analyzer_class(model_name, api_key)