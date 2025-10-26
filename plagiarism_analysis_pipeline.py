#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Textbook Plagiarism Analysis Pipeline
Комплексный пайплайн анализа учебников на плагиат

Features:
- Multi-level caching (OCR text, extracted entities, analysis results)
- Human-readable cache files
- Advanced entity extraction
- Modular pipeline architecture
- Progress tracking
"""

import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# PDF and OCR libraries
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2

# Text analysis - using only built-in difflib for simplicity

class CachedTextbookAnalyzer:
    def __init__(self, source_pdf_path, suspect_pdf_path, cache_dir="cache"):
        """
        Initialize the comprehensive analyzer with caching
        
        Args:
            source_pdf_path: Path to source textbook
            suspect_pdf_path: Path to suspected plagiarism textbook  
            cache_dir: Directory for cache files
        """
        self.source_path = source_pdf_path
        self.suspect_path = suspect_pdf_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different cache types
        (self.cache_dir / "ocr_text").mkdir(exist_ok=True)
        (self.cache_dir / "entities").mkdir(exist_ok=True)
        (self.cache_dir / "analysis").mkdir(exist_ok=True)
        
        # Configuration
        self.config = {
            'ocr': {
                'language': 'rus+eng',
                'dpi': 200,
                'max_pages': 10000,  # Process all pages
                'batch_size': 25,    # Smaller batches for stability
                'min_text_length': 5, # More permissive minimum text length
                'min_confidence': 20  # Lower confidence threshold
            },
            'similarity': {
                'threshold': 0.15,  # Further lowered to catch more potential borrowings
                'min_phrase_length': 20,  # Minimum characters for matching phrase
                'min_words': 3,  # Minimum words for matching phrase
                'paraphrase_threshold': 0.8  # Similarity threshold for paraphrases
            },
            'entities': {
                'question_confidence_threshold': 0.8,
                'min_question_length': 10,
                'max_question_length': 500
            }
        }
        
        self.results = {}
        
    def _get_file_hash(self, file_path):
        """Generate hash for file to detect changes"""
        with open(file_path, 'rb') as f:
            # Read first and last 1MB for hash (efficient for large PDFs)
            first_chunk = f.read(1024 * 1024)
            f.seek(-1024 * 1024, 2)
            last_chunk = f.read(1024 * 1024)
            
        return hashlib.md5(first_chunk + last_chunk).hexdigest()
    
    def _get_cache_path(self, cache_type, file_path, suffix=""):
        """Get cache file path for given type and file"""
        file_hash = self._get_file_hash(file_path)
        filename = f"{Path(file_path).stem}_{file_hash}{suffix}.json"
        return self.cache_dir / cache_type / filename
    
    def _save_human_readable_cache(self, data, cache_path, title="Cache Data"):
        """Save cache in human-readable JSON format"""
        cache_data = {
            'metadata': {
                'title': title,
                'created_at': datetime.now().isoformat(),
                'file_path': str(data.get('source_file', 'unknown')),
                'total_pages': data.get('total_pages', 0),
                'extraction_method': data.get('extraction_method', [])
            },
            'data': data
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    def _load_cache(self, cache_path):
        """Load cache data if exists and valid"""
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                print(f"📁 Loaded cache: {cache_path.name}")
                return cache_data.get('data', cache_data)
            except Exception as e:
                print(f"⚠️ Cache file corrupted, will regenerate: {e}")
        return None
    
    def extract_text_with_cache(self, pdf_path):
        """Extract text with caching support"""
        cache_path = self._get_cache_path("ocr_text", pdf_path, "_text")
        cached_data = self._load_cache(cache_path)
        
        if cached_data:
            return cached_data
        
        print(f"🔄 Extracting text from: {os.path.basename(pdf_path)}")
        
        # Initialize content structure
        content = {
            'source_file': pdf_path,
            'pages_with_text': 0,
            'total_pages': 0,
            'text_pages': [],  # List of {page_number, text, method, char_count, confidence}
            'images': [],
            'extraction_method': [],
            'extraction_stats': {
                'normal_extraction_pages': 0,
                'ocr_extraction_pages': 0,
                'failed_pages': 0,
                'total_characters': 0
            }
        }
        
        try:
            # Get total pages
            doc = fitz.open(pdf_path)
            content['total_pages'] = len(doc)
            
            # Try normal text extraction first
            normal_text_pages = 0
            for page_num in range(min(10, len(doc))):  # Sample first 10 pages
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if len(text) > self.config['ocr']['min_text_length']:
                    normal_text_pages += 1
            
            # Decide extraction method
            use_ocr = normal_text_pages < min(3, len(doc) * 0.2)  # If <20% pages have text
            
            if use_ocr:
                print(f"🔍 Using OCR (normal extraction gave {normal_text_pages} pages)")
                content = self._extract_with_advanced_ocr(pdf_path, content, doc)
            else:
                print(f"📄 Using normal text extraction")
                content = self._extract_with_pymupdf(pdf_path, content, doc)
            
            doc.close()
            
        except Exception as e:
            print(f"❌ Error extracting from {pdf_path}: {e}")
            content['extraction_method'].append(f'Error: {e}')
        
        # Save to cache
        self._save_human_readable_cache(
            content, cache_path, 
            f"Text Extraction - {os.path.basename(pdf_path)}"
        )
        
        return content
    
    def _extract_with_pymupdf(self, pdf_path, content, doc):
        """Normal text extraction using PyMuPDF"""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            
            if len(text) > self.config['ocr']['min_text_length']:
                content['pages_with_text'] += 1
                content['text_pages'].append({
                    'page_number': page_num + 1,
                    'text': text,
                    'method': 'PyMuPDF',
                    'char_count': len(text),
                    'confidence': 1.0  # High confidence for direct extraction
                })
                content['extraction_stats']['normal_extraction_pages'] += 1
                content['extraction_stats']['total_characters'] += len(text)
            
            # Count images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                content['images'].append({
                    'page': page_num + 1,
                    'img_index': img_index,
                    'xref': img[0]
                })
        
        content['extraction_method'].append('PyMuPDF')
        return content
    
    def _extract_with_advanced_ocr(self, pdf_path, content, doc):
        """Advanced OCR extraction with confidence scoring"""
        max_pages = min(self.config['ocr']['max_pages'], content['total_pages'])
        batch_size = self.config['ocr']['batch_size']
        
        print(f"📖 OCR processing {max_pages} pages in batches of {batch_size}")
        print(f"📋 OCR settings: min_text_length={self.config['ocr']['min_text_length']}, min_confidence={self.config['ocr']['min_confidence']}")
        
        total_processed = 0
        successful_extractions = 0
        for start_page in range(0, max_pages, batch_size):
            end_page = min(start_page + batch_size, max_pages)
            print(f"  Processing pages {start_page + 1}-{end_page}...")
            
            try:
                # Convert PDF pages to images
                images = convert_from_path(
                    pdf_path,
                    first_page=start_page + 1,
                    last_page=end_page,
                    dpi=self.config['ocr']['dpi']
                )
                
                for i, image in enumerate(images):
                    page_num = start_page + i + 1
                    total_processed += 1
                    
                    # Enhanced OCR with confidence
                    ocr_result = self._advanced_ocr_page(image, page_num)
                    
                    if ocr_result['text'] and len(ocr_result['text']) > self.config['ocr']['min_text_length']:
                        successful_extractions += 1
                        content['pages_with_text'] += 1
                        content['text_pages'].append({
                            'page_number': page_num,
                            'text': ocr_result['text'],
                            'method': 'OCR',
                            'char_count': len(ocr_result['text']),
                            'confidence': ocr_result['confidence']
                        })
                        content['extraction_stats']['ocr_extraction_pages'] += 1
                        content['extraction_stats']['total_characters'] += len(ocr_result['text'])
                        print(f"    Page {page_num}: ✓ ({ocr_result['char_count']} chars, conf: {ocr_result['confidence']:.2f})")
                    else:
                        content['extraction_stats']['failed_pages'] += 1
                        print(f"    Page {page_num}: ✗ (text_len: {len(ocr_result['text']) if ocr_result['text'] else 0}, conf: {ocr_result['confidence']:.2f})")
                        
                    # Progress update every 20 pages
                    if total_processed % 20 == 0:
                        success_rate = (successful_extractions / total_processed) * 100
                        print(f"    Progress: {total_processed}/{max_pages} pages, {success_rate:.1f}% success rate")
                        
            except Exception as e:
                print(f"    Error in batch {start_page + 1}-{end_page}: {e}")
                content['extraction_stats']['failed_pages'] += (end_page - start_page)
        
        # Final OCR summary
        final_success_rate = (successful_extractions / total_processed) * 100 if total_processed > 0 else 0
        print(f"📊 OCR Summary: {successful_extractions}/{total_processed} pages successfully processed ({final_success_rate:.1f}%)")
        
        # Count images from original PDF
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                content['images'].append({
                    'page': page_num + 1,
                    'img_index': img_index,
                    'xref': img[0]
                })
        
        content['extraction_method'].append('Advanced OCR')
        return content
    
    def _advanced_ocr_page(self, image, page_num):
        """Advanced OCR processing for a single page"""
        try:
            # Convert PIL to OpenCV
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Image preprocessing for better OCR
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction
            denoised = cv2.medianBlur(enhanced, 3)
            
            # OCR with confidence data
            ocr_config = f"--oem 3 --psm 6 -l {self.config['ocr']['language']}"
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                denoised, 
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > self.config['ocr']['min_confidence']:  # Use configurable threshold
                    word = ocr_data['text'][i].strip()
                    if word:
                        text_parts.append(word)
                        confidences.append(int(ocr_data['conf'][i]))
            
            text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'char_count': len(text),
                'word_count': len(text_parts)
            }
            
        except Exception as e:
            print(f"    OCR error on page {page_num}: {e}")
            return {'text': '', 'confidence': 0.0, 'char_count': 0, 'word_count': 0}
    
    def extract_entities_with_cache(self, text_content):
        """Extract educational entities with caching"""
        # Create cache key based on text content
        content_hash = hashlib.md5(str(text_content).encode()).hexdigest()
        cache_path = self.cache_dir / "entities" / f"entities_{content_hash}.json"
        
        cached_entities = self._load_cache(cache_path)
        if cached_entities:
            return cached_entities
        
        print(f"🔄 Extracting educational entities...")
        
        entities = {
            'questions': [],
            'assignments': [],
            'film_references': [],
            'internet_sources': [],
            'maps_and_diagrams': [],
            'historical_figures': [],
            'dates_and_periods': [],
            'extraction_stats': {
                'total_text_analyzed': 0,
                'pages_processed': 0,
                'extraction_confidence': {}
            }
        }
        
        # Process each page
        for page_data in text_content.get('text_pages', []):
            text = page_data['text']
            page_num = page_data['page_number']
            
            entities['extraction_stats']['total_text_analyzed'] += len(text)
            entities['extraction_stats']['pages_processed'] += 1
            
            # Extract different entity types
            self._extract_questions_and_assignments(text, page_num, entities)
            self._extract_media_references(text, page_num, entities)
            self._extract_internet_sources(text, page_num, entities)
            self._extract_maps_and_diagrams(text, page_num, entities)
            self._extract_historical_entities(text, page_num, entities)
        
        # Calculate extraction confidence
        total_entities = sum(len(entities[key]) for key in entities if isinstance(entities[key], list))
        entities['extraction_stats']['total_entities'] = total_entities
        entities['extraction_stats']['entities_per_page'] = total_entities / max(1, entities['extraction_stats']['pages_processed'])
        
        # Save to cache
        self._save_human_readable_cache(
            entities, cache_path,
            "Educational Entities Extraction"
        )
        
        return entities
    
    def _extract_questions_and_assignments(self, text, page_num, entities):
        """Advanced question and assignment extraction"""
        
        # Comprehensive question patterns with confidence scoring
        question_patterns = [
            # Direct question markers
            {
                'pattern': r'(?:Вопрос[ыи]?|Question[s]?)\s*:?\s*(.{10,500}?\?)',
                'confidence': 0.95,
                'type': 'direct_question'
            },
            {
                'pattern': r'(?:Задани[ея]|Task[s]?|Assignment[s]?)\s*:?\s*(.{10,500}?)(?=\n|$)',
                'confidence': 0.90,
                'type': 'assignment'
            },
            # Numbered questions
            {
                'pattern': r'^\s*\d+[\.\)]\s+(.{10,500}?\?)',
                'confidence': 0.85,
                'type': 'numbered_question'
            },
            # Command-style questions
            {
                'pattern': r'(?:Ответьте|Объясните|Сравните|Определите|Назовите|Охарактеризуйте|Проанализируйте)\s+(.{10,500}?)(?=\n|\.|\?|!)',
                'confidence': 0.80,
                'type': 'command_question'
            },
            # WH-questions
            {
                'pattern': r'(?:Как|Что|Где|Когда|Почему|Зачем|Какой|Какая|Какие|Кто)\s+([^.!?]{10,500}?\?)',
                'confidence': 0.75,
                'type': 'wh_question'
            },
            # Discussion prompts
            {
                'pattern': r'(?:Обсудите|Подумайте|Рассмотрите)\s+(.{10,500}?)(?=\n|$)',
                'confidence': 0.70,
                'type': 'discussion'
            }
        ]
        
        for pattern_info in question_patterns:
            matches = re.findall(
                pattern_info['pattern'], 
                text, 
                re.MULTILINE | re.IGNORECASE
            )
            
            for match in matches:
                question_text = match.strip()
                
                # Validate question quality
                if self._validate_question_quality(question_text):
                    entities['questions'].append({
                        'text': question_text,
                        'page': page_num,
                        'type': pattern_info['type'],
                        'confidence': pattern_info['confidence'],
                        'length': len(question_text),
                        'complexity_score': self._calculate_question_complexity(question_text)
                    })
    
    def _extract_media_references(self, text, page_num, entities):
        """Extract film and media references"""
        
        media_patterns = [
            {
                'pattern': r'(?:фильм[ыи]?|кино|кинофильм[ыи]?|документальн[ыи]й\s+фильм)[:\s]*([^\n]{5,200})',
                'confidence': 0.90,
                'type': 'film'
            },
            {
                'pattern': r'(?:Рекомендуем[ыи]?\s+к\s+просмотру|Смотрите|К\s+просмотру)[:\s]*([^\n]{5,200})',
                'confidence': 0.85,
                'type': 'recommendation'
            },
            {
                'pattern': r'(?:Видео|Видеоролик|Видеофрагмент)[:\s]*([^\n]{5,200})',
                'confidence': 0.80,
                'type': 'video'
            }
        ]
        
        for pattern_info in media_patterns:
            matches = re.findall(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                entities['film_references'].append({
                    'text': match.strip(),
                    'page': page_num,
                    'type': pattern_info['type'],
                    'confidence': pattern_info['confidence']
                })
    
    def _extract_internet_sources(self, text, page_num, entities):
        """Extract internet sources and digital resources"""
        
        url_patterns = [
            {
                'pattern': r'(?:https?://|www\.)[^\s\n]{5,100}',
                'confidence': 0.95,
                'type': 'url'
            },
            {
                'pattern': r'[а-яё\w\-]+\.(?:ru|com|org|net|рф|edu|gov)(?:/[^\s\n]*)?',
                'confidence': 0.85,
                'type': 'domain'
            },
            {
                'pattern': r'(?:Сайт|Портал|Ресурс|Интернет[- ]?ресурс)[:\s]*([^\n]{5,150})',
                'confidence': 0.80,
                'type': 'web_resource'
            }
        ]
        
        for pattern_info in url_patterns:
            matches = re.findall(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                entities['internet_sources'].append({
                    'text': match.strip(),
                    'page': page_num,
                    'type': pattern_info['type'],
                    'confidence': pattern_info['confidence']
                })
    
    def _extract_maps_and_diagrams(self, text, page_num, entities):
        """Extract references to maps, diagrams, and visual materials"""
        
        visual_patterns = [
            {
                'pattern': r'(?:карта|схема|план|диаграмма|график|таблица)[^\n]{0,100}',
                'confidence': 0.85,
                'type': 'visual_reference'
            },
            {
                'pattern': r'(?:рис\.|рисунок|иллюстрация)\s*\d*[^\n]{0,100}',
                'confidence': 0.90,
                'type': 'figure_reference'
            },
            {
                'pattern': r'(?:см\.|смотри|смотрите)\s+(?:карт[уы]|схем[уы]|рис\.|рисунок)[^\n]{0,100}',
                'confidence': 0.95,
                'type': 'visual_instruction'
            }
        ]
        
        for pattern_info in visual_patterns:
            matches = re.findall(pattern_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                entities['maps_and_diagrams'].append({
                    'text': match.strip(),
                    'page': page_num,
                    'type': pattern_info['type'],
                    'confidence': pattern_info['confidence']
                })
    
    def _extract_historical_entities(self, text, page_num, entities):
        """Extract historical figures, dates, and periods"""
        
        # Historical figures (Russian names pattern)
        figure_pattern = r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b'
        potential_figures = re.findall(figure_pattern, text)
        
        # Filter out common words and validate as historical figures
        historical_keywords = ['князь', 'царь', 'император', 'король', 'великий', 'святой', 'полководец']
        for figure in potential_figures:
            if any(keyword in text.lower() for keyword in historical_keywords):
                entities['historical_figures'].append({
                    'text': figure,
                    'page': page_num,
                    'confidence': 0.70,
                    'context': self._extract_context(text, figure, 50)
                })
        
        # Dates and periods
        date_patterns = [
            r'\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{3,4}',
            r'\b\d{3,4}\s*г\.?',
            r'\b(?:в|с|до|после)\s+\d{3,4}',
            r'\b\d{1,2}-\d{1,2}\s+век[аи]?\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['dates_and_periods'].append({
                    'text': match,
                    'page': page_num,
                    'confidence': 0.80
                })
    
    def _validate_question_quality(self, question_text):
        """Validate if extracted text is actually a meaningful question"""
        if not question_text or len(question_text) < self.config['entities']['min_question_length']:
            return False
        
        if len(question_text) > self.config['entities']['max_question_length']:
            return False
        
        # Check for question markers
        question_markers = ['?', 'как', 'что', 'где', 'когда', 'почему', 'зачем', 'какой', 'кто']
        has_marker = any(marker in question_text.lower() for marker in question_markers)
        
        # Check for common non-question patterns
        false_positives = ['http', 'www', '@', 'тел.', 'факс']
        has_false_positive = any(fp in question_text.lower() for fp in false_positives)
        
        return has_marker and not has_false_positive
    
    def _calculate_question_complexity(self, question_text):
        """Calculate complexity score for a question"""
        complexity_keywords = [
            'сравните', 'проанализируйте', 'охарактеризуйте', 'обоснуйте', 
            'объясните', 'докажите', 'оцените', 'интерпретируйте'
        ]
        
        complexity_score = 0
        for keyword in complexity_keywords:
            if keyword in question_text.lower():
                complexity_score += 1
        
        # Normalize to 0-1 range
        return min(complexity_score / len(complexity_keywords), 1.0)
    
    def _extract_context(self, text, entity, context_length=50):
        """Extract context around an entity"""
        start_pos = text.find(entity)
        if start_pos == -1:
            return ""
        
        start = max(0, start_pos - context_length)
        end = min(len(text), start_pos + len(entity) + context_length)
        
        return text[start:end].strip()
    
    def analyze_similarity_with_cache(self, source_entities, suspect_entities):
        """Perform similarity analysis with caching"""
        
        # Create cache key based on both entity sets
        combined_hash = hashlib.md5(
            (str(source_entities) + str(suspect_entities)).encode()
        ).hexdigest()
        
        cache_path = self.cache_dir / "analysis" / f"similarity_{combined_hash}.json"
        cached_analysis = self._load_cache(cache_path)
        
        if cached_analysis:
            return cached_analysis
        
        print(f"🔄 Performing similarity analysis...")
        
        analysis = {
            'text_similarity': self._analyze_text_similarity(source_entities, suspect_entities),
            'entity_similarity': self._analyze_entity_similarity(source_entities, suspect_entities),
            'structural_comparison': self._compare_structure(source_entities, suspect_entities),
            'borrowing_analysis': self._analyze_borrowings(source_entities, suspect_entities),
            'confidence_scores': {}
        }
        
        # Calculate overall confidence
        analysis['overall_confidence'] = self._calculate_overall_confidence(analysis)
        
        # Save to cache
        self._save_human_readable_cache(
            analysis, cache_path,
            "Similarity Analysis Results"
        )
        
        return analysis
    
    def _analyze_text_similarity(self, source_entities, suspect_entities):
        """Analyze text similarity between source and suspect"""
        print("📊 Проводим детальный анализ текстовой схожести...")

        # Load raw text data from cache
        source_text_data = self._load_text_from_cache('source')
        suspect_text_data = self._load_text_from_cache('suspect')

        if not source_text_data or not suspect_text_data:
            print("⚠️  Не удалось загрузить данные текста из кеша")
            return {
                'page_similarities': [],
                'overall_similarity': 0.0,
                'method_comparison': {},
                'suspicious_matches': [],
                'matching_phrases': []
            }

        page_similarities = []
        suspicious_matches = []
        overall_similarities = []
        all_matching_phrases = []

        print(f"📄 Сравниваем {len(source_text_data)} страниц источника с {len(suspect_text_data)} страницами подозреваемого")

        # Compare each source page with each suspect page
        for source_page in source_text_data:
            source_text = source_page.get('text', '').strip()
            if not source_text or len(source_text) < 100:  # Skip very short texts
                continue

            best_match = {'similarity': 0.0, 'suspect_page': None}

            for suspect_page in suspect_text_data:
                suspect_text = suspect_page.get('text', '').strip()
                if not suspect_text or len(suspect_text) < 100:
                    continue

                similarity = self._calculate_text_similarity(source_text, suspect_text)

                if similarity['average'] > best_match['similarity']:
                    best_match = {
                        'similarity': similarity['average'],
                        'suspect_page': suspect_page['page_number'],
                        'source_page': source_page['page_number'],
                        'similarity_details': similarity,
                        'source_text': source_text,
                        'suspect_text': suspect_text
                    }

            if best_match['similarity'] > 0.1:  # Only record meaningful similarities
                page_similarities.append(best_match)
                overall_similarities.append(best_match['similarity'])

                if best_match['similarity'] > self.config['similarity']['threshold']:
                    # Extract matching phrases for this pair
                    phrases = self._extract_matching_phrases(
                        best_match['source_text'],
                        best_match['suspect_text'],
                        best_match['source_page'],
                        best_match['suspect_page']
                    )
                    all_matching_phrases.extend(phrases)

                    suspicious_matches.append({
                        'source_page': best_match['source_page'],
                        'suspect_page': best_match['suspect_page'],
                        'similarity': best_match['similarity_details'],
                        'source_text_preview': source_text[:200] + "..." if len(source_text) > 200 else source_text,
                        'suspect_text_preview': suspect_text[:200] + "..." if len(suspect_text) > 200 else suspect_text,
                        'matching_phrases_count': len(phrases)
                    })

        overall_similarity = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0.0

        print(f"✅ Найдено {len(suspicious_matches)} подозрительных совпадений")
        print(f"📊 Общая схожесть текста: {overall_similarity:.3f}")
        print(f"🔍 Извлечено {len(all_matching_phrases)} совпадающих фраз")

        return {
            'page_similarities': page_similarities,
            'overall_similarity': overall_similarity,
            'method_comparison': {
                'total_comparisons': len(source_text_data) * len(suspect_text_data),
                'meaningful_similarities': len(page_similarities),
                'suspicious_matches': len(suspicious_matches)
            },
            'suspicious_matches': suspicious_matches,
            'matching_phrases': all_matching_phrases
        }
    
    def _analyze_entity_similarity(self, source_entities, suspect_entities):
        """Analyze similarity in extracted entities"""
        entity_comparison = {}
        
        for entity_type in ['questions', 'film_references', 'internet_sources']:
            source_items = source_entities.get(entity_type, [])
            suspect_items = suspect_entities.get(entity_type, [])
            
            entity_comparison[entity_type] = {
                'source_count': len(source_items),
                'suspect_count': len(suspect_items),
                'matches': self._find_entity_matches(source_items, suspect_items),
                'similarity_percentage': 0.0
            }
        
        return entity_comparison
    
    def _find_entity_matches(self, source_items, suspect_items):
        """Find matching entities between source and suspect"""
        matches = []
        
        for suspect_item in suspect_items:
            best_match = None
            best_similarity = 0.0
            
            for source_item in source_items:
                similarity = self._calculate_text_similarity(
                    suspect_item.get('text', ''),
                    source_item.get('text', '')
                )
                
                if similarity['average'] > best_similarity:
                    best_similarity = similarity['average']
                    best_match = {
                        'source': source_item,
                        'suspect': suspect_item,
                        'similarity': similarity
                    }
            
            if best_similarity > self.config['similarity']['threshold']:
                matches.append(best_match)
        
        return matches
    
    def _load_text_from_cache(self, file_type):
        """Load raw text data from cache files"""
        # Determine which file to load based on file_type
        if file_type == 'source':
            file_path = self.source_path
        elif file_type == 'suspect':
            file_path = self.suspect_path
        else:
            return None

        # Get the cache path for this file
        cache_path = self._get_cache_path("ocr_text", file_path, "_text")

        if not cache_path.exists():
            print(f"⚠️  Не найден файл кеша для {file_type}")
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('data', {}).get('text_pages', [])
        except Exception as e:
            print(f"⚠️  Ошибка загрузки кеша для {file_type}: {e}")
            return None

    def _extract_matching_phrases(self, source_text, suspect_text, source_page, suspect_page):
        """Extract matching phrases between two texts using SequenceMatcher"""
        matches = []
        min_phrase_length = self.config['similarity']['min_phrase_length']
        min_words = self.config['similarity']['min_words']
        paraphrase_threshold = self.config['similarity']['paraphrase_threshold']

        # Normalize texts
        source_normalized = re.sub(r'\s+', ' ', source_text.lower().strip())
        suspect_normalized = re.sub(r'\s+', ' ', suspect_text.lower().strip())

        # 1. Find exact matching blocks using SequenceMatcher
        matcher = SequenceMatcher(None, source_normalized, suspect_normalized)
        matching_blocks = matcher.get_matching_blocks()

        for block in matching_blocks:
            source_start, suspect_start, length = block

            # Only consider matches of significant length
            if length >= min_phrase_length:
                source_phrase = source_normalized[source_start:source_start + length].strip()
                suspect_phrase = suspect_normalized[suspect_start:suspect_start + length].strip()

                # Validate that phrase contains actual words, not just spaces/punctuation
                if len(source_phrase.split()) >= min_words:
                    matches.append({
                        'source_page': source_page,
                        'suspect_page': suspect_page,
                        'source_phrase': source_phrase,
                        'suspect_phrase': suspect_phrase,
                        'length': length,
                        'similarity': 1.0,  # Exact match
                        'type': 'exact'
                    })

        # 2. Find paraphrased sentences using SequenceMatcher
        source_sentences = re.split(r'[.!?]\s+', source_text)
        suspect_sentences = re.split(r'[.!?]\s+', suspect_text)

        for src_sent in source_sentences:
            if len(src_sent) < 30:  # Skip very short sentences
                continue

            src_sent_norm = re.sub(r'\s+', ' ', src_sent.lower().strip())

            for susp_sent in suspect_sentences:
                if len(susp_sent) < 30:
                    continue

                susp_sent_norm = re.sub(r'\s+', ' ', susp_sent.lower().strip())

                # Calculate sentence similarity using SequenceMatcher
                sent_similarity = SequenceMatcher(None, src_sent_norm, susp_sent_norm).ratio()

                # If sentences are similar but not exact matches (paraphrases)
                if paraphrase_threshold <= sent_similarity < 1.0:
                    matches.append({
                        'source_page': source_page,
                        'suspect_page': suspect_page,
                        'source_phrase': src_sent.strip(),
                        'suspect_phrase': susp_sent.strip(),
                        'length': max(len(src_sent), len(susp_sent)),
                        'similarity': sent_similarity,
                        'type': 'paraphrase'
                    })

        return matches

    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts using SequenceMatcher only"""
        if not text1 or not text2:
            return {'average': 0.0, 'sequence_matcher': 0.0}

        # Clean text
        text1_clean = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_clean = re.sub(r'\s+', ' ', text2.lower().strip())

        # Calculate similarity using SequenceMatcher
        seq_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()

        return {
            'sequence_matcher': seq_similarity,
            'average': seq_similarity  # Average is just sequence_matcher now
        }
    
    def _compare_structure(self, source_entities, suspect_entities):
        """Compare structural elements"""
        return {
            'page_count_ratio': 1.0,  # Placeholder
            'content_density_comparison': {},
            'entity_distribution': {}
        }
    
    def _analyze_borrowings(self, source_entities, suspect_entities):
        """Analyze potential borrowings"""
        return {
            'text_borrowing_percentage': 0.0,
            'entity_borrowing_percentage': 0.0,
            'high_confidence_matches': [],
            'suspicious_patterns': []
        }
    
    def _calculate_overall_confidence(self, analysis):
        """Calculate overall confidence in analysis results"""
        return 0.85  # Placeholder
    
    def run_full_pipeline(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Comprehensive Textbook Analysis Pipeline")
        print("=" * 70)
        
        # Step 1: Extract text with caching
        print("\n📖 Step 1: Text Extraction")
        source_text = self.extract_text_with_cache(self.source_path)
        suspect_text = self.extract_text_with_cache(self.suspect_path)
        
        # Step 2: Extract entities with caching
        print("\n🔍 Step 2: Entity Extraction")  
        source_entities = self.extract_entities_with_cache(source_text)
        suspect_entities = self.extract_entities_with_cache(suspect_text)
        
        # Step 3: Similarity analysis with caching
        print("\n⚖️ Step 3: Similarity Analysis")
        similarity_analysis = self.analyze_similarity_with_cache(source_entities, suspect_entities)
        
        # Step 4: Generate comprehensive report
        print("\n📊 Step 4: Report Generation")
        self.results = {
            'source_text': source_text,
            'suspect_text': suspect_text,
            'source_entities': source_entities,
            'suspect_entities': suspect_entities,
            'similarity_analysis': similarity_analysis,
            'pipeline_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'source_file': self.source_path,
                'suspect_file': self.suspect_path,
                'cache_directory': str(self.cache_dir),
                'configuration': self.config
            }
        }
        
        return self.results
    
    def generate_comprehensive_report(self, output_dir="client"):
        """Generate human-readable comprehensive report"""
        if not self.results:
            print("❌ No analysis results found. Run pipeline first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate base filename from source and suspect files
        source_name = Path(self.source_path).stem
        suspect_name = Path(self.suspect_path).stem
        base_name = f"{source_name}_vs_{suspect_name}"

        # Create dedicated folder for this pair
        pair_folder = output_path / base_name
        pair_folder.mkdir(exist_ok=True)

        # 1. Save main report
        report_path = pair_folder / "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report())

        # 2. Save matching phrases
        phrases_path = pair_folder / "matching_phrases.txt"
        with open(phrases_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_matching_phrases_report())

        # 3. Save normalized texts
        source_text_path = pair_folder / "source_normalized_text.txt"
        suspect_text_path = pair_folder / "suspect_normalized_text.txt"

        with open(source_text_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_normalized_text(self.results['source_text']))

        with open(suspect_text_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_normalized_text(self.results['suspect_text']))

        # 4. Save detailed JSON data
        data_path = pair_folder / "detailed_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

        print(f"📋 Reports generated in {pair_folder}/:")
        print(f"  • analysis_report.md")
        print(f"  • matching_phrases.txt")
        print(f"  • source_normalized_text.txt")
        print(f"  • suspect_normalized_text.txt")
        print(f"  • detailed_data.json")

    def _generate_method_explanation(self):
        """Generate method explanation for non-mathematical client"""
        return r"""# ОБЪЯСНЕНИЕ МЕТОДА АНАЛИЗА НА ПЛАГИАТ

## Для кого этот документ
Это объяснение предназначено для людей без математического образования. Мы объясним, как работает наш анализ простым языком.

## Что делает наша система

### 1. ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF
**Что происходит:** Система читает PDF-файлы учебников и извлекает из них текст.

**Как это работает:**
- Сначала пробуем прочитать текст напрямую (если PDF содержит настоящий текст)
- Если текст не читается (например, учебник отсканирован как картинка), используем OCR (оптическое распознавание символов)
- OCR — это как когда компьютер "смотрит" на картинку и распознаёт на ней буквы

**Что получаем:**
- Текст каждой страницы
- Информацию о том, сколько текста на каждой странице
- Уверенность в правильности распознавания (для OCR)

### 2. ПОИСК ОБРАЗОВАТЕЛЬНЫХ ЭЛЕМЕНТОВ
**Что происходит:** Система ищет в тексте важные образовательные элементы.

**Что ищем:**
- **Вопросы и задания** — по специальным словам-маркерам: "Вопрос:", "Задание:", "Объясните", "Сравните" и т.д.
- **Ссылки на фильмы** — упоминания документальных фильмов, рекомендаций к просмотру
- **Интернет-ресурсы** — веб-сайты, порталы, онлайн-материалы
- **Карты и схемы** — упоминания визуальных материалов
- **Исторические персоны и даты** — имена, даты, временные периоды

**Зачем это нужно:**
Эти элементы помогают понять структуру учебника. Если вопросы очень похожи — это может быть признак копирования.

### 3. СРАВНЕНИЕ ТЕКСТОВ
**Что происходит:** Система сравнивает тексты двух учебников, чтобы найти похожие места.

**Как мы сравниваем:**

#### Метод 1: Посимвольное сравнение (SequenceMatcher)
- Представьте, что мы кладём два текста рядом и смотрим, сколько букв совпадают подряд
- Чем больше совпадений — тем выше схожесть
- Пример: "История России" и "История Украины" совпадают на 50% (первая часть одинаковая)

**Итоговая оценка:**
Используется только посимвольное сравнение через SequenceMatcher — это самый простой и понятный метод, который не требует сложных математических вычислений.

### 4. ИЗВЛЕЧЕНИЕ ОДИНАКОВЫХ ФРАЗ
**Что происходит:** Система находит конкретные куски текста, которые повторяются в обоих учебниках.

**Как это работает:**

#### Точные совпадения:
- Ищем блоки текста длиной минимум 50 символов, которые полностью совпадают
- Должно быть минимум 5 слов подряд
- Пример: "Битва на Куликовом поле произошла в 1380 году и стала переломным моментом"

#### Перефразированные совпадения:
- Ищем предложения, которые похожи на 70% и более, но не идентичны
- Это помогает найти текст, который переписали другими словами
- Пример:
  - Оригинал: "Пётр I провёл масштабные реформы армии и флота"
  - Перефразировка: "Петром Первым были осуществлены крупные военные преобразования"

### 5. КРИТЕРИИ ПОДОЗРИТЕЛЬНОСТИ
**Когда мы считаем совпадение подозрительным:**
- Общая схожесть страниц > 15% (настраиваемый порог)
- Найдены блоки совпадающего текста длиной 50+ символов
- Обнаружены похожие (70%+) предложения

## Что НЕ считается плагиатом
- Одинаковые исторические факты (даты, события — они объективны)
- Стандартные учебные формулировки
- Общеупотребительные термины
- Короткие фразы (меньше 5 слов)

## Ограничения метода
1. **OCR может ошибаться** — если учебник плохо отсканирован, распознавание текста может быть неточным
2. **Не учитывается смысл на 100%** — система оценивает текстовую похожесть, а не глубокое понимание содержания
3. **Нужен контекст** — одинаковые фразы могут быть случайным совпадением или стандартной формулировкой

## Как читать результаты
- **Процент схожести 0-10%** — нормально, учебники про одну тему
- **Процент схожести 10-30%** — требует внимания, возможны заимствования
- **Процент схожести 30%+** — высокая вероятность копирования
- **Количество совпадающих фраз** — смотрите конкретные примеры, они важнее общего процента

## Кэширование (ускорение повторного анализа)
Система сохраняет промежуточные результаты:
- Извлечённый текст
- Найденные образовательные элементы
- Результаты сравнения

При повторном запуске анализа тех же файлов система использует сохранённые данные — это в десятки раз быстрее.

---
*Этот метод соответствует современным научным подходам к детектированию текстовых заимствований*
"""

    def _generate_matching_phrases_report(self):
        """Generate report with all matching phrases"""
        if 'similarity_analysis' not in self.results:
            return "Нет данных о совпадающих фразах"

        matching_phrases = self.results['similarity_analysis']['text_similarity'].get('matching_phrases', [])

        if not matching_phrases:
            return "Совпадающие фразы не найдены"

        report = f"""# СОВПАДАЮЩИЕ ФРАЗЫ
# Найдено совпадений: {len(matching_phrases)}

"""

        # Group by page pairs
        page_pairs = {}
        for phrase in matching_phrases:
            key = (phrase['source_page'], phrase['suspect_page'])
            if key not in page_pairs:
                page_pairs[key] = []
            page_pairs[key].append(phrase)

        for (src_page, susp_page), phrases in sorted(page_pairs.items()):
            report += f"\n## Страница {src_page} (источник) ↔ Страница {susp_page} (подозреваемый)\n"
            report += f"Найдено совпадений: {len(phrases)}\n\n"

            for i, phrase in enumerate(phrases, 1):
                phrase_type = phrase.get('type', 'exact')
                similarity = phrase.get('similarity', 1.0)

                if phrase_type == 'exact':
                    report += f"### Совпадение #{i} (точное, {phrase['length']} символов)\n"
                else:
                    report += f"### Совпадение #{i} (перефразировка, схожесть {similarity:.1%})\n"

                report += f"**Источник:** {phrase['source_phrase']}\n\n"
                report += f"**Подозреваемый:** {phrase['suspect_phrase']}\n\n"
                report += "---\n\n"

        return report

    def _generate_normalized_text(self, text_data):
        """Generate normalized text from extraction data"""
        output = []
        output.append(f"# ИЗВЛЕЧЁННЫЙ И НОРМАЛИЗОВАННЫЙ ТЕКСТ")
        output.append(f"# Файл: {text_data['source_file']}")
        output.append(f"# Всего страниц: {text_data['total_pages']}")
        output.append(f"# Страниц с текстом: {text_data['pages_with_text']}")
        output.append(f"# Метод извлечения: {', '.join(text_data['extraction_method'])}")
        output.append(f"# Всего символов: {text_data['extraction_stats']['total_characters']:,}")
        output.append("\n" + "="*80 + "\n")

        for page_data in text_data.get('text_pages', []):
            output.append(f"\n--- СТРАНИЦА {page_data['page_number']} ---")
            output.append(f"Метод: {page_data['method']}, Символов: {page_data['char_count']}, Уверенность: {page_data.get('confidence', 1.0):.2f}")
            output.append("")
            output.append(page_data['text'])
            output.append("\n" + "-"*80 + "\n")

        return "\n".join(output)
    
    def _generate_markdown_report(self):
        """Generate markdown format report"""
        source_text = self.results['source_text']
        suspect_text = self.results['suspect_text']
        source_entities = self.results['source_entities']
        suspect_entities = self.results['suspect_entities']
        similarity_analysis = self.results.get('similarity_analysis', {})

        # Extract similarity metrics
        text_similarity = similarity_analysis.get('text_similarity', {})
        overall_similarity = text_similarity.get('overall_similarity', 0.0)
        matching_phrases = text_similarity.get('matching_phrases', [])
        suspicious_matches = text_similarity.get('suspicious_matches', [])

        # Calculate exact and paraphrase matches
        exact_matches = [p for p in matching_phrases if p.get('type') == 'exact']
        paraphrase_matches = [p for p in matching_phrases if p.get('type') == 'paraphrase']

        report = f"""# TEXTBOOK PLAGIARISM ANALYSIS REPORT

**Analysis Date:** {datetime.now().strftime('%d %B %Y, %H:%M')}
**Source Textbook:** {os.path.basename(self.source_path)}
**Suspect Textbook:** {os.path.basename(self.suspect_path)}

## SIMILARITY ANALYSIS

### Overall Text Similarity: {overall_similarity:.1%}

| Metric | Value |
|--------|-------|
| Overall Similarity Score | {overall_similarity:.1%} |
| Suspicious Page Matches | {len(suspicious_matches)} |
| Total Matching Phrases | {len(matching_phrases)} |
| Exact Matches | {len(exact_matches)} |
| Paraphrase Matches | {len(paraphrase_matches)} |

### Interpretation
- **0-10%**: Normal overlap for textbooks on similar topics
- **10-30%**: Requires attention, possible borrowing
- **30%+**: High probability of plagiarism

## TEXT EXTRACTION STATISTICS

| Metric | Source | Suspect | Ratio |
|--------|---------|---------|--------|
| Total Pages | {source_text['total_pages']} | {suspect_text['total_pages']} | {suspect_text['total_pages']/source_text['total_pages']:.2f} |
| Pages with Text | {source_text['pages_with_text']} | {suspect_text['pages_with_text']} | {suspect_text['pages_with_text']/source_text['pages_with_text']:.2f} |
| Total Characters | {source_text['extraction_stats']['total_characters']:,} | {suspect_text['extraction_stats']['total_characters']:,} | {suspect_text['extraction_stats']['total_characters']/source_text['extraction_stats']['total_characters']:.2f} |
| Images | {len(source_text['images'])} | {len(suspect_text['images'])} | {len(suspect_text['images'])/max(len(source_text['images']),1):.2f} |

## ENTITY EXTRACTION STATISTICS

| Entity Type | Source Count | Suspect Count | Ratio |
|-------------|--------------|---------------|--------|
| Questions | {len(source_entities['questions'])} | {len(suspect_entities['questions'])} | {len(suspect_entities['questions'])/max(len(source_entities['questions']),1):.2f} |
| Film References | {len(source_entities['film_references'])} | {len(suspect_entities['film_references'])} | {len(suspect_entities['film_references'])/max(len(source_entities['film_references']),1):.2f} |
| Internet Sources | {len(source_entities['internet_sources'])} | {len(suspect_entities['internet_sources'])} | {len(suspect_entities['internet_sources'])/max(len(source_entities['internet_sources']),1):.2f} |
| Maps/Diagrams | {len(source_entities['maps_and_diagrams'])} | {len(suspect_entities['maps_and_diagrams'])} | {len(suspect_entities['maps_and_diagrams'])/max(len(source_entities['maps_and_diagrams']),1):.2f} |

---
*Analysis generated: {datetime.now().strftime('%d %B %Y, %H:%M')}*
"""
        return report

def main():
    """Main execution function"""
    print("🎯 COMPREHENSIVE TEXTBOOK ANALYSIS PIPELINE")
    print("=" * 60)

    # Define book pairs to analyze
    book_pairs = [
        ("books/6 класс. История России.pdf", "books/6 класс вторая группа.pdf"),
        ("books/7 класс. История России.pdf", "books/7 класс вторая группа.pdf"),
        ("books/8 класс. История России.pdf", "books/8 класс вторая группа.pdf")
    ]

    output_dir = "client"

    for i, (source_pdf, suspect_pdf) in enumerate(book_pairs, 1):
        print(f"\n{'='*70}")
        print(f"📚 АНАЛИЗ ПАРЫ {i}/{len(book_pairs)}")
        print(f"{'='*70}")
        print(f"Источник: {os.path.basename(source_pdf)}")
        print(f"Подозреваемый: {os.path.basename(suspect_pdf)}")

        if not os.path.exists(source_pdf) or not os.path.exists(suspect_pdf):
            print(f"❌ PDF файлы не найдены!")
            continue

        # Initialize analyzer
        analyzer = CachedTextbookAnalyzer(source_pdf, suspect_pdf)

        try:
            # Run pipeline
            results = analyzer.run_full_pipeline()

            # Generate reports
            analyzer.generate_comprehensive_report(output_dir=output_dir)

            print(f"\n✅ Анализ пары {i} завершён!")

        except Exception as e:
            print(f"❌ Анализ не удался: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"🎉 ВСЕ АНАЛИЗЫ ЗАВЕРШЕНЫ!")
    print(f"📁 Все результаты сохранены в: {output_dir}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
