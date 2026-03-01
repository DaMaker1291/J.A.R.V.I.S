"""
J.A.S.O.N. Visual Replication Engine
Coordinate-based handwriting drawing with style matching, human-error variance,
and iterative visual debugging loop.
"""

import os
import json
import math
import random
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Image processing
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    pil_available = True
except ImportError:
    pil_available = False

# Gemini AI for style analysis & verification
try:
    import google.generativeai as genai
    genai_available = True
except ImportError:
    genai_available = False

logger = logging.getLogger(__name__)


class HandwritingStyleProfile:
    """Captures and stores handwriting style characteristics"""

    def __init__(self):
        self.slant_degrees: float = 0.0  # -15 to +15 degrees
        self.slant_direction: str = "right"  # left, right, vertical
        self.stroke_weight: str = "medium"  # thin, medium, thick, variable
        self.pressure: str = "medium"  # light, medium, heavy
        self.text_spacing: str = "normal"  # tight, normal, loose
        self.style_type: str = "printed"  # cursive, printed, mixed
        self.inconsistency_level: float = 0.3  # 0.0 (perfect) to 1.0 (very messy)
        self.letter_size_px: int = 22
        self.ink_color: Tuple[int, int, int] = (25, 25, 80)  # Dark blue
        self.baseline_drift: float = 0.15  # How much baseline wanders
        self.letter_spacing_variance: float = 0.2
        self.word_spacing_px: int = 8
        self.line_height_px: int = 30

    def to_dict(self) -> dict:
        return {
            'slant_degrees': self.slant_degrees,
            'slant_direction': self.slant_direction,
            'stroke_weight': self.stroke_weight,
            'pressure': self.pressure,
            'text_spacing': self.text_spacing,
            'style_type': self.style_type,
            'inconsistency_level': self.inconsistency_level,
            'letter_size_px': self.letter_size_px,
            'ink_color': self.ink_color,
            'baseline_drift': self.baseline_drift,
            'letter_spacing_variance': self.letter_spacing_variance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'HandwritingStyleProfile':
        profile = cls()
        for key, value in data.items():
            if hasattr(profile, key):
                if key == 'ink_color' and isinstance(value, list):
                    setattr(profile, key, tuple(value))
                else:
                    setattr(profile, key, value)
        return profile


class VisualReplicationEngine:
    """
    Visual Replication Engine for J.A.R.V.I.S.
    Analyzes handwriting styles and replicates them with human-like variance.
    Includes iterative visual debugging loop for quality assurance.
    """

    # macOS system fonts that look handwritten
    HANDWRITING_FONTS = [
        "/System/Library/Fonts/MarkerFelt.ttc",
        "/System/Library/Fonts/Supplemental/AppleChancery.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
        "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
        "/System/Library/Fonts/Supplemental/Chalkboard SE.ttc",
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.api_key = self.config.get('api_keys', {}).get('gemini', '')

        # Engine state
        self.current_style = HandwritingStyleProfile()
        self.debug_iterations = 0
        self.max_debug_iterations = 5

        # Output directory
        self.output_dir = Path(os.path.dirname(__file__)) / '..' / '..' / 'output' / 'handwriting'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find available handwriting fonts
        self.available_fonts = []
        for font_path in self.HANDWRITING_FONTS:
            if os.path.exists(font_path):
                self.available_fonts.append(font_path)

        if not self.available_fonts:
            logger.warning("No handwriting fonts found, will use default font")

        # Stats
        self.stats = {
            'worksheets_processed': 0,
            'debug_loops_total': 0,
            'average_quality_score': 0.0,
        }

        logger.info(f"Visual Replication Engine initialized (fonts: {len(self.available_fonts)})")

    def _get_model(self):
        """Get Gemini model for analysis"""
        if not genai_available:
            raise RuntimeError("google-generativeai not available")
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel('gemini-2.0-flash')

    # ─── STYLE ANALYSIS ──────────────────────────────────────────

    def analyze_handwriting_style(self, reference_image_path: str) -> HandwritingStyleProfile:
        """
        Analyze a reference image to extract handwriting style parameters.
        Uses Gemini Vision to understand slant, pressure, spacing, etc.
        """
        if not os.path.exists(reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        logger.info(f"Analyzing handwriting style from: {reference_image_path}")

        try:
            model = self._get_model()
            img = Image.open(reference_image_path)

            prompt = """Analyze the handwriting style in this image with extreme precision.
Return a JSON object with these EXACT fields:
{
    "slant_degrees": <float, -15 to +15, negative=left lean, positive=right lean>,
    "slant_direction": "<left|right|vertical>",
    "stroke_weight": "<thin|medium|thick|variable>",
    "pressure": "<light|medium|heavy>",
    "text_spacing": "<tight|normal|loose>",
    "style_type": "<cursive|printed|mixed>",
    "inconsistency_level": <float 0.0-1.0, how inconsistent/human the writing is>,
    "letter_size_px": <int, estimated letter height in pixels>,
    "ink_color_rgb": [<r>, <g>, <b>],
    "baseline_drift": <float 0.0-0.5, how much the baseline wanders>,
    "letter_spacing_variance": <float 0.0-0.5, variance in spacing between letters>,
    "observations": "<any additional notes about unique characteristics>"
}

Only return valid JSON, no markdown formatting."""

            response = model.generate_content([prompt, img])
            text = response.text.strip()

            # Clean JSON
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            style_data = json.loads(text)

            # Build profile
            profile = HandwritingStyleProfile()
            profile.slant_degrees = float(style_data.get('slant_degrees', 0))
            profile.slant_direction = style_data.get('slant_direction', 'vertical')
            profile.stroke_weight = style_data.get('stroke_weight', 'medium')
            profile.pressure = style_data.get('pressure', 'medium')
            profile.text_spacing = style_data.get('text_spacing', 'normal')
            profile.style_type = style_data.get('style_type', 'printed')
            profile.inconsistency_level = float(style_data.get('inconsistency_level', 0.3))
            profile.letter_size_px = int(style_data.get('letter_size_px', 22))
            if 'ink_color_rgb' in style_data:
                profile.ink_color = tuple(style_data['ink_color_rgb'][:3])
            profile.baseline_drift = float(style_data.get('baseline_drift', 0.15))
            profile.letter_spacing_variance = float(style_data.get('letter_spacing_variance', 0.2))

            self.current_style = profile
            logger.info(f"Style analyzed: {profile.style_type}, slant={profile.slant_degrees}°, "
                        f"pressure={profile.pressure}")

            return profile

        except Exception as e:
            logger.error(f"Style analysis failed: {e}")
            # Return default profile
            return HandwritingStyleProfile()

    # ─── WORKSHEET ANALYSIS ───────────────────────────────────────

    def analyze_worksheet(self, blank_image_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a blank worksheet to find fill-in regions.
        Returns list of {question, coords, answer, line_width} items.
        """
        if not os.path.exists(blank_image_path):
            raise FileNotFoundError(f"Worksheet image not found: {blank_image_path}")

        logger.info(f"Analyzing worksheet layout: {blank_image_path}")

        try:
            model = self._get_model()
            img = Image.open(blank_image_path)
            width, height = img.size

            prompt = f"""Analyze this worksheet image ({width}x{height} pixels).
Identify ALL blank lines, fill-in areas, and answer spaces.

For EACH blank/answer area, provide:
1. The question or prompt text near the blank
2. The pixel coordinates (x, y) of where text should START being written
3. The expected answer based on the subject matter
4. The approximate width in pixels available for writing

Return as a JSON array:
[
    {{
        "question": "text of the question/prompt",
        "coords": [x, y],
        "answer": "the correct answer to write",
        "line_width_px": estimated width available,
        "line_number": 1
    }}
]

IMPORTANT: Coordinates must be precise pixel positions on the blank lines.
Only return valid JSON, no markdown formatting."""

            response = model.generate_content([prompt, img])
            text = response.text.strip()

            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            fill_data = json.loads(text)

            logger.info(f"Found {len(fill_data)} fill regions in worksheet")
            return fill_data

        except Exception as e:
            logger.error(f"Worksheet analysis failed: {e}")
            return []

    # ─── HANDWRITING RENDERING ────────────────────────────────────

    def render_handwriting(
        self,
        base_image_path: str,
        fill_data: List[Dict],
        style: HandwritingStyleProfile = None,
        output_path: str = None,
    ) -> str:
        """
        Render handwritten text onto a worksheet image.
        Uses coordinate-based placement with human-error variance.
        """
        if not pil_available:
            raise RuntimeError("PIL/Pillow not available")

        style = style or self.current_style
        if output_path is None:
            output_path = str(self.output_dir / f"filled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        logger.info(f"Rendering handwriting onto: {base_image_path}")

        # Load base image
        img = Image.open(base_image_path).convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Select font
        font = self._select_font(style)

        for item in fill_data:
            answer = str(item.get('answer', ''))
            coords = item.get('coords', [0, 0])
            x_start, y_start = int(coords[0]), int(coords[1])

            # Apply human-like variance to the entire text block
            self._draw_handwritten_text(
                draw, answer, x_start, y_start, font, style
            )

        # Composite overlay onto base
        result = Image.alpha_composite(img, overlay)
        result = result.convert('RGB')

        # Apply subtle post-processing for realism
        result = self._apply_realism_filters(result, style)

        result.save(output_path, quality=95)
        logger.info(f"Handwritten worksheet saved: {output_path}")

        return output_path

    def _select_font(self, style: HandwritingStyleProfile) -> ImageFont.FreeTypeFont:
        """Select and configure font based on style profile"""
        font_size = style.letter_size_px

        # Choose font based on style
        font_path = None
        if style.style_type == 'cursive' and self.available_fonts:
            # Prefer chancery/script fonts
            for fp in self.available_fonts:
                if 'Chancery' in fp or 'Noteworthy' in fp or 'Bradley' in fp:
                    font_path = fp
                    break
        elif style.style_type == 'printed' and self.available_fonts:
            for fp in self.available_fonts:
                if 'Marker' in fp or 'Chalkboard' in fp or 'Comic' in fp:
                    font_path = fp
                    break

        if font_path is None and self.available_fonts:
            font_path = self.available_fonts[0]

        try:
            if font_path:
                return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass

        return ImageFont.load_default()

    def _draw_handwritten_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        style: HandwritingStyleProfile,
    ):
        """Draw text character by character with human-like imperfections"""
        variance = style.inconsistency_level
        cursor_x = x
        cursor_y = y

        # Base ink color with slight random tint
        r, g, b = style.ink_color

        for i, char in enumerate(text):
            if char == ' ':
                space_width = style.word_spacing_px + random.randint(-2, 2)
                cursor_x += space_width
                continue

            # Per-character variance
            char_y_offset = random.gauss(0, variance * 3)  # Baseline drift
            char_x_offset = random.gauss(0, variance * 1.5)  # Horizontal jitter

            # Slight size variation
            size_variance = 1 + random.gauss(0, variance * 0.08)
            current_size = max(8, int(style.letter_size_px * size_variance))

            # Per-character font (for size variation)
            try:
                char_font = font.font_variant(size=current_size)
            except Exception:
                char_font = font

            # Ink color variance (simulates pressure changes)
            pressure_var = random.gauss(0, variance * 15)
            char_r = max(0, min(255, int(r + pressure_var)))
            char_g = max(0, min(255, int(g + pressure_var)))
            char_b = max(0, min(255, int(b + pressure_var)))

            # Opacity variation (pressure simulation)
            opacity = max(150, min(255, int(230 + random.gauss(0, variance * 25))))
            char_color = (char_r, char_g, char_b, opacity)

            # Draw the character
            char_x = int(cursor_x + char_x_offset)
            char_y = int(cursor_y + char_y_offset)

            draw.text((char_x, char_y), char, font=char_font, fill=char_color)

            # Simulate stroke weight by drawing again with slight offset
            if style.stroke_weight in ('thick', 'variable'):
                weight_extra = 1 if style.stroke_weight == 'thick' else (1 if random.random() > 0.5 else 0)
                if weight_extra:
                    lighter_color = (char_r, char_g, char_b, max(80, opacity - 80))
                    draw.text((char_x + 1, char_y), char, font=char_font, fill=lighter_color)

            # Advance cursor
            try:
                bbox = char_font.getbbox(char)
                char_width = bbox[2] - bbox[0]
            except Exception:
                char_width = current_size * 0.6

            spacing = char_width + random.gauss(0, variance * 2)
            # Text spacing modifier
            if style.text_spacing == 'tight':
                spacing *= 0.85
            elif style.text_spacing == 'loose':
                spacing *= 1.2

            cursor_x += max(3, spacing)

    def _apply_realism_filters(self, img: Image.Image, style: HandwritingStyleProfile) -> Image.Image:
        """Apply subtle filters to make the text look more natural"""
        # Slight Gaussian blur to simulate ink bleed
        if style.pressure == 'heavy':
            img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        elif style.pressure == 'light':
            # Slightly reduce contrast for light pressure
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.97)

        # Very subtle noise for paper texture realism
        np_img = np.array(img)
        noise = np.random.normal(0, 1.5, np_img.shape).astype(np.int8)
        np_img = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

        return img

    # ─── VISUAL DEBUGGING LOOP ────────────────────────────────────

    def verify_output(self, result_path: str, reference_path: str = None) -> Dict[str, Any]:
        """
        Visual debugging verification step.
        Compares rendered output against reference and identifies issues.
        """
        if not genai_available:
            return {'status': 'SKIP', 'reason': 'Gemini not available for verification'}

        logger.info("Starting visual verification loop...")

        try:
            model = self._get_model()
            result_img = Image.open(result_path)

            images = [result_img]
            ref_note = ""

            if reference_path and os.path.exists(reference_path):
                ref_img = Image.open(reference_path)
                images.append(ref_img)
                ref_note = "Compare the generated worksheet (Image 1) with the reference handwriting style (Image 2)."

            prompt = f"""You are a visual QA inspector for handwritten worksheet generation.
{ref_note}

Analyze the generated image for:
1. TEXT ALIGNMENT: Is text properly placed on blank lines? Not floating above or below?
2. STYLE CONSISTENCY: Does the handwriting look consistent and natural throughout?
3. REALISM: Does it look like a human wrote it, or does it appear computer-generated?
4. COMPLETENESS: Are all blanks filled in?
5. READABILITY: Is the text readable and clear?

Score each category 1-10 and provide an overall PASS/FAIL.

Return ONLY valid JSON:
{{
    "status": "PASS" or "FAIL",
    "overall_score": <1-10>,
    "alignment_score": <1-10>,
    "style_score": <1-10>,
    "realism_score": <1-10>,
    "completeness_score": <1-10>,
    "readability_score": <1-10>,
    "discrepancies": ["list of specific issues found"],
    "adjustments": {{
        "x_offset": <int pixels to shift horizontally>,
        "y_offset": <int pixels to shift vertically>,
        "font_size_mult": <float multiplier for font size>,
        "increase_variance": <bool, should more human error be added>,
        "darken_ink": <bool, should ink be darker>
    }}
}}"""

            response = model.generate_content([prompt] + images)
            text = response.text.strip()

            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            verification = json.loads(text)
            self.debug_iterations += 1

            logger.info(f"Verification result: {verification.get('status')} "
                        f"(score: {verification.get('overall_score', 'N/A')}/10)")

            return verification

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'overall_score': 0,
            }

    def apply_corrections(
        self,
        base_image_path: str,
        fill_data: List[Dict],
        adjustments: Dict[str, Any],
        style: HandwritingStyleProfile = None,
        output_path: str = None,
    ) -> str:
        """Apply corrections from the visual debugging loop and re-render"""
        style = style or self.current_style

        # Apply adjustments
        if adjustments.get('x_offset', 0) != 0 or adjustments.get('y_offset', 0) != 0:
            x_off = adjustments.get('x_offset', 0)
            y_off = adjustments.get('y_offset', 0)
            for item in fill_data:
                item['coords'][0] += x_off
                item['coords'][1] += y_off

        if adjustments.get('font_size_mult', 1.0) != 1.0:
            style.letter_size_px = int(style.letter_size_px * adjustments['font_size_mult'])

        if adjustments.get('increase_variance', False):
            style.inconsistency_level = min(0.8, style.inconsistency_level + 0.1)

        if adjustments.get('darken_ink', False):
            r, g, b = style.ink_color
            style.ink_color = (max(0, r - 20), max(0, g - 20), max(0, b - 20))

        # Re-render
        return self.render_handwriting(base_image_path, fill_data, style, output_path)

    # ─── FULL PIPELINE ─────────────────────────────────────────────

    def fill_worksheet_pipeline(
        self,
        blank_worksheet_path: str,
        reference_style_path: str = None,
        custom_answers: Dict[str, str] = None,
        output_path: str = None,
    ) -> Dict[str, Any]:
        """
        Full autonomous pipeline:
        1. Analyze reference handwriting style
        2. Analyze blank worksheet layout
        3. Render handwritten answers
        4. Visual debugging loop until quality threshold met
        5. Return proof-of-life result
        """
        result = {
            'status': 'processing',
            'timestamp': datetime.now().isoformat(),
            'debug_iterations': 0,
            'quality_score': 0,
            'output_path': None,
        }

        # Step 1: Analyze style
        if reference_style_path:
            logger.info("Step 1: Analyzing handwriting style...")
            style = self.analyze_handwriting_style(reference_style_path)
        else:
            style = HandwritingStyleProfile()
            # Use natural-looking defaults
            style.inconsistency_level = 0.35
            style.slant_degrees = random.uniform(-5, 8)
            style.pressure = random.choice(['medium', 'heavy'])

        result['style_profile'] = style.to_dict()

        # Step 2: Analyze worksheet
        logger.info("Step 2: Analyzing worksheet layout...")
        fill_data = self.analyze_worksheet(blank_worksheet_path)

        if not fill_data:
            result['status'] = 'error'
            result['error'] = 'No fill regions found in worksheet'
            return result

        # Override with custom answers if provided
        if custom_answers:
            for item in fill_data:
                q = item.get('question', '')
                for pattern, answer in custom_answers.items():
                    if pattern.lower() in q.lower():
                        item['answer'] = answer

        result['fill_data'] = fill_data

        # Step 3: Render
        logger.info("Step 3: Rendering handwritten text...")
        if output_path is None:
            output_path = str(self.output_dir / f"worksheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        rendered_path = self.render_handwriting(blank_worksheet_path, fill_data, style, output_path)

        # Step 4: Visual debugging loop
        logger.info("Step 4: Starting visual debugging loop...")
        iteration = 0
        best_score = 0

        while iteration < self.max_debug_iterations:
            iteration += 1
            result['debug_iterations'] = iteration

            verification = self.verify_output(rendered_path, reference_style_path)
            score = verification.get('overall_score', 0)

            logger.info(f"Debug iteration {iteration}: score={score}/10, "
                        f"status={verification.get('status')}")

            if score > best_score:
                best_score = score

            # Pass threshold: score >= 7 or status is PASS
            if verification.get('status') == 'PASS' or score >= 7:
                logger.info(f"✅ Quality threshold met (score: {score}/10)")
                break

            # Apply corrections and re-render
            if verification.get('adjustments'):
                logger.info(f"Applying corrections: {verification['adjustments']}")
                rendered_path = self.apply_corrections(
                    blank_worksheet_path,
                    fill_data,
                    verification['adjustments'],
                    style,
                    output_path,
                )
            else:
                break  # No adjustments suggested, stop looping

        result['status'] = 'complete'
        result['quality_score'] = best_score
        result['output_path'] = rendered_path
        result['verification'] = verification

        # Update stats
        self.stats['worksheets_processed'] += 1
        self.stats['debug_loops_total'] += iteration
        total = self.stats['worksheets_processed']
        self.stats['average_quality_score'] = (
            (self.stats['average_quality_score'] * (total - 1) + best_score) / total
        )

        logger.info(f"Pipeline complete: {result['output_path']} (score: {best_score}/10, "
                     f"iterations: {iteration})")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            'available_fonts': len(self.available_fonts),
            'current_style': self.current_style.to_dict(),
        }
