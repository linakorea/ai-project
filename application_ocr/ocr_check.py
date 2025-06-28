import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re
import numpy as np
import cv2

# PDF 파일 경로
PDF_PATH = 'G2505260099602_20250527093430.pdf'

# 계약자 이름 (필요시 자동 추출 또는 파라미터화 가능)
CONTRACTOR_NAME = '한진희'

# 동의 확인란 키워드 (예시: '동의', '확인', '체크' 등)
CHECKBOX_KEYWORDS = [
    '동의', '확인', '체크', '동의함', '동의합니다', '동의란', '동의 여부'
]

# 체크박스 패턴 (OCR 결과에서 체크박스는 보통 □, ■, ✓, ✔, ☑ 등으로 인식됨)
CHECKBOX_PATTERNS = [
    r'\[\s*\]',   # [ ]
    r'\[x\]',      # [x]
    r'\[X\]',      # [X]
    r'□', r'■', r'✓', r'✔', r'☑',
]

# 체크된 박스 패턴 (체크된 것으로 인식되는 문자)
CHECKED_PATTERNS = [r'■', r'✓', r'✔', r'☑', r'\[x\]', r'\[X\]']


def pdf_to_images(pdf_path: str):
    """PDF 파일을 이미지 리스트로 변환"""
    return convert_from_path(pdf_path)


def extract_text_from_image(image: Image.Image) -> str:
    """이미지에서 Tesseract로 텍스트 추출"""
    return pytesseract.image_to_string(image, lang='kor+eng')


def analyze_lines(text: str):
    """동의 관련 줄만 추출 및 체크박스/체크여부 분석"""
    results = []
    for line in text.split('\n'):
        if any(keyword in line for keyword in CHECKBOX_KEYWORDS):
            checkbox = any(re.search(pattern, line) for pattern in CHECKBOX_PATTERNS)
            checked = any(re.search(pattern, line) for pattern in CHECKED_PATTERNS)
            results.append({'text': line, 'checkbox': checkbox, 'checked': checked})
    return results


def print_ocr_and_checkbox_results(page_result):
    print("="*60)
    print("[첫 페이지] OCR 및 체크박스 검증 결과 요약")
    print("="*60)
    if not page_result['lines']:
        print("  동의 관련 문구 없음")
    else:
        for line_info in page_result['lines']:
            preview = line_info['text'][:40].replace('\n', ' ')
            print(f"  - 행: '{preview}...' | 체크박스: {line_info['checkbox']} | 체크여부: {'✔' if line_info['checked'] else '✘'}")
    print("="*60)


def pdf_to_first_image(pdf_path):
    # 해상도 높이기(dpi=300)
    images = convert_from_path(pdf_path, dpi=300)
    return images[0]


def preprocess_image(image):
    # 흑백 변환 및 이진화
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, 10)
    return Image.fromarray(proc)


def find_signature_box(image, name=CONTRACTOR_NAME):
    data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    name_idx = None
    # 이름의 각 글자(성, 이름1, 이름2)만 추출
    name_chars = [c for c in name if '\uac00' <= c <= '\ud7a3']
    # 정규표현식: 한글 3글자 + 중간에 * 또는 · 또는 아무거나 허용
    if len(name_chars) == 3:
        name_pattern = f'{name_chars[0]}[^가-힣]{{0,2}}{name_chars[1]}[^가-힣]{{0,2}}{name_chars[2]}'
    else:
        name_pattern = name
    for i in range(n_boxes):
        txt = data['text'][i].replace(' ', '')
        if re.search(name_pattern, txt):
            name_idx = i
            break
    if name_idx is None:
        print(f'이름({name})을 찾을 수 없습니다.')
        return None
    name_x, name_y, name_w, name_h = data['left'][name_idx], data['top'][name_idx], data['width'][name_idx], data['height'][name_idx]
    for i in range(n_boxes):
        if '(서명)' in data['text'][i]:
            sign_x, sign_y, sign_w, sign_h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            if abs(sign_y - name_y) < name_h and sign_x > name_x:
                return (sign_x, sign_y, sign_w, sign_h)
    print('(서명)란을 찾을 수 없습니다.')
    return None


def is_signed(image, box):
    x, y, w, h = box
    crop = np.array(image)[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    ink_ratio = np.sum(thresh > 0) / thresh.size
    return ink_ratio > 0.01


def main():
    if not os.path.exists(PDF_PATH):
        print(f'PDF 파일을 찾을 수 없습니다: {PDF_PATH}')
        return
    image = pdf_to_first_image(PDF_PATH)
    proc_image = preprocess_image(image)
    # OCR 전체 결과 미리보기
    ocr_text = pytesseract.image_to_string(proc_image, lang='kor+eng')
    print('==== OCR 전체 결과 미리보기 (일부) ====')
    print(ocr_text[:1000])
    print('=======================================')
    box = find_signature_box(proc_image, name=CONTRACTOR_NAME)
    print('===============================================')
    print('계약자 (서명)란 서명 검증 결과')
    print('===============================================')
    if box:
        signed = is_signed(proc_image, box)
        print(f'서명란 서명 여부: {"✔ 서명 있음" if signed else "✘ 서명 없음"}')
    else:
        print('서명란을 찾을 수 없습니다.')
    print('===============================================')
    # 첫 페이지만 OCR 및 동의 체크박스 검증
    text = extract_text_from_image(proc_image)
    lines = analyze_lines(text)
    print_ocr_and_checkbox_results({'lines': lines})
    checked = any(line['checked'] for line in lines)
    if checked:
        print('\n[최종 결과] 동의 확인란이 체크되어 있습니다.')
    else:
        print('\n[최종 결과] 동의 확인란이 체크되어 있지 않습니다.')


if __name__ == '__main__':
    main() 