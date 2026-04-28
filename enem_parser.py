import pdfplumber
import re


def extract_column(page, bbox):
    cropped = page.crop(bbox)

    words = cropped.extract_words(
        x_tolerance=2,
        y_tolerance=2,
        keep_blank_chars=False
    )

    if words:
        words.sort(key=lambda w: (round(w["top"], 1), w["x0"]))

        lines = []
        current_line = []
        current_top = None

        for word in words:
            if current_top is None:
                current_top = word["top"]

            if abs(word["top"] - current_top) > 5:
                lines.append(" ".join(current_line))
                current_line = []
                current_top = word["top"]

            current_line.append(word["text"])

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    fallback = cropped.extract_text()
    if fallback:
        return fallback

    return ""


def normalize_text(text):
    # normaliza QUESTAO / QUESTÃO
    text = re.sub(r"QUESTAO", "QUESTÃO", text, flags=re.IGNORECASE)

    # remove lixo repetido tipo EM2024ENEM...
    text = re.sub(r"(EM\d{4})+", "", text)

    # remove sequências muito estranhas (opcional, mas ajuda)
    text = re.sub(r"[A-Z0-9]{20,}", "", text)

    return text


def split_by_question(text):
    pattern = r"(QUEST[ÃA]O\s+\d+.*?)(?=QUEST[ÃA]O\s+\d+|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    return [m.strip() for m in matches if m.strip()]


def extract_text_enem(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        print(f"📄 Total de páginas: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages):
            print(f"➡️ Processando página {i+1}")

            width = page.width
            height = page.height

            left_bbox = (0, 0, width * 0.48, height)
            right_bbox = (width * 0.52, 0, width, height)

            left_text = extract_column(page, left_bbox)
            right_text = extract_column(page, right_bbox)

            page_text = (left_text + "\n" + right_text).strip()

            if not page_text:
                print(f"⚠️ Página {i+1} veio vazia")

            full_text += "\n" + page_text

    return full_text


def extract_questions(text):
    return re.findall(
        r'QUESTÃO\s+\d+[\s\S]*?A[\s\S]*?B[\s\S]*?C[\s\S]*?D[\s\S]*?E[\s\S]*?(?=QUESTÃO\s+\d+|$)',
        text,
        flags=re.IGNORECASE
    )


def parse_question(block):
    num_match = re.search(r'QUEST[ÃA]O\s+(\d+)', block, re.IGNORECASE)
    if not num_match:
        return None
    number = int(num_match.group(1))

    alt_pattern = r'(?m)^([A-E])\s+(.+?)(?=^[A-E]\s+|\Z)'
    alternatives = {}
    alt_start = len(block)
    for m in re.finditer(alt_pattern, block, re.DOTALL):
        letter = m.group(1).lower()
        alt_text = m.group(2).strip()
        alternatives[letter] = alt_text
        alt_start = min(alt_start, m.start())

    if len(alternatives) != 5:
        return None

    text_start = num_match.end()
    text = block[text_start:alt_start].strip()

    return {"number": number, "text": text, "alternatives": alternatives}


def parse_all(text):
    blocks = extract_questions(text)
    parsed = [parse_question(b) for b in blocks]
    return [p for p in parsed if p is not None]


def main():
    pdf_path = "matematica_2024_enem.pdf"

    raw_text = extract_text_enem(pdf_path)

    if not raw_text.strip():
        print("❌ Nenhum texto extraído.")
        return

    # 🔧 normalização leve
    clean_text = normalize_text(raw_text)

    # 🔪 split por questão
    questions = split_by_question(clean_text)

    if not questions:
        print("❌ Nenhuma questão detectada")
        return

    print(f"📊 Total de questões: {len(questions)}")

    # salva debug
    with open("debug_questions.txt", "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            f.write(f"\n\n===== QUESTÃO {i+1} =====\n\n")
            f.write(q)

    print("✅ Questões separadas com sucesso!")


if __name__ == "__main__":
    main()