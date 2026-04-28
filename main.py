import json
import sys
from enem_parser import extract_text_enem, split_by_question
from build_prompt import build_prompt
from call_api import call_local_ai


def get_metadata_from_filename(pdf_file):
    # expects format like "matematica_2024_enem.pdf"
    parts = pdf_file.replace(".pdf", "").split("_")
    area = parts[0] if len(parts) > 0 else "unknown"
    year = parts[1] if len(parts) > 1 else "unknown"
    return area, year


def clean_response(response):
    import re
    match = re.search(r'\[.*\]', response, re.DOTALL)
    return match.group(0) if match else None


def process_pdf(pdf_file):
    print(f"\n📄 Processando: {pdf_file}")

    area, year = get_metadata_from_filename(pdf_file)

    text = extract_text_enem(pdf_file)

    if not text.strip():
        print("❌ Texto vazio")
        return

    chunks = split_by_question(text)

    if not chunks:
        print("❌ Nenhuma questão detectada")
        return

    print(f"📊 Total de chunks: {len(chunks)}")

    all_questions = []

    # limpa arquivos de debug antes
    open("debug_chunks.txt", "w").close()
    open("debug_prompts.txt", "w").close()
    open("debug_responses.txt", "w").close()

    for i, chunk in enumerate(chunks):
        print(f"🤖 Chunk {i+1}/{len(chunks)}")

        
        # salva chunk
        with open("debug_chunks.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n===== CHUNK {i+1} =====\n\n")
            f.write(chunk)

        prompt = build_prompt(chunk)

        # salva prompt
        with open("debug_prompts.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n===== PROMPT {i+1} =====\n\n")
            f.write(prompt)

        response = call_local_ai(prompt)

        # salva resposta
        with open("debug_responses.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n===== RESPONSE {i+1} =====\n\n")
            f.write(response if response else "EMPTY")

        if not response or len(response.strip()) < 5:
            print("⚠️ Resposta vazia, pulando...")
            continue

        cleaned = clean_response(response)

        if not cleaned:
            print("⚠️ Não encontrou JSON")
            continue

        try:
            data = json.loads(cleaned)
            all_questions.extend(data)
        except Exception as e:
            print("❌ JSON inválido:", e)
            print(cleaned[:200])

    output_file = f"questions_{area}_{year}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"✅ Salvo em {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <arquivo.pdf>")
        sys.exit(1)
    process_pdf(sys.argv[1])