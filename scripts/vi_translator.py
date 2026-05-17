import modules.scripts as scripts
import gradio as gr
from modules.processing import StableDiffusionProcessing

tokenizer = None
trans_model = None
cache = {}

def load_model():
    global tokenizer, trans_model
    if tokenizer is not None:
        return
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        print("[VI Translator] Đang load model dịch...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        trans_model = MarianMTModel.from_pretrained(model_name)
        print("[VI Translator] Load xong!")
    except Exception as e:
        print(f"[VI Translator] Lỗi load model: {e}")

def translate_vi_en(text):
    if not text or not text.strip():
        return text
    if text in cache:
        return cache[text]
    try:
        import langdetect
        if langdetect.detect(text) != 'vi':
            return text
        load_model()
        if trans_model is None:
            return text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = trans_model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        cache[text] = result
        print(f"[VI→EN] {text}\n     → {result}")
        return result
    except Exception as e:
        print(f"[VI Translator] Lỗi dịch: {e}")
        return text

class ViTranslatorScript(scripts.Script):
    def title(self):
        return "Auto Translate VI→EN"

    def show(self, is_img2img):
        return scripts.AlwaysVisible  # Hiện ở cả Txt2img lẫn Img2img

    def ui(self, is_img2img):
        with gr.Accordion("Auto Translate VI→EN", open=False):
            enabled = gr.Checkbox(label="Tự động dịch tiếng Việt sang tiếng Anh", value=False)
            gr.Markdown("*Dùng Helsinki-NLP/opus-mt-vi-en. Lần đầu chạy sẽ tải model ~300MB.*")
        return [enabled]

    def process(self, p: StableDiffusionProcessing, enabled):
        if not enabled:
            return
        # Xử lý cả trường hợp prompt là string lẫn list (batch)
        if isinstance(p.prompt, list):
            p.prompt = [translate_vi_en(x) for x in p.prompt]
        else:
            p.prompt = translate_vi_en(p.prompt)

        if isinstance(p.negative_prompt, list):
            p.negative_prompt = [translate_vi_en(x) for x in p.negative_prompt]
        else:
            p.negative_prompt = translate_vi_en(p.negative_prompt)