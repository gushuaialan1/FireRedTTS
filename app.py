import gradio as gr
import os
import torchaudio
from fireredtts.fireredtts import FireRedTTS
from datetime import datetime

# 初始化 FireRedTTS
try:
    print("Initializing FireRedTTS...")
    tts = FireRedTTS(
        config_path="configs/config_24k.json",
        pretrained_path="./pretrained_models", 
    )
    print("FireRedTTS initialized successfully.")
except Exception as e:
    print(f"Error initializing FireRedTTS: {e}")
    raise e

# 创建输出目录
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# 定义生成 TTS 音频的函数
def generate_tts(prompt_wav_path, text, lang):
    try:
        print(f"Input prompt_wav_path: {prompt_wav_path}")
        print(f"Input text: {text}")
        print(f"Input lang: {lang}")
        
        rec_wavs = tts.synthesize(
            prompt_wav=prompt_wav_path,
            text=text,
            lang=lang,
        )
        print("Synthesis complete. Detaching audio...")
        rec_wavs = rec_wavs.detach().cpu()
        
        # 使用时间戳命名文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_wav_path = os.path.abspath(os.path.join(output_dir, f"output_{timestamp}.wav"))
        
        print(f"Saving output audio to {out_wav_path}...")
        torchaudio.save(out_wav_path, rec_wavs, 24000)
        print("Audio saved successfully.")
        
        # 返回相对路径供 Gradio 加载
        relative_path = os.path.relpath(out_wav_path, ".")
        print(f"Returning relative path: {relative_path}")
        return relative_path
    except Exception as e:
        error_message = f"Error during TTS synthesis: {str(e)}"
        print(error_message)
        return None

# 创建 Gradio 界面
with gr.Blocks() as app:
    gr.Markdown("## FireRedTTS 示例")
    
    with gr.Row():
        input_prompt_wav = gr.Audio(label="提示音频 (wav 格式)", type="filepath")
        input_text = gr.Textbox(label="输入文本", placeholder="请输入需要合成的文本")
        input_lang = gr.Textbox(label="语言", value="zh", placeholder="默认 zh (中文)")
    
    output_audio = gr.Audio(label="生成的音频", type="filepath")
    
    submit_button = gr.Button("生成 TTS 音频")
    submit_button.click(
        fn=generate_tts,
        inputs=[input_prompt_wav, input_text, input_lang],
        outputs=output_audio,  # 确保返回音频文件路径供预览和下载
    )

# 运行 Gradio 应用
print("Launching Gradio app...")
app.launch(server_name="0.0.0.0", server_port=6006)
print("Gradio app running on port 6006...")
