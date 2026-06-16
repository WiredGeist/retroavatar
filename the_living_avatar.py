import os
import time
import cv2
import serial
import struct
import threading
import textwrap
import wave
import json
import io
import subprocess
import numpy as np
from pathlib import Path  # Dynamic user home path resolution
from dotenv import load_dotenv  # Load configurations from .env
from google import genai
from google.genai import types
from luma.oled.device import ssd1351
from luma.core.interface.serial import spi
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
import ollama
from vosk import Model, KaldiRecognizer  # Local 32-bit STT

# --- MODULAR TOOLS IMPORT ---
from mcp_tools.sensors import read_sensor_data  # Sensor telemetry
from mcp_tools.internet import get_current_weather, search_wikipedia  # Internet tools

# --- PORTABLE INITIALIZATION ---
# Force absolute path loading of the .env file (foolproof for systemd)
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Find where this python file lives
USER_HOME = str(Path.home())

# SELF-HEALING: If run as root (via systemd), Path.home() returns "/root".
if USER_HOME == "/root" and SCRIPT_DIR.startswith("/home/"):
    parts = SCRIPT_DIR.split("/")
    if len(parts) >= 3:
        USER_HOME = f"/home/{parts[2]}"

# --- CONFIGURATION (With Dynamic Fallbacks) ---
SOURCE_MODEL = os.getenv("SOURCE_MODEL", "OLLAMA")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
API_KEY = os.getenv("GEMINI_API_KEY", "")
AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "default")

# Paths automatically adjust to wherever the script is cloned
SPRITE_DIR = os.getenv("SPRITE_DIR", f"{SCRIPT_DIR}/sprites")
CASCADE_PATH = os.getenv("CASCADE_PATH", f"{SCRIPT_DIR}/haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.getenv("EYE_CASCADE_PATH", f"{SCRIPT_DIR}/haarcascade_eye_tree_eyeglasses.xml")
PORT = os.getenv("SERIAL_PORT", "/dev/ttyACM0")
BAUD = int(os.getenv("SERIAL_BAUD", 2000000))

# PIPER TTS PATHS (Dynamically resolved to actual user's home)
PIPER_BINARY = os.getenv("PIPER_BINARY", f"{USER_HOME}/piper/piper/piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", f"{USER_HOME}/piper/en_US-amy-low.onnx")
PIPER_LIB = os.getenv("PIPER_LIB", f"{USER_HOME}/piper/piper")

# CALIBRATION
MIC_THRESHOLD = 3000  
GAIN_BOOST = 25.0     
RECORD_TIME = 5.0     

# --- HARDWARE STATUS CORES ---
OLED_OK = False
CAMERA_OK = False
XIAO_OK = False
VOSK_OK = False

# --- 1. INITIALIZE OLED SCREEN ---
try:
    oled_spi = spi(device=0, port=0, baudrate=24000000, gpio_DC=25, gpio_RST=27)
    device = ssd1351(oled_spi, width=128, height=128, bgr=True)
    OLED_OK = True
except Exception as e:
    print(f"[BOOT ERROR] OLED Initialization failed: {e}")
    device = None

# --- 2. INITIALIZE CAMERA ---
try:
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    config["main"]["size"] = (320, 320)
    config["main"]["format"] = "BGR888"
    picam2.configure(config)
    picam2.start()
    CAMERA_OK = True
except Exception as e:
    print(f"[BOOT ERROR] Camera Initialization failed: {e}")
    picam2 = None

# --- 3. INITIALIZE XIAO MICROCONTROLLER SERIAL ---
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    XIAO_OK = True
except Exception as e:
    print(f"[BOOT ERROR] Seeeduino Xiao not found on {PORT}: {e}")
    ser = None

# --- 4. INITIALIZE VOSK STT MODEL & BACKGROUND WAKE-RECOGNIZER ---
try:
    VOSK_MODEL_PATH = os.path.join(SCRIPT_DIR, "vosk-model")
    if SOURCE_MODEL == "OLLAMA":
        print(f"Loading local Vosk STT model from: {VOSK_MODEL_PATH}...")
        if os.path.exists(VOSK_MODEL_PATH):
            stt_model = Model(VOSK_MODEL_PATH)
            # Create dedicated background listener for real-time wake word checks
            bg_rec = KaldiRecognizer(stt_model, 16000)
            bg_rec.SetWords(False)
            VOSK_OK = True
        else:
            print("[BOOT WARNING] Vosk model directory missing!")
            stt_model = None
            bg_rec = None
    else:
        stt_model = None
        bg_rec = None
except Exception as e:
    print(f"[BOOT ERROR] Vosk initialization failed: {e}")
    stt_model = None
    bg_rec = None

# --- 5. INITIALIZE & TEST AUDIO OUTPUT DEVICE ---
try:
    subprocess.run(["aplay", "-D", AUDIO_DEVICE, "-d", "1", "/dev/zero"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[AUDIO] Using configured output device: {AUDIO_DEVICE}")
except Exception:
    print(f"[AUDIO WARNING] Device '{AUDIO_DEVICE}' is offline. Falling back to default system audio (AUX/HDMI)...")
    AUDIO_DEVICE = "default"

# Initialize Ollama / Gemini Clients
if SOURCE_MODEL == "GEMINI":
    client = genai.Client(api_key=API_KEY)
else:
    client = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if SOURCE_MODEL == "OLLAMA":
    ollama_client = ollama.Client(host=OLLAMA_HOST)
else:
    ollama_client = None

spi_lock = threading.Lock()

# Load cascading trackers
face_cascade = cv2.CascadeClassifier(CASCADE_PATH) if os.path.exists(CASCADE_PATH) else None
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH) if os.path.exists(EYE_CASCADE_PATH) else None

# --- GLOBAL STATES ---
class GlobalState:
    running = True
    mode = "AVATAR"  
    shared_sensors = {"a": [0,0,0], "g": [0,0,0], "m": 0}
    audio_buffer = []
    is_recording = False
    face_detected = False
    current_emotion = "neutral"
    last_vision_image = None  # Saved to let local portrait tools access the frame
    portrait_img = None

gs = GlobalState()

# --- SPRITES ---
def load_sprites_from_folder(folder_name):
    frames = []
    path = os.path.join(SPRITE_DIR, folder_name)
    if os.path.exists(path):
        files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
        for f in files:
            img = Image.open(os.path.join(path, f)).convert("RGB").resize((128,128), Image.NEAREST)
            frames.append(img)
    if not frames:
        frames.append(Image.new("RGB", (128,128), (50,50,50)))
    return frames

idle_frames = load_sprites_from_folder("idle")
talk_frames = load_sprites_from_folder("talk")

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
except:
    font = ImageFont.load_default()

# --- DIAGNOSTICS DISPLAY ---
def show_system_check(oled_ok, camera_ok, xiao_ok, vosk_ok):
    """Draws visual diagnostic checklist on boot if OLED is connected."""
    img = Image.new("RGB", (128, 128), "black")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "SYSTEM INITS:", fill="yellow")
    
    draw.text((5, 25), "OLED:", fill="white")
    draw.text((65, 25), "OK" if oled_ok else "ERR", fill="lime" if oled_ok else "red")
    
    draw.text((5, 45), "CAMERA:", fill="white")
    draw.text((65, 45), "OK" if camera_ok else "MISSING", fill="lime" if camera_ok else "yellow")
    
    draw.text((5, 65), "XIAO MIC:", fill="white")
    draw.text((65, 65), "OK" if xiao_ok else "FALLBACK", fill="lime" if xiao_ok else "yellow")
    
    draw.text((5, 85), "VOSK STT:", fill="white")
    draw.text((65, 85), "OK" if vosk_ok else "MISSING", fill="lime" if vosk_ok else "red")
    
    if oled_ok and device is not None:
        device.display(img)
        time.sleep(4.0)

# --- WORKERS ---
def xiao_worker():
    if not XIAO_OK or ser is None:
        print("[XIAO WORKER] Xiao missing. Worker thread inactive.")
        return
    while gs.running:
        header = ser.read(1)
        if header == b'\xaa':
            raw = ser.read(24)
            if len(raw) == 24:
                s = struct.unpack('ffffff', raw)
                gs.shared_sensors["a"], gs.shared_sensors["g"] = s[0:3], s[3:6]
        elif header == b'\xbb':
            raw = ser.read(512)
            if len(raw) == 512:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                samples *= GAIN_BOOST
                current_m = int(np.abs(samples).max())
                gs.shared_sensors["m"] = current_m
                
                # --- OFFLINE BACKGROUND WAKE-WORD DETECTION ---
                if gs.mode == "AVATAR" and gs.face_detected and VOSK_OK and bg_rec is not None:
                    bg_rec.AcceptWaveform(raw)
                    part_res = json.loads(bg_rec.PartialResult())
                    partial_text = part_res.get("partial", "").strip()
                    
                    # Target trigger words
                    wake_words = ["avatar", "wake", "hey", "listen", "hello", "robot", "buddy", "computer"]
                    if any(word in partial_text for word in wake_words):
                        print(f"[WAKE WORD] Detected trigger: '{partial_text}'! Starting AI...")
                        bg_rec.Reset()  # Clear buffer to prevent double triggers
                        threading.Thread(target=trigger_voice_ai, daemon=True).start()
                
                if gs.is_recording:
                    clipped = np.clip(samples, -32768, 32767).astype(np.int16)
                    gs.audio_buffer.append(clipped.tobytes())

def vision_worker():
    if not CAMERA_OK or picam2 is None or face_cascade is None or eye_cascade is None:
        print("[VISION WORKER] Camera or cascade models missing. Worker thread inactive.")
        return
    while gs.running:
        if gs.mode == "AVATAR":
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 2)
            
            looking_at_camera = False
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                    if len(eyes) >= 1:
                        looking_at_camera = True
                        break
            
            gs.face_detected = looking_at_camera
        time.sleep(0.3)

# --- UI HELPERS ---
def draw_bubble_frame(base_img, lines, y_offset=0):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([5, 5, 123, 46], radius=5, fill="white", outline="black", width=2)
    current_y = 10 - y_offset
    for line in lines:
        if 5 <= current_y <= 36:
            draw.text((13, current_y), line, fill="black", font=font)
        current_y += 12
    return img

def avatar_speak(text):
    print(f"[AI RESPONSE] {text}")
    gs.mode = "SPEAKING"

    # --- START LOCAL TTS (PIPER) ---
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = PIPER_LIB + ":" + my_env.get("LD_LIBRARY_PATH", "")
    
    try:
        # Scale output voice volume down by 50%
        audio_proc = subprocess.Popen([PIPER_BINARY, "--model", PIPER_MODEL, "--output_raw", "--volume", "0.5"], 
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=my_env)
        # Added "-D", AUDIO_DEVICE to play through configured device
        play_proc = subprocess.Popen(["aplay", "-D", AUDIO_DEVICE, "-r", "16000", "-f", "S16_LE", "-t", "raw"], 
                                      stdin=audio_proc.stdout)
        audio_proc.stdin.write(text.encode())
        audio_proc.stdin.close()
    except Exception as e:
        print(f"[ERROR] TTS Start Failed: {e}")

    # Determine background image based on whether a GameBoy portrait was generated
    if gs.portrait_img is not None:
        bg_idle = gs.portrait_img
        bg_talk_frame = gs.portrait_img  # Lock portrait on screen while AI speaks
    else:
        bg_idle = idle_frames[0]
        bg_talk_frame = None

    # --- TYPEWRITER & MOUTH ---
    wrapped = textwrap.wrap(text, width=18)
    displayed = []
    talk_idx = 0
    for line in wrapped:
        if len(displayed) >= 3:
            for off in range(0, 13, 4):
                frame_img = bg_idle
                if OLED_OK and device is not None:
                    with spi_lock: device.display(draw_bubble_frame(frame_img, displayed, y_offset=off))
                time.sleep(0.01)
            displayed.pop(0)
        displayed.append("")
        for char in line:
            displayed[-1] += char
            if len(displayed[-1]) % 2 == 0: 
                talk_idx = (talk_idx + 1) % len(talk_frames)
            
            # Resolve speaking background (GameBoy portrait OR normal speaking frames)
            frame_img = bg_talk_frame if bg_talk_frame is not None else talk_frames[talk_idx]
            
            if OLED_OK and device is not None:
                with spi_lock: device.display(draw_bubble_frame(frame_img, displayed))
            time.sleep(0.04)
        time.sleep(0.2)
    
    try: play_proc.wait()
    except: pass
    
    # Clear the portrait from memory so standard idle face takes over on the next standby
    gs.portrait_img = None
    time.sleep(1); gs.mode = "AVATAR"

# --- AI RESPONSE PARSER ---
def parse_ai_response(raw_text):
    clean_text = raw_text.strip()
    thought_end = clean_text.rfind("<channel|>")
    if thought_end != -1:
        clean_text = clean_text[thought_end + len("<channel|>"):].strip()
    think_end = clean_text.rfind("</think>")
    if think_end != -1:
        clean_text = clean_text[think_end + len("</think>"):].strip()

    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = clean_text[start_idx:end_idx+1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    return {
        "reply": clean_text.strip(),
        "emotion": "neutral"
    }

# --- LOCAL HARDWARE TOOLS (Exposed to Gemma 4 via Ollama with 0 Parameters) ---
def get_avatar_sensor_telemetry() -> str:
    """
    Retrieves the live, real-time physical hardware sensor telemetry from the avatar.
    This includes:
    1. Accelerometer vector [X, Y, Z] (which shows physical gravity/tilt/orientation).
    2. Gyroscope vector [X, Y, Z] (which shows active rotational speed/spinning).
    3. Microphone volume peak level.
    Returns:
        str: A JSON-formatted string of the active telemetry data.
    """
    return read_sensor_data(gs.shared_sensors)

def generate_pixel_portrait() -> str:
    """
    Converts the user's captured camera image into a dithered, retro 
    GameBoy-style pixel-art portrait, and stores it so it displays on the avatar's 
    OLED face while speaking. Call this if the user asks you to draw them, paint them, 
    or take their portrait.
    Returns:
        str: Success message.
    """
    from mcp_tools.portrait import create_gameboy_portrait
    
    # Fallback in case no image was captured
    img_to_process = gs.last_vision_image if gs.last_vision_image is not None else Image.new("RGB", (128, 128), "GRAY")
    
    # Generate the GameBoy portrait
    portrait = create_gameboy_portrait(img_to_process)
    
    # Save globally so avatar_speak() renders it
    gs.portrait_img = portrait
    
    return "GameBoy retro portrait generated successfully."

# --- AI LOGIC ---
def trigger_voice_ai():
    if gs.mode != "AVATAR": return
    gs.mode = "LISTENING"
    gs.audio_buffer = []
    gs.is_recording = True
    
    # 1. RECORDING PHASE
    if XIAO_OK:
        time.sleep(RECORD_TIME)
        gs.is_recording = False
        gs.mode = "THINKING"
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            wf.writeframes(b"".join(gs.audio_buffer))
        wav_bytes = wav_io.getvalue()
    else:
        print("[AUDIO] Xiao missing. Recording from default system microphone...")
        record_file = os.path.join(SCRIPT_DIR, "system_mic.wav")
        try:
            subprocess.run(["arecord", "-D", "default", "-r", "16000", "-f", "S16_LE", "-c", "1", "-d", str(int(RECORD_TIME)), record_file], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            gs.is_recording = False
            gs.mode = "THINKING"
            with open(record_file, 'rb') as f:
                wav_bytes = f.read()
        except Exception as rec_err:
            print(f"[AUDIO ERROR] arecord fallback failed: {rec_err}")
            gs.is_recording = False
            gs.mode = "THINKING"
            wav_bytes = b""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reply": types.Schema(type=types.Type.STRING),
                "emotion": types.Schema(type=types.Type.STRING, enum=["happy", "sad", "angry", "surprised", "neutral"]),
            },
            required=["reply", "emotion"],
        ),
    )

    try:
        if SOURCE_MODEL == "OLLAMA" and VOSK_OK:
            # --- 1. LOCAL SPEECH-TO-TEXT (VOSK) ---
            print("[STT] Transcribing voice recording locally with Vosk...")
            user_prompt = ""
            if stt_model and len(wav_bytes) > 44:
                try:
                    wf = wave.open(io.BytesIO(wav_bytes), "rb")
                    rec = KaldiRecognizer(stt_model, wf.getframerate())
                    rec.SetWords(False)
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0: break
                        rec.AcceptWaveform(data)
                    res = json.loads(rec.FinalResult())
                    user_prompt = res.get("text", "").strip()
                except Exception as stt_err:
                    print(f"[STT ERROR] Vosk failed: {stt_err}")
            
            print(f"[STT RESULT] User said: '{user_prompt}'")
            if not user_prompt:
                user_prompt = "The user is present but silent."

            # --- 2. EVALUATE VISION NEED ---
            # Standardized visual triggers (avoids triggering on general queries like 'what is the weather')
            prompt_lower = user_prompt.lower()
            visual_keywords = [
                "this", "that", "holding", "hand", "camera", "face", "look", "see", 
                "show", "draw", "paint", "portrait", "portrayed", "retro", "picture", 
                "color", "shirt", "wear", "identify", "object", "item", "thing"
            ]
            use_vision = any(kw in prompt_lower for kw in visual_keywords)
            
            # Non-visual overrides to bypass camera on common text/telemetry tools
            non_visual_overrides = ["weather", "forecast", "temperature", "temp in", "wikipedia", "search for"]
            if any(override in prompt_lower for override in non_visual_overrides):
                use_vision = False
            
            img_bytes = None
            if use_vision and CAMERA_OK and picam2 is not None:
                print("[VISION] Trigger word detected. Opening viewfinder...")
                gs.mode = "SNAP"
                time.sleep(2.0)
                
                if OLED_OK and device is not None:
                    with spi_lock: device.display(Image.new("RGB", (128, 128), "WHITE"))
                time.sleep(0.15)
                
                with spi_lock:
                    hd_frame = picam2.capture_array()
                
                gs.mode = "THINKING"
                rgb_frame = cv2.cvtColor(hd_frame, cv2.COLOR_BGR2RGB)
                vision_img = Image.fromarray(rgb_frame)
                vision_img.thumbnail((1024, 768)) 
                
                # Save the captured image globally so tools can access it
                gs.last_vision_image = vision_img
                
                img_byte_arr = io.BytesIO()
                vision_img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
            else:
                print("[VISION] Chat mode. Skipping camera capture...")
                gs.mode = "THINKING"

            # --- 3. OLLAMA LOCAL AI EXECUTION ---
            system_instruction = (
                "You are a retro pixel-art AI companion. You have access to an attached camera image of the user, "
                "and can also call your tools on demand.\n\n"
                "TOOLS AVAILABLE:\n"
                "1. 'get_avatar_sensor_telemetry' (0 parameters): Call this if the user asks about your physical state, tilt, movement, position, or room noise levels.\n"
                "2. 'generate_pixel_portrait' (0 parameters): Call this if the user asks you to draw them, paint them, take their portrait, or render their retro image.\n"
                "3. 'get_current_weather' (parameter: location): Call this if the user asks about the current weather of any city or location.\n"
                "4. 'search_wikipedia' (parameter: query): Call this if the user asks about facts, historical events, people, general knowledge, or topics requiring research.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. You MUST execute the matching tool first before generating your final text response whenever the user's prompt requires it.\n"
                "2. Keep your final response short, friendly, and in ONE sentence.\n"
                "3. Match the tone of the user's prompt.\n"
                "4. Your output MUST be a JSON object matching this exact structure:\n"
                "{\n"
                "  \"reply\": \"your friendly, short response here\",\n"
                "  \"emotion\": \"happy\" (or sad, angry, surprised, neutral)\n"
                "}"
            )
            
            message_content = {
                'role': 'user', 
                'content': f"The user asked: '{user_prompt}'. Reply to their question."
            }
            if img_bytes:
                message_content['images'] = [img_bytes]

            # Construct tools list dynamically based on hardware connectivity
            tools_list = [get_avatar_sensor_telemetry, generate_pixel_portrait, get_current_weather, search_wikipedia] if XIAO_OK else [generate_pixel_portrait, get_current_weather, search_wikipedia]
            
            response = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': system_instruction},
                    message_content
                ],
                tools=tools_list,  # Expose live sensor function directly
            )
            
            if response.message.tool_calls:
                available_functions = {
                    'get_avatar_sensor_telemetry': get_avatar_sensor_telemetry,
                    'generate_pixel_portrait': generate_pixel_portrait,
                    'get_current_weather': get_current_weather,
                    'search_wikipedia': search_wikipedia
                }
                for tool in response.message.tool_calls:
                    function_to_call = available_functions.get(tool.function.name)
                    if function_to_call:
                        # Dynamically extract arguments passed by Gemma 4
                        tool_args = tool.function.arguments if tool.function.arguments else {}
                        print(f"[TOOL EXECUTION] Model invoked {tool.function.name} with args {tool_args}")
                        
                        # Execute the tool with the mapped parameters
                        tool_output = function_to_call(**tool_args)
                        print(f"[TOOL OUTPUT] Result: {tool_output}")
                        
                        final_response = ollama_client.chat(
                            model=OLLAMA_MODEL,
                            messages=[
                                {'role': 'system', 'content': system_instruction},
                                message_content,
                                response.message,
                                {
                                    'role': 'tool',
                                    'content': tool_output,
                                    'name': tool.function.name
                                }
                            ],
                            format='json'
                        )
                        res_data = parse_ai_response(final_response.message.content)
            else:
                res_data = parse_ai_response(response.message.content)
            
        else:
            # --- GOOGLE GEMINI CLOUD EXECUTION ---
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=[
                    "SYSTEM: You are a retro pixel-art AI. Use the provided image and audio to talk to the user. "
                    "Respond in ONE short sentence. Pick an emotion.",
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav") if len(wav_bytes) > 44 else "User is present.",
                    Image.fromarray(picam2.capture_array()) if (CAMERA_OK and picam2 is not None) else "Camera offline."
                ],
                config=config
            )
            res_data = json.loads(response.text)

        avatar_speak(res_data['reply'])
    except Exception as e:
        print(f"[AI ERROR] {e}")
        avatar_speak("My brain is foggy today!")

# --- MAIN LOOP ---
def main_loop():
    threading.Thread(target=xiao_worker, daemon=True).start()
    threading.Thread(target=vision_worker, daemon=True).start()
    idx = 0
    last_log = 0
    consecutive_look_frames = 0  # Fallback face tracking counter
    
    # Run Boot Diagnostics if OLED is connected
    if OLED_OK and device is not None:
        show_system_check(OLED_OK, CAMERA_OK, XIAO_OK, VOSK_OK)

    while gs.running:
        now = time.time()
        if now - last_log > 0.5:
            print(f"[LOG] Mode: {gs.mode} | Face: {'YES' if gs.face_detected else 'NO'} | Mic: {gs.shared_sensors['m']}")
            last_log = now

        if OLED_OK and device is not None:
            with spi_lock:
                if gs.mode == "AVATAR":
                    img = idle_frames[idx % len(idle_frames)].copy()
                    draw = ImageDraw.Draw(img)
                    
                    if gs.face_detected:
                        draw.ellipse([110, 10, 120, 20], fill="red")
                        # Fallback trigger: if Xiao is missing, look at camera for 1.5 seconds to wake up!
                        if not XIAO_OK:
                            consecutive_look_frames += 1
                            if consecutive_look_frames >= 3:
                                consecutive_look_frames = 0
                                threading.Thread(target=trigger_voice_ai, daemon=True).start()
                    else:
                        consecutive_look_frames = 0
                        
                    device.display(img)
                    idx += 1
                    time.sleep(0.5)
                elif gs.mode == "LISTENING" and CAMERA_OK and picam2 is not None:
                    img = Image.fromarray(picam2.capture_array()).resize((128, 128))
                    draw = ImageDraw.Draw(img)
                    
                    if gs.is_recording:
                        draw.rectangle([0,0,127,127], outline="red", width=2)
                        if XIAO_OK:
                            v_bar = min(gs.shared_sensors["m"] // 200, 110)
                            draw.rectangle([10, 110, 10+v_bar, 118], fill="lime")
                        else:
                            draw.text((25, 105), "RECORDING...", fill="lime", font=font)
                    else:
                        draw.rectangle([0,0,127,127], outline="red", width=2)
                        draw.text((20, 105), "PROCESSING...", fill="red", font=font)
                    
                    device.display(img)
                elif gs.mode == "SNAP" and CAMERA_OK and picam2 is not None:
                    img = Image.fromarray(picam2.capture_array()).resize((128, 128))
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([0, 0, 127, 127], outline="yellow", width=4)
                    draw.text((30, 105), "HOLD STILL!", fill="yellow", font=font)
                    device.display(img)
                elif gs.mode == "THINKING":
                    # Cycle thinking dots: ".", "..", "..."
                    dots_cycle = [".", "..", "..."]
                    dots = dots_cycle[(idx // 3) % len(dots_cycle)]
                    base_img = idle_frames[idx % len(idle_frames)].copy()
                    img = draw_bubble_frame(base_img, ["Thinking" + dots])
                    device.display(img)
                    idx += 1
                    time.sleep(0.1)
        else:
            # OLED missing: run silently in the background
            time.sleep(0.1)

    if device is not None: device.clear()
    if picam2 is not None: picam2.stop()

if __name__ == "__main__":
    main_loop()
