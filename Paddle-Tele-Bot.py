import asyncio
import json
import logging
import sys
from io import BytesIO
import cv2
import jwt

import openai
import pytesseract
import requests
import os

PADDLE_OCR_PATH = "./PaddleOCR"
if PADDLE_OCR_PATH not in sys.path:
    sys.path.insert(0, PADDLE_OCR_PATH)

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message
from telegram.ext import CallbackContext

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pydub import AudioSegment
from telegram import Update

from aiogram.types import InputFile
from aiogram.types import FSInputFile

from typing import List
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from translate import Translator

from ocr_predictor import PaddleOCRRecognizer
from dotenv import load_dotenv
load_dotenv()


# Provide absolute or relative paths
config_path = "PaddleOCR/configs/rec/multi_language/rec_crnn_armenian.yml"
model_path = "./model/model/best_accuracy"
image_path = "img_00265.png"

# Initialize the recognizer
ocr = PaddleOCRRecognizer(config_path, model_path)

# Run OCR on the image
result = ocr.predict(image_path)

print(f"Recognized text: {result}")

app = logging.getLogger(__name__)

# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

print(f"Using Google credentials from: {key_path}")

credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

translate_client = translate.Client(credentials=credentials)


def translate_object_list(objects: List[str], target_language='hy') -> List[str]:
    if not objects or objects[0] == "No objects detected":
        return objects

    try:
        results = translate_client.translate(objects, target_language=target_language)
        return [result['translatedText'] for result in results]
    except Exception as e:
        logger.error(f"Object translation error: {e}")
        return objects


project_url = "https://wav.am/generate_audio/"
access_token = os.getenv("WAV_AM_ACCESS_TOKEN")

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

OD_path = "./detector.tflite"

base_options = python.BaseOptions(model_asset_path=OD_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.ObjectDetector.create_from_options(options)


def detect_objects(image_path: str) -> list:
    try:
        image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(image)

        OBJECT_TRANSLATIONS = {
            "person": "մարդ",
            "bicycle": "հեծանիվ",
            "car": "մեքենա",
            "motorcycle": "մոտոցիկլետ",
            "airplane": "ինքնաթիռ",
            "bus": "ավտոբուս",
            "train": "գնացք",
            "truck": "բեռնատար",
            "boat": "նավակ",
            "traffic light": "լուսացույց",
            "fire hydrant": "հրշեջ հիդրանտ",
            "stop sign": "կանգառի նշան",
            "parking meter": "ավտոկայանատեղի հաշվիչ",
            "bench": "նստարան",
            "bird": "թռչուն",
            "cat": "կատու",
            "dog": "շուն",
            "horse": "ձի",
            "sheep": "ոչխար",
            "cow": "կով",
            "elephant": "փիղ",
            "bear": "արջ",
            "zebra": "զեբր",
            "giraffe": "ընձուղտ",
            "backpack": "ուղեպարկ",
            "umbrella": "անձրևանոց",
            "handbag": "ձեռքի պայուսակ",
            "tie": "փողկապ",
            "suitcase": "ճամպրուկ",
            "frisbee": "ֆրիսբի",
            "skis": "դահուկներ",
            "snowboard": "սնոուբորդ",
            "sports ball": "սպորտային գնդակ",
            "kite": "օդապարուկ",
            "baseball bat": "բեյսբոլի մահակ",
            "baseball glove": "բեյսբոլի ձեռնոց",
            "skateboard": "սքեյթբորդ",
            "surfboard": "սերֆի տախտակ",
            "tennis racket": "թենիսի ռակետ",
            "bottle": "շիշ",
            "wine glass": "գինու բաժակ",
            "cup": "բաժակ",
            "fork": "պտտահատ",
            "knife": "դանակ",
            "spoon": "գդալ",
            "bowl": "ափսե",
            "banana": "բանան",
            "apple": "խնձոր",
            "sandwich": "սենդվիչ",
            "orange": "նարինջ",
            "broccoli": "բրոկոլի",
            "carrot": "գազար",
            "hot dog": "հոտ դոգ",
            "pizza": "պիցցա",
            "donut": "դոնատ",
            "cake": "տորթ",
            "chair": "աթոռ",
            "couch": "բազմոց",
            "potted plant": "թաղարով բույս",
            "bed": "մահճակալ",
            "dining table": "ճաշասեղան",
            "toilet": "զուգարան",
            "tv": "հեռուստացույց",
            "laptop": "նոութբուք",
            "mouse": "մկնիկ",
            "remote": "հեռակառավարման վահանակ",
            "keyboard": "ստեղնաշար",
            "cell phone": "բջջային հեռախոս",
            "microwave": "միկրոալիքային վառարան",
            "oven": "ջեռոց",
            "toaster": "տոստեր",
            "sink": "լվացարան",
            "refrigerator": "սառնարան",
            "book": "գիրք",
            "clock": "ժամացույց",
            "vase": "վազա",
            "scissors": "մկրատ",
            "teddy bear": "փափուկ արջուկ",
            "hair drier": "մազերի չորանոց",
            "toothbrush": "ատամի խոզանակ"
        }

        detected_objects = []
        for detection in detection_result.detections:
            category = detection.categories[0]
            if category.score > 0.4:
                # Getting English object name
                obj_name = category.category_name

                # Translating to Armenian
                armenian_name = OBJECT_TRANSLATIONS.get(
                    obj_name.lower(),
                    obj_name
                )

                detected_objects.append(
                    f"{armenian_name} ({category.score:.0%})"
                )

        return detected_objects if detected_objects else ["No objects detected"]

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return ["Detection error"]


def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Հնարավոր չեղավ բեռնել նկարը։"

        print(image_path)
        # extracted_text = pytesseract.image_to_string(image, config='--psm 6', lang='hye+eng+rus')
        extracted_text = ocr.predict(image_path)
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return f"Error extracting text: {str(e)}"


payload = jwt.decode(access_token, options={"verify_signature": False})
print(payload)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories for storing downloaded images and audio
IMAGE_DIR = "downloaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
AUDIO_DIR = "downloaded_audios"
os.makedirs(AUDIO_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Telegram bot setup
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

bot = Bot(token=API_TOKEN)

openai.api_key = os.getenv("OPENAI_API_KEY")

dp = Dispatcher()
router = Router()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_mp3_from_url(url, token, filename=None):
    os.makedirs("downloaded_audios", exist_ok=True)

    headers = {
        "Authorization": token,
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename.endswith(".mp3"):
                filename += ".mp3"

        save_path = os.path.join("downloaded_audios", filename)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Successfully saved MP3 to: {save_path}")
        return save_path

    except Exception as e:
        print(f"❌ Failed to download MP3: {e}")
        return None


@router.message(Command("start"))
async def command_start_handler(message: Message):
    await message.answer("Ողջույն, այսուհետ ես կլինեմ Ձեր աչքերը։")


async def handle_audio_sending(message: Message, mp3_path: str):
    try:
        try:
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            ogg_bytes = BytesIO()
            audio.export(ogg_bytes, format="ogg")
            ogg_bytes.seek(0)

            await message.answer_voice(
                voice=InputFile(ogg_bytes, filename="voice_message.ogg")
            )
            print("Voice message sent successfully")
            return

        except Exception as convert_error:
            logger.warning(f"OGG conversion failed, falling back to MP3: {convert_error}")

            with open(mp3_path, 'rb') as audio_file:
                await message.answer_audio(
                    audio=InputFile(audio_file, filename="audio_message.mp3")
                )
            print("MP3 audio sent successfully")

    except Exception as e:
        logger.error(f"Audio sending failed: {e}")
        await message.answer("Failed to send audio")

    finally:
        if 'ogg_bytes' in locals():
            ogg_bytes.close()
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


@router.message(F.photo)
async def generate_audio(message: Message):
    file_path = None
    mp3_path = None
    try:
        photo = message.photo[-1]
        bot = message.bot

        file = await bot.get_file(photo.file_id)
        file_path = os.path.join(IMAGE_DIR, f"{photo.file_id}.jpg")
        await bot.download_file(file.file_path, destination=file_path)

        # Extracting text using PyTesseract
        extracted_text = extract_text_from_image(file_path)

        if not extracted_text.strip():
            await message.answer("Տեքստ չի գտնվել, հավանաբար նկարում օբյեկտներ կան։")
            if os.path.exists(file_path):
                os.remove(file_path)
            return

        english_objects = detect_objects(file_path)

        armenian_objects = translate_object_list(english_objects)

        object_message = ""
        text_message = ""

        if armenian_objects[0] != "No objects detected":
            object_message = "🔍 Տեքստ չի գտնվել, հավանաբար նկարում օբյեկտներ կան:\n" + "\n".join(armenian_objects)

        if extracted_text.strip():
            text_message = f"\n Նկարում կա տեքստ, ահա, խնդրեմ տեքստը՝ \n{extracted_text}"

        if object_message:
            full_message = "🔍 Տեքստ չի գտնվել, հավանաբար նկարում օբյեկտներ կան, դրանք են՝ \n" +"\n".join(armenian_objects)
        elif text_message:
            full_message = text_message
        else:
            full_message = "Ոչինչ չի գտնվել"

        api_url = "https://wav.am/generate_audio/"
        headers = {
            "Authorization": access_token,
            "Content-Type": "application/json",
        }
        data = {
            "project_id": "239",
            "text": full_message,
            "voice": "Ani",
            "format": "mp3"
        }

        tts_response = requests.post(api_url, headers=headers, json=data)
        print("TTS response status:", tts_response.status_code)
        print("TTS response text:", full_message[:50] + "..." if len(full_message) > 50 else full_message)

        if tts_response.status_code != 200:
            await message.answer("Ձայնային ֆայլը ստեղծել չհաջողվեց։")
            return

        try:
            response_data = tts_response.json()
            print("Full API response:", response_data)

            url = "https://wav.am" + response_data["path"]

            print("Constructed audio URL:", url)
        except json.JSONDecodeError:
            await message.answer("Սխալ API պատասխան։")
            print("Invalid JSON response:", tts_response.text)
            if os.path.exists(file_path):
                os.remove(file_path)
            return

        audio_response = requests.get(url)
        print("Audio download status:", audio_response.status_code)

        mp3_path = download_mp3_from_url(url, access_token, f"audio_{message.message_id}.mp3")
        if not os.path.exists(mp3_path):
            await message.answer("Չհաջողվեց ներբեռնել ձայնային ֆայլը։")
            print("MP3 file does not exist at path:", mp3_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            return
        try:
            voice_file = FSInputFile(mp3_path)

            try:
                await message.answer_voice(voice=voice_file)
                print("Audio successfully sent as voice message")
            except Exception as voice_error:
                # If sending as voice fails, trying to send as audio
                print(f"Failed to send as voice: {str(voice_error)}")
                voice_file.seek(0)
                await message.answer_audio(audio=voice_file)
                print("Audio successfully sent as audio message")
        except Exception as e:
            logger.error(f"Failed to send audio: {e}")
            await message.answer("Ձայնային ֆայլը ուղարկել չհաջողվեց։")
            print("Audio sending error:", str(e))
        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

            if os.path.exists(file_path):
                os.remove(file_path)
    except Exception as send_error:
        logger.error(f"Audio sending failed: {send_error}")
        await message.answer("Չհաջողվեց ուղարկել ձայնային ֆայլը։")


@router.message(F.photo)
async def handle_photo_message(message: Message):
    await generate_audio(message)


async def main() -> None:
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())

