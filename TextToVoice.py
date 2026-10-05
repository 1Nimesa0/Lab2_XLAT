import asyncio
import edge_tts
import os

VOICE = "vi-VN-NamMinhNeural"


async def generate_audio(text, output_file):
    communicate = edge_tts.Communicate(
        text,
        VOICE,
        rate="-5%",
        volume="+0%",
        pitch="-2Hz"
    )
    # Lưu file âm thanh vào thư mục audio
    await communicate.save(output_file)
    print(f"Đã tạo file: {output_file}")


async def main():
    file_number = 1

    while True:
        text = input("Nhập văn bản cần chuyển đổi: ").strip()
        if not text:
            print("Văn bản không được để trống.")
            continue

        output_file = f"{file_number}.mp3"
        while os.path.exists(output_file):
            file_number += 1
            output_file = f"{file_number}.mp3"

        await generate_audio(text, output_file)
        print("File được lưu tại:", os.path.abspath(output_file))
        file_number += 1

        stop = input("Nhập 'dừng' để kết thúc, hoặc nhấn Enter để tiếp tục: ")
        if stop.strip().lower() == "dừng":
            break


if __name__ == "__main__":
    asyncio.run(main())