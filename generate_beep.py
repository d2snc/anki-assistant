import wave, struct, base64, io

buf = io.BytesIO()
with wave.open(buf, "w") as w:
    w.setnchannels(1)
    w.setsampwidth(1) # 8-bit
    w.setframerate(8000)
    # 0.15 seconds beep at 800Hz
    frames = bytearray()
    import math
    for i in range(int(8000 * 0.15)):
        value = int(127.0 * math.sin(2.0 * math.pi * 880.0 * i / 8000.0) * (1 - i/(8000*0.15))) + 128
        frames.append(value)
    w.writeframes(bytes(frames))

print("data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode())
