"""Original slideshow-style faceless YouTube videos from Gemini + TTS.

Uses Gemini for storyboards and original on-screen text, edge-tts for speech,
Pillow for custom scene backgrounds, and FFmpeg for the finished MP4.
No unattended publication and no third-party stock footage.
"""
import asyncio
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

from google import genai
from PIL import Image, ImageDraw, ImageFont, ImageOps
from io import BytesIO
import edge_tts

OUT = Path('output')
WORK = Path('work')
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)
TOPIC = os.getenv('TOPIC', 'How AI agents could change small businesses').strip()
TARGET = max(3, min(12, int(os.getenv('TARGET_MINUTES', '8'))))
API_KEY = os.getenv('GEMINI_API_KEY', '')
MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
if not API_KEY:
    raise SystemExit('Missing GEMINI_API_KEY. Add it in Repository Settings > Secrets and variables > Actions.')


def call_gemini():
    client = genai.Client(api_key=API_KEY)
    # 125-145 spoken words/minute; do not guarantee exact finished duration.
    total = TARGET * 130
    prompt = f'''Create an original, engaging faceless English YouTube video on: {TOPIC}
Audience: curious adults in US and worldwide. Target voice-over length: {total} words.
Generate 9 to 14 distinct scenes, ~{total} spoken words total. No plagiarism,
no unverified statistics or invented citations. For claims needing external
fact checking, use cautious language and include a note in fact_checks.
Provide valid JSON ONLY (not fenced) with exactly these top-level keys:
"title" (under 65 chars), "description" (2 short paragraphs with hashtags),
"tags" (array of 10 strings), "thumbnail_text" (under 5 words),
"fact_checks" (array of specific statements for user to verify, may be empty),
"scenes" (array of objects with "heading" (under 6 words),
"on_screen" (under 18 words), "narration" (spoken English, no bracket directions)).
Make narration coherent, compelling, factual or clearly hypothetical. Do not
pretend you conducted live research. Avoid generic repetitive filler.
'''
    response = client.models.generate_content(model=MODEL, contents=prompt)
    raw = (response.text or '').strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw).strip()
    data = json.loads(raw)
    if not isinstance(data.get('scenes'), list) or len(data['scenes']) < 3:
        raise ValueError('Gemini returned insufficient scenes')
    return data


def font(size, bold=False):
    p = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(p, size)


def wrap(draw, string, use_font, max_width, max_lines):
    words, lines, current = str(string).split(), [], ''
    for w in words:
        test = f'{current} {w}'.strip()
        if draw.textbbox((0, 0), test, font=use_font)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines[:max_lines]


def scene_image(scene, idx, n, dest):
    # Simple original graphics; NOT AI-generated illustrations or moving characters.
    W, H = 1280, 720
    img = Image.new('RGB', (W, H))
    px = img.load()
    palette = [(10, 22, 47), (13, 31, 56), (20, 22, 55), (11, 39, 48)]
    c = palette[idx % len(palette)]
    for y in range(H):
        d = int(y * 18 / H)
        for x in range(W):
            px[x, y] = (min(255,c[0]+d), min(255,c[1]+d), min(255,c[2]+d))
    draw = ImageDraw.Draw(img)
    accent = [(68, 231, 198), (116, 166, 255), (255, 180, 111)][idx % 3]
    for k in range(12):
        x = 830 + (k * 83 + idx * 53) % 460
        y = 80 + (k * 97) % 560
        r = 22 + (k * 17) % 75
        draw.ellipse((x-r,y-r,x+r,y+r), outline=accent, width=2)
    draw.rounded_rectangle((74, 90, 750, 104), 7, fill=accent)
    draw.text((78, 136), f'SCENE {idx+1:02d} / {n:02d}', font=font(29), fill=accent)
    heading_lines = wrap(draw, scene.get('heading',''), font(67,True), 1040, 3)
    y = 210
    for line in heading_lines:
        draw.text((76, y), line, font=font(67,True), fill='white')
        y += 89
    y += 20
    for line in wrap(draw, scene.get('on_screen',''), font(36), 1030, 4):
        draw.text((80,y), line, font=font(36), fill=(215,225,242))
        y += 53
    draw.text((78, 665), 'AI EXPLAINED  |  ORIGINAL STORY', font=font(23), fill=accent)
    img.save(dest)


def ai_image(scene, idx, dest):
    """Optional original AI image. Falls back to local graphics on failure."""
    if os.getenv('AI_IMAGES', 'false').lower() != 'true':
        return False
    try:
        client = genai.Client(api_key=API_KEY)
        prompt = (f"Create a cinematic original 16:9 illustration for an educational YouTube video about {TOPIC}. "
                  f"Scene: {scene.get('heading','')}. {scene.get('on_screen','')}. "
                  "No text, logos, watermark, brand marks or recognizable real people. "
                  "Distinct composition, visually informative, clean 3D editorial illustration.")
        r = client.models.generate_content(
            model=os.getenv('IMAGE_MODEL', 'gemini-2.5-flash-image'),
            contents=prompt,
            config={'response_modalities': ['IMAGE', 'TEXT']})
        for candidate in (r.candidates or []):
            for part in (candidate.content.parts or []):
                if getattr(part, 'inline_data', None) and part.inline_data.data:
                    im = Image.open(BytesIO(part.inline_data.data)).convert('RGB')
                    ImageOps.fit(im, (1280,720), method=Image.Resampling.LANCZOS).save(dest, quality=90)
                    return True
    except Exception as exc:
        print(f'AI image for scene {idx+1} unavailable: {exc}. Using original graphic fallback.')
    return False


def thumbnail(data):
    img = Image.new('RGB',(1280,720),(12,22,46))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((60,60,1220,660),radius=45,outline=(80,230,200),width=9)
    d.ellipse((870,125,1170,425),fill=(42,83,150),outline=(92,220,230),width=12)
    d.text((963,185),'AI',font=font(85,True),fill='white')
    lines=wrap(d,data.get('thumbnail_text') or data.get('title','AI EXPLAINED'),font(92,True),750,4)
    y=180
    for line in lines:
        d.text((95,y),line,font=font(92,True),fill='white',stroke_width=2,stroke_fill=(0,0,0))
        y+=115
    img.save(OUT/'thumbnail.jpg',quality=92)


def srt_timestamp(seconds):
    ms=int(round(seconds*1000)); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); sec,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{sec:02},{ms:03}'


async def synthesize(scenes):
    for i, scene in enumerate(scenes):
        narrative = str(scene['narration']).strip()
        if not narrative:
            raise ValueError(f'Empty narration for scene {i+1}')
        await edge_tts.Communicate(narrative, voice='en-US-AriaNeural').save(str(WORK/f'voice_{i:02d}.mp3'))


def command(args):
    subprocess.run(args, check=True)


def duration(path):
    p = subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)], text=True)
    return float(p.strip())


def render(scenes):
    pieces = []
    subtitles=[]
    elapsed=0.0
    for i, scene in enumerate(scenes):
        image = WORK / f'slide_{i:02d}.jpg'
        audio = WORK / f'voice_{i:02d}.mp3'
        clip = WORK / f'clip_{i:02d}.mp4'
        if not ai_image(scene, i, image):
            scene_image(scene, i, len(scenes), image)
        length = duration(audio) + 0.3
        # Scene-timed subtitles: one cue per scene; not word-synced.
        subtitle = re.sub(r'\s+', ' ', str(scene['narration'])).strip()
        subtitles.append(f'{i+1}\n{srt_timestamp(elapsed)} --> {srt_timestamp(elapsed+length)}\n{subtitle}\n')
        elapsed += length
        command(['ffmpeg','-y','-loglevel','error','-loop','1','-framerate','25','-i',str(image),
                 '-i',str(audio),'-t',f'{length:.3f}','-c:v','libx264','-preset','veryfast',
                 '-pix_fmt','yuv420p','-r','25','-c:a','aac','-b:a','128k',
                 '-vf','scale=1280:720','-movflags','+faststart',str(clip)])
        pieces.append(clip.resolve())
    (OUT/'subtitles.srt').write_text('\n'.join(subtitles),encoding='utf-8')
    manifest = WORK/'concat.txt'
    manifest.write_text(''.join("file '"+str(p).replace("'", "'\\''")+"'\n" for p in pieces))
    command(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(manifest),
             '-c','copy','-movflags','+faststart',str(OUT/'video.mp4')])


def main():
    print(f'Creating original YouTube video for: {TOPIC!r}')
    data = call_gemini()
    (OUT/'video_metadata.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'upload_description.txt').write_text(f"{data.get('title','')}\n\n{data.get('description','')}\n\nTags: {', '.join(data.get('tags',[]))}\n\nThumbnail text: {data.get('thumbnail_text','')}",encoding='utf-8')
    checks = data.get('fact_checks', [])
    (OUT/'REVIEW_BEFORE_PUBLISHING.txt').write_text(
        'Human review required. Check all facts, original value, pronunciation, audio, and visuals.\n'
        'This video is not guaranteed to qualify for YouTube monetization.\n'
        'Statements to verify:\n' + '\n'.join('- '+str(x) for x in checks),encoding='utf-8')
    scenes = data['scenes']
    thumbnail(data)
    asyncio.run(synthesize(scenes))
    render(scenes)
    print('DONE: output/video.mp4')


if __name__ == '__main__':
    main()
