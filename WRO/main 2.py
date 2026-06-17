import cv2
import numpy as np
import mediapipe as mp
import time
import os
import serial
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL   = os.path.join(BASE_DIR, 'face_landmarker.task')
SAVE_PATH    = os.path.join(BASE_DIR, 'result.png')
CAPTURE_ZONE = 0.10


SERIAL_PORT = 'COM20'
SERIAL_BAUD = 9600


JOY_LEFT       = 300   
JOY_RIGHT      = 800   
JOY_UP         = 300   
JOY_DOWN       = 800   
JOY_CENTER_MIN = 300
JOY_CENTER_MAX = 800

TEMPLATES = [
    {'file': os.path.join(BASE_DIR, 'icon1.png'), 'name': 'Cloth 1'},
    {'file': os.path.join(BASE_DIR, 'icon2.png'), 'name': 'Cloth 2'},
    {'file': os.path.join(BASE_DIR, 'icon3.png'), 'name': 'Cloth 3'},
    {'file': os.path.join(BASE_DIR, 'icon4.png'), 'name': 'Cloth 4'},
    {'file': os.path.join(BASE_DIR, 'icon5.png'), 'name': 'Cloth 5'},
    {'file': os.path.join(BASE_DIR, 'icon6.png'), 'name': 'Cloth 6'},
    {'file': os.path.join(BASE_DIR, 'icon7.png'), 'name': 'Cloth 7'},
    {'file': os.path.join(BASE_DIR, 'icon8.png'), 'name': 'Cloth 8'},
]


ORNAMENT_DESCRIPTIONS = [
    "Cloth 1\n\n Taban: endurance, longevity and a safe journey \n for travelers \n Kazmoyin: flexibility, grace and the ability to \n overcome difficulties",
    "Cloth 2\n\n Taban: endurance, longevity and a safe journey \n for travelers \n Muyiz: prosperity and abundance",
    "Cloth 3\n\n Qoshqar muyiz: symbol of strength and abundance \n Muyiz: prosperity and abundance",
    "Cloth 4\n\n Taban: endurance, longevity and a safe journey \n for travelers",
    "Cloth 5\n\n Taban: endurance, longevity and a safe journey \n for travelers",
    "Cloth 6\n\n Qoshqar muyiz: symbol of strength and \n abundance \n Muyiz: prosperity and abundance",
    "Cloth 7\n\n Muyiz: prosperity and abundance \n Taban: endurance, longevity and a safe journey for travelers",
    "Cloth 8\n\n Tortqulaq: stability and strength \n Taban: endurance, longevity and a safe journey \n for travelers",
]

FULL_HEAD_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

#joystic set up
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
    print(f'Джойстик подключён: {SERIAL_PORT}')
except Exception as e:
    print(f'ошибка подключения к Arduino ({SERIAL_PORT}): {e}')
    print('Продолжаем без джойстика — используй клавиатуру.')
    ser = None

def read_joystick():
    
    if ser is None:
        return 512, 512, False

    try:
        last_line = None
        while ser.in_waiting > 0:
            last_line = ser.readline().decode('utf-8', errors='ignore').strip()

        if last_line:
            parts = last_line.split(',')
            if len(parts) >= 3:
                x, y, button = int(parts[0]), int(parts[1]), int(parts[2])
                return x, y, (button == 0)
            elif len(parts) == 2:
                x, y = int(parts[0]), int(parts[1])
                return x, y, False
    except Exception:
        pass
    return 512, 512, False

class JoyEdge:
    
    def __init__(self):
        self.prev_left  = False
        self.prev_right = False
        self.prev_up    = False
        self.prev_down  = False
        self.prev_push  = False

    def update(self, jx, jy, sw_pressed=False):
        left  = jx < JOY_LEFT
        right = jx > JOY_RIGHT
        up    = jy < JOY_UP
        down  = jy > JOY_DOWN

        fired_left  = left       and not self.prev_left
        fired_right = right      and not self.prev_right
        fired_up    = up         and not self.prev_up
        fired_down  = down       and not self.prev_down
        fired_push  = sw_pressed and not self.prev_push

        self.prev_left  = left
        self.prev_right = right
        self.prev_up    = up
        self.prev_down  = down
        self.prev_push  = sw_pressed

        return fired_left, fired_right, fired_up, fired_down, fired_push

def wait_joystick_neutral(timeout_ms=2000):
    
    start = int(time.time() * 1000)
    while True:
        jx, jy, _ = read_joystick()
        in_center = (JOY_CENTER_MIN <= jx <= JOY_CENTER_MAX and
                     JOY_CENTER_MIN <= jy <= JOY_CENTER_MAX)
        if in_center:
            break
        if int(time.time() * 1000) - start > timeout_ms:
            break
        time.sleep(0.05)

#samples
def load_template(path):
    if not os.path.exists(path):
        print(f'  файл не найден: {path}')
        return None, None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'  Не удалось загрузить: {path}')
        return None, None

    bgr = img[:, :, :3] if (img.ndim == 3 and img.shape[2] == 4) else img

    # 1) Зелёный овал по HSV
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 60), (90, 255, 255))

    # 2) Прозрачность
    if mask.sum() == 0 and img.ndim == 3 and img.shape[2] == 4:
        _, mask = cv2.threshold(img[:, :, 3], 10, 255, cv2.THRESH_BINARY_INV)

    # 3) Тёмные пиксели
    if mask.sum() == 0:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print(f'  Вырез не найден: {path}')
        return None, None

    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    dst = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    print(f'  OK {os.path.basename(path)}: вырез x={x} y={y} w={w} h={h}')

    bgr_clean    = bgr.copy()
    border_color = bgr[max(0, y-4):y, x:x+w].mean(axis=(0, 1)) if y > 4 else np.array([128, 128, 128])
    cv2.ellipse(bgr_clean, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, border_color.tolist(), -1)
    return bgr_clean, dst

#mediapipe 
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
        num_faces=1))

def composite(frame, icon_bgr, dst_pts):
    
    h_f, w_f = frame.shape[:2]
    h_i, w_i = icon_bgr.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not res.face_landmarks:
        return None, None, False, None

    landmarks = res.face_landmarks[0]
    pts = np.array([[lm.x*w_f, lm.y*h_f] for lm in landmarks], dtype=np.float32)

    face_pts = np.array([pts[i] for i in FULL_HEAD_IDX], dtype=np.int32)
    hull     = cv2.convexHull(face_pts)
    fx, fy, fw, fh = cv2.boundingRect(hull)
    src_pts  = np.float32([[fx, fy], [fx+fw, fy], [fx+fw, fy+fh], [fx, fy+fh]])

    M      = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (w_i, h_i))

    cw   = cv2.perspectiveTransform(hull.astype(np.float32).reshape(-1, 1, 2), M)
    mask = np.zeros((h_i, w_i), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cw.astype(np.int32).reshape(-1, 2), 255)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    alpha  = cv2.merge([mask, mask, mask]).astype(float) / 255.0
    output = (warped * alpha + icon_bgr * (1 - alpha)).astype(np.uint8)

    nose    = landmarks[1]
    in_zone = abs(nose.x - 0.5) < CAPTURE_ZONE and abs(nose.y - 0.5) < CAPTURE_ZONE
    return output, pts, in_zone, nose

def composite_from_snapshot(snapshot, icon_bgr, dst_pts):
    return composite(snapshot, icon_bgr, dst_pts)

# panels
def draw_text(img, text, pos, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,  cv2.LINE_AA)

def draw_panel(img, lines, x=16, y=16):
    lh, pad = 18, 7  ##razmer menu sleva sverhu
    total_h = len(lines) * lh + pad * 2
    max_w   = max((cv2.getTextSize(l.lstrip('>'), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                   for l in lines), default=100) + pad * 2
    cv2.rectangle(img, (x, y), (x+max_w, y+total_h), (15, 15, 15), -1)
    cv2.rectangle(img, (x, y), (x+max_w, y+total_h), (70, 70, 70), 1)
    for i, line in enumerate(lines):
        col = (0, 230, 160) if line.startswith('>') else (220, 220, 220)
        draw_text(img, line.lstrip('>'), (x+pad, y+pad+(i+1)*lh-4), 0.4, col, 1)

def draw_joystick_indicator(img, jx, jy, cx, cy, r=30):
    cv2.circle(img, (cx, cy), r, (60, 60, 60), -1)
    cv2.circle(img, (cx, cy), r, (100, 100, 100), 1)
    dx = int((jx - 512) / 512 * (r - 8))
    dy = int((jy - 512) / 512 * (r - 8))
    cv2.circle(img, (cx+dx, cy+dy), 8, (0, 200, 255), -1)

def overlay_template_on_frame(frame, bgr, h, w):
    
    dark = frame.copy()
    cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, dark, 0.7, 0)
    
    if bgr is not None:
        th, tw = bgr.shape[:2]
        scale = min(int(h * 0.75) / th, int(w * 0.8) / tw) 
        
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        small = cv2.resize(bgr, (int(tw * scale), int(th * scale)), interpolation=interp)
        
        sh, sw_img = small.shape[:2]
        ox = (w - sw_img) // 2
        oy = (h - sh) // 2
        
        
        frame[oy:oy+sh, ox:ox+sw_img] = (
            frame[oy:oy+sh, ox:ox+sw_img].astype(float) * 0.1 +
            small.astype(float) * 0.9).astype(np.uint8)
            
    return frame

#ornament door
def show_ornament_info(description, template_bgr):
   
    wait_joystick_neutral(timeout_ms=1000)
    edge = JoyEdge()

    
    if template_bgr is not None:
        base = cv2.resize(template_bgr, (800, 600))
        base = cv2.GaussianBlur(base, (61, 61), 0)
        base = cv2.addWeighted(base, 0.25, np.zeros_like(base), 0.75, 0)
    else:
        base = np.zeros((600, 800, 3), dtype=np.uint8)

    h, w = base.shape[:2]

    # 
    card_x1, card_y1 = int(w * 0.1), int(h * 0.1)
    card_x2, card_y2 = int(w * 0.9), int(h * 0.85)
    display = base.copy()
    cv2.rectangle(display, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), -1)
    cv2.rectangle(display, (card_x1, card_y1), (card_x2, card_y2), (100, 200, 150), 2)

    
    lines = description.split('\n')
    line_h = 40
    text_y = card_y1 + 55
    for i, line in enumerate(lines):
        color = (0, 230, 160) if i == 0 else (220, 220, 220)
        scale = 0.85 if i == 0 else 0.65
        cv2.putText(display, line,
                    (card_x1 + 30, text_y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(display, line,
                    (card_x1 + 30, text_y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    
    hint = 'down/any key - back'
    cv2.putText(display, hint,
                (card_x1 + 30, card_y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(display, hint,
                (card_x1 + 30, card_y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    cv2.namedWindow('ornament info', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ornament info', 800, 600)
    cv2.imshow('ornament info', display)

    while True:
        jx, jy, sw = read_joystick()
        _, _, _, fired_down, _ = edge.update(jx, jy, sw)

        key = cv2.waitKey(30) & 0xFF

        if fired_down or key != 0xFF:
            cv2.destroyWindow('ornament info')
            wait_joystick_neutral(timeout_ms=800)
            return

# photo shoot
def phase_photo():
    print('съёмка лица')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print('Камера не найдена')
        return None

    wait_joystick_neutral(timeout_ms=3000)

    edge   = JoyEdge()
    output = None

    cv2.namedWindow('shooting', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('shooting', 1920, 1080)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        crop_w = int(w * 0.5) ##
        crop_h = int(h * 0.5) ##face zone
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        frame = frame[y1:y1+crop_h, x1:x1+crop_w]

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        UI = {
            "pad":        int(w * 0.02),
            "font_scale": w / 900,
            "thickness":  max(1, int(w / 500))
        }

        jx, jy, sw = read_joystick()
        fired_left, fired_right, fired_up, fired_down, fired_push = edge.update(jx, jy, sw)

        display = frame.copy()
        display = cv2.resize(display,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) #quality
        h, w = display.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        face_found = bool(res.face_landmarks)
        in_zone    = False
        if face_found:
            landmarks = res.face_landmarks[0]
            nose = landmarks[1]
            in_zone = abs(nose.x - 0.5) < CAPTURE_ZONE and abs(nose.y - 0.5) < CAPTURE_ZONE
            

        z  = 0.12
        ts = (int(w * (0.5 - z)), int(h * (0.5 - z)))
        te = (int(w * (0.5 + z)), int(h * (0.5 + z)))
        zone_color = (0, 255, 0) if (face_found and in_zone) else (200, 200, 200)
        cv2.rectangle(display, ts, te, zone_color, 2)

        sy = int((time.time() * 180) % h)
        cv2.line(display, (0, sy), (w, sy), (0, 200, 0), 1)

        key        = cv2.waitKey(1) & 0xFF
        take_photo = fired_down
        if key in (ord(' '), 13):
            take_photo = True
        if key in (ord('q'), 27):
            cap.release(); cv2.destroyAllWindows(); return None

        if take_photo:
            snapshot = frame.copy()
            cap.release(); cv2.destroyAllWindows()
            return snapshot

        status_text = 'face in the zone - shoot!' if (face_found and in_zone) else \
                      ('face is found'            if face_found else 'face is not found')
        status_col  = (0, 255, 0) if (face_found and in_zone) else \
                      ((0, 220, 255) if face_found else (0, 80, 255))
        status_scale = UI["font_scale"] * 1.3
        status_thick = max(2, UI["thickness"])
        sw2 = cv2.getTextSize(
            status_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            status_scale,
            status_thick)[0][0]
        cv2.putText(
            display,
            status_text,
            (w // 2 - sw2 // 2, int(h * 0.92)),
            cv2.FONT_HERSHEY_SIMPLEX,
            status_scale,
            status_col,
            status_thick,
            cv2.LINE_AA)
        btn_col = (0, 200, 80) if jy > JOY_DOWN else (30, 30, 30)

        bw, bh = int(w * 0.55), int(h * 0.11)
        bx, by = w // 2 - bw // 2, int(h * 0.76)

        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), btn_col, -1)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (120, 120, 120), 2)

        line1 = "Take a picture of face"
        line2 = "down/space - shot"
        font = cv2.FONT_HERSHEY_SIMPLEX

        max_width = int(bw * 0.90)

        text_scale = 1.0
        text_thickness = 2

        while text_scale > 0.3:
            l1_size = cv2.getTextSize(
            line1, font, text_scale, text_thickness)[0]

            l2_size = cv2.getTextSize(
            line2, font, text_scale, text_thickness)[0]
            if max(l1_size[0], l2_size[0]) <= max_width:
               break
    
            text_scale -= 0.05

        l1_w, l1_h = l1_size
        l2_w, l2_h = l2_size

        gap = int(bh * 0.10)
        total_h = l1_h + l2_h + gap
        start_y = by + (bh - total_h) // 2 + l1_h

        cv2.putText(display,line1,(bx + (bw - l1_w) // 2, start_y),font,text_scale,(240, 240, 240),text_thickness,cv2.LINE_AA)
        cv2.putText(display,line2,(bx + (bw - l2_w) // 2, start_y + l2_h + gap),font,text_scale,(0, 255, 180),text_thickness,cv2.LINE_AA)
        draw_joystick_indicator(display, jx, jy, w - 50, h - 50)
        cv2.imshow('shooting', display)
       
            
    cap.release(); cv2.destroyAllWindows(); return None

# gender select
def phase_gender():
    print('Фаза: выбор пола')

    options = ['female', 'male']
    gi = 0

    wait_joystick_neutral(timeout_ms=2000)
    edge = JoyEdge()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('gender choice', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('gender choice', 1920, 1280)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        crop_w = int(w * 0.6)
        crop_h = int(h * 0.6)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        frame = frame[y1:y1+crop_h, x1:x1+crop_w]

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        jx, jy, sw = read_joystick()
        fired_left, fired_right, fired_up, fired_down, fired_push = edge.update(jx, jy, sw)

        if fired_left or fired_right:
            gi = (gi + 1) % 2
            cv2.waitKey(300)

        if fired_up:
            cap.release()
            cv2.destroyAllWindows()
            return gi  # 0 female, 1 male

        if fired_down:
            cap.release()
            cv2.destroyAllWindows()
            return None

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('a'), ord('d')):
            gi = (gi + 1) % 2
        elif key == 13:
            cap.release()
            cv2.destroyAllWindows()
            return gi
        elif key in (ord('q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            return None

        display = frame.copy()

        text = options[gi]
        draw_text(display, f'choice: {text}', (w//2 - 100, h//2), 1.0)

     

        draw_text(display, 'left / right - choose gender', (20, h - 80), 0.6)
        draw_text(display, 'up / Enter - confirm',      (20, h - 50), 0.6)
        draw_text(display, 'down - exit',               (20, h - 20), 0.6)
        cv2.imshow('gender choice', display)

    cap.release()
    cv2.destroyAllWindows()
    return None

# sample choice windpw
def phase_select(snapshot, allowed_indices=None):
    print('выбор шаблона')

    print('Загрузка шаблонов')
    loaded = []
    for t in TEMPLATES:
        bgr, dst = load_template(t['file'])
        loaded.append((bgr, dst))

    if allowed_indices is None:
        allowed_indices = list(range(len(TEMPLATES)))

    original_indices = [
        i for i in allowed_indices
        if loaded[i][0] is not None
    ]
    valid = [(TEMPLATES[i], loaded[i]) for i in original_indices]
    if not valid:
        print('Нет доступных шаблонов')
        return 'quit', None

    print('Рендер лица в шаблоны')
    previews = []
    for tpl, (bgr, dst) in valid:
        result, _, _, _ = composite_from_snapshot(snapshot, bgr, dst)
        previews.append(result if result is not None else bgr.copy())

    vi = 0
    wait_joystick_neutral(timeout_ms=3000)
    edge = JoyEdge()

    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

    cv2.namedWindow('select picture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('select picture', 1920, 1280)
    
    if not cap.isOpened():
        print('Камера не найдена!')
        return 'quit', None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release(); cv2.destroyAllWindows(); return 'quit', None

        
        h, w = frame.shape[:2]
        crop_w = int(w * 0.9)
        crop_h = int(h * 0.9)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        frame = frame[y1:y1+crop_h, x1:x1+crop_w]

        frame = cv2.flip(frame, 1)
        
        
        display_frame = cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC)
        h, w = display_frame.shape[:2]
        
        UI = {
            "pad":        int(w * 0.02),
            "font_scale": w / 900,
            "thickness":  max(1, int(w / 500))
        }

        jx, jy, sw = read_joystick()
        fired_left, fired_right, fired_up, fired_down, fired_push = edge.update(jx, jy, sw)

        if fired_left:
            vi = (vi - 1) % len(valid)
            cv2.waitKey(500)
        elif fired_right:
            vi = (vi + 1) % len(valid)
            cv2.waitKey(500)

       
        if fired_push:
            desc_idx    = original_indices[vi]
            description = (ORNAMENT_DESCRIPTIONS[desc_idx]
                           if desc_idx < len(ORNAMENT_DESCRIPTIONS)
                           else TEMPLATES[desc_idx]['name'])
            show_ornament_info(description, previews[vi])
            wait_joystick_neutral(timeout_ms=500)
            edge = JoyEdge()   #
            continue
        # 

        if fired_up:
            chosen = previews[vi]
            cv2.imwrite(SAVE_PATH, chosen)
            print(f'Сохранено: {SAVE_PATH}')
            cap.release(); cv2.destroyAllWindows()
            return 'saved', original_indices[vi]

        if fired_down:
            cap.release(); cv2.destroyAllWindows()
            return 'retake', None

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('a'), ord('A'), 81):
            vi = (vi - 1) % len(valid)
        elif key in (ord('d'), ord('D'), 83):
            vi = (vi + 1) % len(valid)
        elif key == ord('i'):
            desc_idx    = original_indices[vi]
            description = (ORNAMENT_DESCRIPTIONS[desc_idx]
                           if desc_idx < len(ORNAMENT_DESCRIPTIONS)
                           else TEMPLATES[desc_idx]['name'])
            show_ornament_info(description, previews[vi])
            edge = JoyEdge()
        elif key == 13:
            chosen = previews[vi]
            cv2.imwrite(SAVE_PATH, chosen)
            print(f'Сохранено: {SAVE_PATH}')
            cap.release(); cv2.destroyAllWindows()
            return 'saved', original_indices[vi]
        elif key in (ord('r'), ord('R')):
            cap.release(); cv2.destroyAllWindows()
            return 'retake', None
        elif key in (ord('q'), 27):
            cap.release(); cv2.destroyAllWindows()
            return 'quit', None

        tpl, _ = valid[vi]
        name   = tpl['name']
        total  = len(valid)
        bgr_preview = previews[vi]

        
        display = overlay_template_on_frame(display_frame.copy(), bgr_preview, h, w)

        left_col  = (0, 255, 180) if jx < JOY_LEFT  else (180, 180, 180)
        right_col = (0, 255, 180) if jx > JOY_RIGHT else (180, 180, 180)
        draw_text(display, '<', (int(w * 0.03), h // 2),
                  UI["font_scale"] * 2, left_col, UI["thickness"] + 2)
        draw_text(display, '>', (int(w * 0.95), h // 2),
                  UI["font_scale"] * 2, right_col, UI["thickness"] + 2)

        scale  = UI["font_scale"] * 1.2
        tw_px  = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, UI["thickness"])[0][0]
        draw_text(display, name, (w // 2 - tw_px // 2, int(h * 0.1)),
                  scale, (255, 255, 255), UI["thickness"])
        draw_text(display, f'{vi+1} / {total}',
                  (w // 2 - 22, h - 85), 0.6, (160, 160, 160), 1)

       
        up_col = (0, 200, 80) if jy < JOY_UP else (60, 60, 60)
        dn_col = (0, 80, 220) if jy > JOY_DOWN else (60, 60, 60)
        
        text_up = 'left/right - scrolling  up/enter - save'
        text_dn = 'SW/i - ornament info  down/R - new shoot'
        
        btn_font_scale = w / 1100  
        btn_thickness = max(1, int(w / 800))

        (w_up, h_up), _ = cv2.getTextSize(text_up, cv2.FONT_HERSHEY_SIMPLEX, btn_font_scale, btn_thickness)
        (w_dn, h_dn), _ = cv2.getTextSize(text_dn, cv2.FONT_HERSHEY_SIMPLEX, btn_font_scale, btn_thickness)

        padding_x = int(w * 0.02)
        padding_y = int(h * 0.015)
        
        bw = max(w_up, w_dn) + (padding_x * 2)  
        bh = max(h_up, h_dn) + (padding_y * 2)  

        
        bx, by = w // 2 - bw // 2, int(h * 0.74)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), up_col, -1)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)
        text_x1 = bx + (bw - w_up) // 2
        text_y1 = by + (bh + h_up) // 2 - 2  
        draw_text(display, text_up, (text_x1, text_y1), btn_font_scale, (255, 255, 255), btn_thickness)

        
        bx2, by2 = w // 2 - bw // 2, int(h * 0.85)
        cv2.rectangle(display, (bx2, by2), (bx2 + bw, by2 + bh), dn_col, -1)
        cv2.rectangle(display, (bx2, by2), (bx2 + bw, by2 + bh), (100, 100, 100), 1)
        text_x2 = bx2 + (bw - w_dn) // 2
        text_y2 = by2 + (bh + h_dn) // 2 - 2
        draw_text(display, text_dn, (text_x2, text_y2), btn_font_scale, (255, 255, 255), btn_thickness)
        # 

        cv2.imshow('select picture', display)

    cap.release(); cv2.destroyAllWindows(); return 'quit', None
#observing the result
def show_result(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f'Не удалось открыть результат: {image_path}')
        return

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 1920, 1080)

    h, w = img.shape[:2]

    wait_joystick_neutral(timeout_ms=1500)
    edge = JoyEdge()

    while True:
        display = img.copy()

        cv2.putText(display, 'enter - exit',
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(display, 'enter - exit',
                    (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display, 'joystick down - plotter print',
                    (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imshow('result', display)

        jx, jy, sw = read_joystick()
        _, _, _, fired_down, _ = edge.update(jx, jy, sw)

        key = cv2.waitKey(1) & 0xFF

        if fired_down:
            cv2.destroyAllWindows()
            return 'print'

        if key in (13, ord('q'), 27, ord(' ')):
            cv2.destroyAllWindows()
            return 'exit'

# sending g code to plotter
def send_gcode_to_plotter(gcode_path, port, baud=115200):
    if not os.path.exists(gcode_path):
        print(f'G-code файл не найден: {gcode_path}')
        return

    print(f'Подключение к плоттеру ({port}, {baud} baud)...')
    try:
        plotter = serial.Serial(port, baud, timeout=10)
    except Exception as e:
        print(f'Ошибка подключения к плоттеру: {e}')
        return

    time.sleep(2)
    plotter.flushInput()
    plotter.write(b"$#\n")
    plotter.flush()

    time.sleep(0.5)
    

    while plotter.in_waiting:
        print(plotter.readline().decode())
    plotter.write(b"G92 X0 Y0 Z0\n")

    greeting = plotter.readline().decode('utf-8', errors='ignore').strip()
    if greeting:
        print(f'GRBL: {greeting}')

    print(f'Отправка файла: {gcode_path}')
    with open(gcode_path, 'r') as f:
        lines = f.readlines()

    total = len([l for l in lines if l.strip() and not l.strip().startswith(';')])
    sent  = 0

    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue

        plotter.write((line + '\n').encode('utf-8'))
        plotter.flush()
        sent += 1

        while True:
            response = plotter.readline().decode('utf-8', errors='ignore').strip()
            if response.lower().startswith('ok'):
                break
            elif response.lower().startswith('error'):
                print(f'  GRBL ошибка на строке "{line}": {response}')
                break
            elif response:
                print(f'  GRBL: {response}')

        if sent % 20 == 0 or sent == total:
            print(f'  Прогресс: {sent}/{total} команд отправлено')

    print('Печать завершена')
    plotter.close()

# plotter set up
PLOTTER_PORT = 'COM18'
PLOTTER_BAUD = 115200

GCODE_FILES = [
    os.path.join(BASE_DIR, 'icon1.gcode'),
    os.path.join(BASE_DIR, 'icon2.gcode'),
    os.path.join(BASE_DIR, 'icon3.gcode'),
    os.path.join(BASE_DIR, 'icon4.gcode'),
    os.path.join(BASE_DIR, 'icon5.gcode'),
    os.path.join(BASE_DIR, 'icon6.gcode'),
    os.path.join(BASE_DIR, 'icon7.gcode'),
    os.path.join(BASE_DIR, 'icon8.gcode'),
]


if __name__ == '__main__':
    print('Запуск')
    chosen_index = None

    while True:
       
        snapshot = phase_photo()
        if snapshot is None:
            print('Выход.')
            break

       
        gender = phase_gender()
        if gender is None:
            print('Выход.')
            break

        
        if gender == 0:   # female
            allowed = [0, 1, 2, 3]
        else:             # male
            allowed = [4, 5, 6, 7]

        # 
        result, chosen_index = phase_select(snapshot, allowed)

        if result == 'saved':
            print(f'Готово! Файл сохранён: {SAVE_PATH}')

            os.startfile(SAVE_PATH)
            action = 'print'

            if action == 'print':
                if chosen_index is not None and chosen_index < len(GCODE_FILES):
                    gcode = GCODE_FILES[chosen_index]
                    print(f'Отправка на плоттер: {gcode}')
                    
                    send_gcode_to_plotter(gcode, PLOTTER_PORT, PLOTTER_BAUD)
            else:
                print('Печать пропущена')
            break

        elif result == 'retake':
            print('Повторная съёмка')
            wait_joystick_neutral()
            continue
        else:  # 'quit'
            print('Выход без сохранения')
            break

    if ser and not ser.closed:
        ser.close()
