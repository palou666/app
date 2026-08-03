# ================================================================
# 第1层：导入层
# ================================================================
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.text import LabelBase
import cv2
import numpy as np
import os
import math
import time
import random

# ---- 语音引擎：使用 plyer（Android 原生 TTS） ----
try:
    from plyer import tts
    TTS_AVAILABLE = True
    print("✅ 语音引擎: plyer (Android TTS)")
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ plyer 未安装，语音功能不可用。执行: pip install plyer")

# 解决Kivy中文显示方框问题（使用项目内的字体文件）
# 请将 NotoSansCJK-Regular.ttf 放在项目根目录
# 如果没有该字体，可以删除下面两行，使用默认字体（中文可能为方框）
try:
    LabelBase.register(name='Roboto', fn_regular='NotoSansCJK-Regular.ttf')
except Exception:
    pass  # 若找不到字体则使用默认，不影响运行

# ================================================================
# 第2层：工具层 - 姿态分析器
# ================================================================
class PostureAnalyzer:
    """
    五种姿态检测（低头最优先）：
    1. 低头/头前倾  -> 鼻子垂直下降量（最优先）
    2. 歪头/侧颈    -> 鼻子水平偏移（但肩差小）
    3. 身体侧倾     -> 双肩高度差
    4. 圆肩驼背     -> 肩宽/髋宽比
    5. 后仰瘫坐     -> 躯干后倾角度
    """

    # ===== 阈值配置（可随时调整） =====
    DROP_THRESHOLD = 8                # 低头：鼻子下降量（像素），越小越灵敏
    SHOULDER_DIFF_THRESHOLD = 15      # 侧倾：双肩高度差（像素）
    HEAD_TILT_THRESHOLD = 25          # 歪头：鼻子水平偏移（像素）
    HEAD_TILT_SHOULDER_THRESHOLD = 10 # 歪头时肩差需小于此值
    RECLINE_ANGLE_THRESHOLD = 15      # 后仰：躯干后倾角度（度）
    SHOULDER_HIP_RATIO = 0.7          # 圆肩：肩宽/髋宽比低于此值
    CONSECUTIVE_FRAMES = 2            # 连续几帧触发才报警（防抖）

    def __init__(self):
        self.head_forward_count = 0
        self.side_tilt_count = 0
        self.round_shoulder_count = 0
        self.head_tilt_count = 0
        self.recline_count = 0
        self.last_posture = "正确姿势"

        self.baseline_vertical = None
        self.baseline_frames = 0
        self.posture_start_time = time.time()

    def analyze(self, keypoints):
        if not keypoints or len(keypoints) < 17:
            return "未检测", "请站在摄像头前", {}, time.time()

        nose = keypoints[0]
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[9]
        r_hip = keypoints[10]

        if nose[0] == 0 and nose[1] == 0:
            return "未检测", "鼻子未检测到", {}, time.time()

        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2

        vertical_offset = nose[1] - shoulder_mid_y
        horizontal_offset = nose[0] - shoulder_mid_x
        shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])

        trunk_vec_x = shoulder_mid_x - hip_mid_x
        trunk_vec_y = shoulder_mid_y - hip_mid_y
        if trunk_vec_y > 1:
            recline_angle = math.degrees(math.atan2(abs(trunk_vec_x), trunk_vec_y))
        else:
            recline_angle = 0.0

        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        hip_width = abs(l_hip[0] - r_hip[0])
        shoulder_hip_ratio = shoulder_width / (hip_width + 1) if hip_width > 0 else 1.0

        if self.baseline_frames < 30:
            if self.baseline_vertical is None:
                self.baseline_vertical = vertical_offset
            else:
                self.baseline_vertical = 0.9 * self.baseline_vertical + 0.1 * vertical_offset
            self.baseline_frames += 1
            return "正确姿势", "校准中...", {}, time.time()

        drop_amount = vertical_offset - self.baseline_vertical

        metrics = {
            'drop': drop_amount,
            'h_offset': horizontal_offset,
            'v_offset': vertical_offset,
            'shoulder_diff': shoulder_diff,
            'recline_angle': recline_angle,
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'baseline': self.baseline_vertical
        }

        # 低头
        if drop_amount > self.DROP_THRESHOLD:
            self.head_forward_count += 1
            self._reset_all_counts(exclude='head_forward')
            if self.head_forward_count >= self.CONSECUTIVE_FRAMES:
                self.last_posture = "低头/头前倾"
                self.posture_start_time = time.time()
                return "低头/头前倾", f"清阳不升证：头部下沉 {drop_amount:.0f}px", metrics, self.posture_start_time
            return self.last_posture, "检测中...", metrics, self.posture_start_time

        # 歪头
        if abs(horizontal_offset) > self.HEAD_TILT_THRESHOLD and shoulder_diff < self.HEAD_TILT_SHOULDER_THRESHOLD:
            self.head_tilt_count += 1
            self._reset_all_counts(exclude='head_tilt')
            if self.head_tilt_count >= self.CONSECUTIVE_FRAMES:
                self.last_posture = "歪头/侧颈"
                self.posture_start_time = time.time()
                return "歪头/侧颈", f"颈部侧屈：建议缓慢左右拉伸 (偏移{int(horizontal_offset)}px)", metrics, self.posture_start_time
            return self.last_posture, "检测中...", metrics, self.posture_start_time

        # 身体侧倾
        if shoulder_diff > self.SHOULDER_DIFF_THRESHOLD:
            self.side_tilt_count += 1
            self._reset_all_counts(exclude='side_tilt')
            if self.side_tilt_count >= self.CONSECUTIVE_FRAMES:
                self.last_posture = "身体侧倾"
                self.posture_start_time = time.time()
                return "身体侧倾", f"少阳经气郁滞：双肩差 {shoulder_diff:.0f}px", metrics, self.posture_start_time
            return self.last_posture, "检测中...", metrics, self.posture_start_time

        # 圆肩驼背
        if shoulder_hip_ratio < self.SHOULDER_HIP_RATIO:
            self.round_shoulder_count += 1
            self._reset_all_counts(exclude='round_shoulder')
            if self.round_shoulder_count >= self.CONSECUTIVE_FRAMES:
                self.last_posture = "圆肩驼背"
                self.posture_start_time = time.time()
                return "圆肩驼背", "心肺气机郁闭证：建议开弓射雕", metrics, self.posture_start_time
            return self.last_posture, "检测中...", metrics, self.posture_start_time

        # 后仰瘫坐
        if recline_angle > self.RECLINE_ANGLE_THRESHOLD:
            self.recline_count += 1
            self._reset_all_counts(exclude='recline')
            if self.recline_count >= self.CONSECUTIVE_FRAMES:
                self.last_posture = "后仰瘫坐"
                self.posture_start_time = time.time()
                return "后仰瘫坐", f"躯干后倾：建议坐直 (后倾{recline_angle:.1f}°)", metrics, self.posture_start_time
            return self.last_posture, "检测中...", metrics, self.posture_start_time

        # 正常
        self._reset_all_counts(exclude=None)
        self.last_posture = "正确姿势"
        self.posture_start_time = time.time()
        return "正确姿势", "气血通畅，阴阳平衡。继续保持！", metrics, self.posture_start_time

    def _reset_all_counts(self, exclude=None):
        if exclude != 'head_forward':
            self.head_forward_count = 0
        if exclude != 'side_tilt':
            self.side_tilt_count = 0
        if exclude != 'round_shoulder':
            self.round_shoulder_count = 0
        if exclude != 'head_tilt':
            self.head_tilt_count = 0
        if exclude != 'recline':
            self.recline_count = 0


# ================================================================
# 第3层：引擎层 - 姿态检测引擎
# ================================================================
class PoseDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = 320
        self.conf_threshold = 0.3

        if not os.path.exists(model_path):
            print(f"❌ 模型不存在: {model_path}")
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print("✅ ONNX 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.session = None

    def detect(self, frame):
        if self.session is None:
            return None

        h, w = frame.shape[:2]

        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        output = self.session.run(
            [self.output_name],
            {self.input_name: img}
        )[0]
        output = output.squeeze(0)

        scores = output[4, :]
        scores = 1 / (1 + np.exp(-scores))
        max_idx = np.argmax(scores)
        max_conf = scores[max_idx]

        if max_conf < self.conf_threshold:
            return None

        box = output[:, max_idx]
        kp_raw = box[5:56]

        keypoints = []
        for i in range(17):
            x = (kp_raw[i*3] / self.input_size) * w
            y = (kp_raw[i*3 + 1] / self.input_size) * h
            conf = 1 / (1 + np.exp(-kp_raw[i*3 + 2]))
            keypoints.append([x, y, conf])

        return {'keypoints': keypoints, 'success': True}


# ================================================================
# 第4层：界面层 - 摄像头界面（使用 plyer TTS）
# ================================================================
class CameraWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.detector = PoseDetector('yolov8n-pose.onnx')
        self.analyzer = PostureAnalyzer()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.add_widget(Button(text="❌ 摄像头打开失败", background_color=(1,0,0,1)))
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.img = Image()
        self.add_widget(self.img)

        self.status_btn = Button(
            text="⏳ 等待检测...",
            size_hint_y=0.08,
            background_color=(0.2, 0.6, 0.4, 1)
        )
        self.add_widget(self.status_btn)

        self.bad_posture_start_time = None
        self.last_reminder_time = 0
        self.remind_cooldown = 30
        self.bad_threshold = 5
        self.last_spoken_posture = None

        self.voice_messages = {
            "低头/头前倾": [
                "小背挺直，头抬高哦～",
                "检测到头部前倾，请将头部向后平移，让耳朵位于肩膀正上方。",
                "低头太久啦，抬头看看远处，休息一下吧。"
            ],
            "身体侧倾": [
                "坐正一点，别歪着身子哦。",
                "身体侧倾，请将重心均匀落在两臀之间。"
            ],
            "圆肩驼背": [
                "肩膀打开，挺起胸膛来～",
                "检测到圆肩驼背，请双肩向后绕半圈，然后沉肩，打开锁骨。"
            ],
            "后仰瘫坐": [
                "坐直一点，别瘫在椅子上哦。",
                "躯干后倾，请将坐骨坐实，让腰部自然挺起。"
            ],
            "歪头/侧颈": [
                "头摆正，别歪着哦。",
                "检测到颈部侧屈，请缓慢将头回到正中位置。"
            ]
        }

        if TTS_AVAILABLE:
            try:
                tts.speak("坐姿监测已启动")
                print("🔊 播报: 坐姿监测已启动")
            except Exception as e:
                print(f"启动播报失败: {e}")
        else:
            print("⚠️ 语音不可用，请安装 plyer: pip install plyer")

        Clock.schedule_interval(self.update_frame, 1.0/25.0)

    def draw_skeleton(self, frame, keypoints):
        if not keypoints:
            return frame

        points = []
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.3:
                points.append((int(x), int(y)))
            else:
                points.append((0, 0))

        skeleton = [
            (0,1),(0,2),(1,3),(2,4),
            (3,5),(4,6),
            (5,7),(6,8),
            (5,9),(6,10),
            (9,11),(10,12),
            (11,13),(12,14),
            (5,6),(9,10)
        ]

        for i, j in skeleton:
            if i < len(points) and j < len(points):
                if points[i] != (0,0) and points[j] != (0,0):
                    cv2.line(frame, points[i], points[j], (0, 255, 0), 2)

        for pt in points:
            if pt != (0,0):
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        return frame

    def _speak(self, text):
        if not TTS_AVAILABLE:
            print(f"🔇 语音未播报（引擎不可用）: {text}")
            return
        try:
            tts.speak(text)
            print(f"🔊 播报: {text}")
        except Exception as e:
            print(f"❌ 语音播报失败: {e}")

    def _get_voice_message(self, posture_type):
        messages = self.voice_messages.get(posture_type, [f"注意，{posture_type}，请调整姿势。"])
        return random.choice(messages)

    def update_frame(self, dt):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        result = self.detector.detect(frame)

        posture_type = "未检测"
        advice_text = "请站在摄像头前"
        metrics = {}
        color = (0, 255, 255)

        if result and result.get('success'):
            keypoints = result.get('keypoints', [])
            frame = self.draw_skeleton(frame, keypoints)

            posture_type, advice_text, metrics, posture_start_time = self.analyzer.analyze(keypoints)

            if posture_type == "正确姿势":
                color = (0, 255, 0)
            elif posture_type == "检测中...":
                color = (255, 255, 0)
            else:
                color = (0, 0, 255)

            current_time = time.time()
            is_bad_posture = posture_type not in ["正确姿势", "检测中...", "未检测"]

            if is_bad_posture:
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = current_time

                bad_duration = current_time - self.bad_posture_start_time

                if (bad_duration > self.bad_threshold and 
                    current_time - self.last_reminder_time > self.remind_cooldown and
                    posture_type != self.last_spoken_posture):

                    message = self._get_voice_message(posture_type)
                    self._speak(message)
                    self.last_reminder_time = current_time
                    self.last_spoken_posture = posture_type
                    self.bad_posture_start_time = current_time

                cv2.putText(frame, f"不良持续: {int(bad_duration)}s", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            else:
                self.bad_posture_start_time = None
                if posture_type == "正确姿势":
                    self.last_spoken_posture = None

            drop = metrics.get('drop', 0)
            h_off = metrics.get('h_offset', 0)
            shoulder_diff = metrics.get('shoulder_diff', 0)
            recline = metrics.get('recline_angle', 0)
            baseline = metrics.get('baseline', 0)

            cv2.putText(frame, f"姿态: {posture_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame,
                        f"下沉:{int(drop)}px  水平:{int(h_off)}px  肩差:{int(shoulder_diff)}px  后倾:{int(recline)}°",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, f"基线:{int(baseline)}px", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(frame, advice_text[:35], (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            self.status_btn.text = f"✅ {posture_type}"

        else:
            self.status_btn.text = "⏳ 未检测到人体"
            self.analyzer._reset_all_counts(exclude=None)
            self.bad_posture_start_time = None
            self.last_spoken_posture = None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(frame_rgb, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img.texture = texture

    def on_stop(self):
        if self.cap:
            self.cap.release()


# ================================================================
# 第5层：启动层
# ================================================================
class PostureApp(App):
    def build(self):
        return CameraWidget()

if __name__ == '__main__':
    PostureApp().run()