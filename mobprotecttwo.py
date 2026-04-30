# ==============================
# COLAB BIOMECHANICS VISUALIZER
# FULL REWRITE WITH TOGGLES + STABILITY POLYGON + TIMELINE JSON
# + MEDIAPIPE .TASK (POSE LANDMARKER)
# ==============================

# Install deps
!pip -q install mediapipe opencv-python-headless

# ------------------------------
# (Optional) Download a .task model for Pose Landmarker
# ------------------------------
import os, urllib.request

TASK_PATH = "pose_landmarker_lite.task"
if not os.path.exists(TASK_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, TASK_PATH)

# ------------------------------
# Imports
# ------------------------------
from google.colab import files
from IPython.display import Video, display
import cv2
import mediapipe as mp
import numpy as np
import math
import json

# Tasks API
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ------------------------------
# 1) UPLOAD VIDEO
# ------------------------------
uploaded = files.upload()
INPUT_VIDEO = next(iter(uploaded.keys()))
OUTPUT_VIDEO = "annotated_biomechanics.mp4"
TIMELINE_JSON = "timeline.json"

# ------------------------------
# 2) TOGGLES
# ------------------------------
SHOW = {
    "landmarks": True,
    "connections": True,
    "joint_angles": True,
    "angle_arcs": True,
    "joint_labels": True,
    "hand_vectors": True,
    "hand_xy_components": True,
    "foot_vectors": True,
    "com": True,
    "cog": True,
    "support_polygon": True,
    "kinetic_chain": True,
    "stats_box": True,
    "title": True,
    "stability_state": True
}

# ------------------------------
# 3) VIDEO / PHYSICS SETTINGS
# ------------------------------
TARGET_W, TARGET_H = 420, 640
BODY_WEIGHT_KG = 24
G = 9.81
BODY_WEIGHT_N = BODY_WEIGHT_KG * G
PX_PER_NEWTON = 0.06
STABLE_SCORE_THRESHOLD = 60.0

# ------------------------------
# 4) MEDIAPIPE INIT (Tasks + Fallback)
# ------------------------------
# Tasks (preferred)
base_options = mp_tasks.BaseOptions(model_asset_path=TASK_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_task = mp_vision.PoseLandmarker.create_from_options(options)

# Fallback (Solutions)
mp_pose = mp.solutions.pose
pose_sol = mp_pose.Pose(
    model_complexity=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark

# Landmark index mapping for Tasks -> names (same order as Solutions)
IDX = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

# ------------------------------
# 5) HELPERS
# ------------------------------
def pt_norm(xn, yn, w, h):
    return np.array([int(xn * w), int(yn * h)], dtype=np.int32)

def midpoint(a, b):
    return np.array([(a[0] + b[0]) // 2, (a[1] + b[1]) // 2], dtype=np.int32)

def safe_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (nba * nbc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def draw_arc(frame, a, b, c, color=(0, 255, 255), radius=28, thickness=2):
    ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
    ang1 = math.degrees(math.atan2(ba[1], ba[0]))
    ang2 = math.degrees(math.atan2(bc[1], bc[0]))
    if ang1 < 0: ang1 += 360
    if ang2 < 0: ang2 += 360
    diff = (ang2 - ang1) % 360
    start, end = (ang1, ang2) if diff <= 180 else (ang2, ang1)
    cv2.ellipse(frame, (int(b[0]), int(b[1])), (radius, radius), 0, start, end, color, thickness)

def draw_label(frame, text, pos, color=(255,255,255), scale=0.55, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_arrow(frame, start, end, color, thickness=3, tip=0.25):
    cv2.arrowedLine(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, thickness, tipLength=tip)

def normalize(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n >= 1e-6 else np.array([0.0, 0.0], dtype=np.float32)

def estimate_com(kp):
    segments = [
        ("head", kp["NOSE"], 0.081),
        ("trunk", midpoint(kp["LEFT_SHOULDER"], kp["LEFT_HIP"]), 0.497),
        ("l_upper_arm", midpoint(kp["LEFT_SHOULDER"], kp["LEFT_ELBOW"]), 0.028),
        ("r_upper_arm", midpoint(kp["RIGHT_SHOULDER"], kp["RIGHT_ELBOW"]), 0.028),
        ("l_lower_arm", midpoint(kp["LEFT_ELBOW"], kp["LEFT_WRIST"]), 0.016),
        ("r_lower_arm", midpoint(kp["RIGHT_ELBOW"], kp["RIGHT_WRIST"]), 0.016),
        ("l_thigh", midpoint(kp["LEFT_HIP"], kp["LEFT_KNEE"]), 0.100),
        ("r_thigh", midpoint(kp["RIGHT_HIP"], kp["RIGHT_KNEE"]), 0.100),
        ("l_shank", midpoint(kp["LEFT_KNEE"], kp["LEFT_ANKLE"]), 0.0465),
        ("r_shank", midpoint(kp["RIGHT_KNEE"], kp["RIGHT_ANKLE"]), 0.0465),
        ("l_foot", midpoint(kp["LEFT_ANKLE"], kp["LEFT_FOOT_INDEX"]), 0.0145),
        ("r_foot", midpoint(kp["RIGHT_ANKLE"], kp["RIGHT_FOOT_INDEX"]), 0.0145),
    ]
    total, com = 0.0, np.array([0.0, 0.0], dtype=np.float32)
    for _, p, w in segments:
        com += np.array(p, dtype=np.float32) * w
        total += w
    return np.array(com / total, dtype=np.int32) if total >= 1e-6 else np.array([0,0], dtype=np.int32)

def build_support_polygon(kp):
    pts = np.array([
        kp["LEFT_HEEL"], kp["RIGHT_HEEL"],
        kp["LEFT_FOOT_INDEX"], kp["RIGHT_FOOT_INDEX"],
        kp["LEFT_ANKLE"], kp["RIGHT_ANKLE"]
    ], dtype=np.int32).reshape(-1,1,2)
    if len(pts) < 3: return None
    hull = cv2.convexHull(pts)
    return hull.reshape(-1,2) if hull is not None and len(hull) >= 3 else None

def torso_upright_score(kp):
    sm = midpoint(kp["LEFT_SHOULDER"], kp["RIGHT_SHOULDER"])
    hm = midpoint(kp["LEFT_HIP"], kp["RIGHT_HIP"])
    v = sm.astype(np.float32) - hm.astype(np.float32)
    nv = np.linalg.norm(v)
    if nv < 1e-6: return 0.0
    vertical = np.array([0.0, -1.0], dtype=np.float32)
    cos = np.clip(np.dot(v/nv, vertical), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return max(0.0, 1.0 - min(ang/45.0, 1.0))

def knee_control_score(lk, rk):
    avg = 0.5*(lk+rk); spread = abs(lk-rk)
    flex = math.exp(-((avg-145.0)**2)/(2*35.0*35.0))
    sym = max(0.0, 1.0 - min(spread/90.0, 1.0))
    return 0.7*flex + 0.3*sym

def stance_width_score(kp):
    ad = np.linalg.norm(kp["LEFT_ANKLE"].astype(np.float32)-kp["RIGHT_ANKLE"].astype(np.float32))
    sd = np.linalg.norm(kp["LEFT_SHOULDER"].astype(np.float32)-kp["RIGHT_SHOULDER"].astype(np.float32))
    if sd < 1e-6: return 0.0
    return max(0.0, min((ad/sd)/1.4, 1.0))

def compute_stability(kp, com, poly, angles, h):
    torso = torso_upright_score(kp)
    knee = knee_control_score(angles["Left Knee"], angles["Right Knee"])
    width = stance_width_score(kp)
    if poly is not None and len(poly) >= 3:
        d = cv2.pointPolygonTest(poly.astype(np.int32), (int(com[0]), int(com[1])), True)
        inside = d >= 0
        base = 1.0 if inside else max(0.0, 1.0 + d/(0.25*h))
    else:
        inside = False; base = 0.35
    score = (0.50*base + 0.20*torso + 0.15*knee + 0.15*width) * 100.0
    stable = (score >= STABLE_SCORE_THRESHOLD) and inside
    return stable, score, inside

# ------------------------------
# 6) VIDEO I/O
# ------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (TARGET_W, TARGET_H))

timeline = []
frame_idx = 0

# ------------------------------
# 7) PROCESS VIDEO
# ------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (TARGET_W, TARGET_H))
    h, w = frame.shape[:2]

    # ---- Try Tasks API ----
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res_task = pose_task.detect_for_video(mp_img, int(frame_idx * (1000.0 / fps)))

    if res_task.pose_landmarks:
        lm = res_task.pose_landmarks[0]  # list of 33 normalized landmarks
        kp = {
            "NOSE": pt_norm(lm[IDX["NOSE"]].x, lm[IDX["NOSE"]].y, w, h),

            "LEFT_SHOULDER": pt_norm(lm[IDX["LEFT_SHOULDER"]].x, lm[IDX["LEFT_SHOULDER"]].y, w, h),
            "RIGHT_SHOULDER": pt_norm(lm[IDX["RIGHT_SHOULDER"]].x, lm[IDX["RIGHT_SHOULDER"]].y, w, h),

            "LEFT_ELBOW": pt_norm(lm[IDX["LEFT_ELBOW"]].x, lm[IDX["LEFT_ELBOW"]].y, w, h),
            "RIGHT_ELBOW": pt_norm(lm[IDX["RIGHT_ELBOW"]].x, lm[IDX["RIGHT_ELBOW"]].y, w, h),

            "LEFT_WRIST": pt_norm(lm[IDX["LEFT_WRIST"]].x, lm[IDX["LEFT_WRIST"]].y, w, h),
            "RIGHT_WRIST": pt_norm(lm[IDX["RIGHT_WRIST"]].x, lm[IDX["RIGHT_WRIST"]].y, w, h),

            "LEFT_HIP": pt_norm(lm[IDX["LEFT_HIP"]].x, lm[IDX["LEFT_HIP"]].y, w, h),
            "RIGHT_HIP": pt_norm(lm[IDX["RIGHT_HIP"]].x, lm[IDX["RIGHT_HIP"]].y, w, h),

            "LEFT_KNEE": pt_norm(lm[IDX["LEFT_KNEE"]].x, lm[IDX["LEFT_KNEE"]].y, w, h),
            "RIGHT_KNEE": pt_norm(lm[IDX["RIGHT_KNEE"]].x, lm[IDX["RIGHT_KNEE"]].y, w, h),

            "LEFT_ANKLE": pt_norm(lm[IDX["LEFT_ANKLE"]].x, lm[IDX["LEFT_ANKLE"]].y, w, h),
            "RIGHT_ANKLE": pt_norm(lm[IDX["RIGHT_ANKLE"]].x, lm[IDX["RIGHT_ANKLE"]].y, w, h),

            "LEFT_HEEL": pt_norm(lm[IDX["LEFT_HEEL"]].x, lm[IDX["LEFT_HEEL"]].y, w, h),
            "RIGHT_HEEL": pt_norm(lm[IDX["RIGHT_HEEL"]].x, lm[IDX["RIGHT_HEEL"]].y, w, h),

            "LEFT_FOOT_INDEX": pt_norm(lm[IDX["LEFT_FOOT_INDEX"]].x, lm[IDX["LEFT_FOOT_INDEX"]].y, w, h),
            "RIGHT_FOOT_INDEX": pt_norm(lm[IDX["RIGHT_FOOT_INDEX"]].x, lm[IDX["RIGHT_FOOT_INDEX"]].y, w, h),
        }
    else:
        # ---- Fallback to Solutions ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_sol = pose_sol.process(rgb)
        if not res_sol.pose_landmarks:
            writer.write(frame); frame_idx += 1; continue
        lm = res_sol.pose_landmarks.landmark
        def p(i): return np.array([int(lm[i].x*w), int(lm[i].y*h)], dtype=np.int32)
        kp = {
            "NOSE": p(L.NOSE.value),
            "LEFT_SHOULDER": p(L.LEFT_SHOULDER.value),
            "RIGHT_SHOULDER": p(L.RIGHT_SHOULDER.value),
            "LEFT_ELBOW": p(L.LEFT_ELBOW.value),
            "RIGHT_ELBOW": p(L.RIGHT_ELBOW.value),
            "LEFT_WRIST": p(L.LEFT_WRIST.value),
            "RIGHT_WRIST": p(L.RIGHT_WRIST.value),
            "LEFT_HIP": p(L.LEFT_HIP.value),
            "RIGHT_HIP": p(L.RIGHT_HIP.value),
            "LEFT_KNEE": p(L.LEFT_KNEE.value),
            "RIGHT_KNEE": p(L.RIGHT_KNEE.value),
            "LEFT_ANKLE": p(L.LEFT_ANKLE.value),
            "RIGHT_ANKLE": p(L.RIGHT_ANKLE.value),
            "LEFT_HEEL": p(L.LEFT_HEEL.value),
            "RIGHT_HEEL": p(L.RIGHT_HEEL.value),
            "LEFT_FOOT_INDEX": p(L.LEFT_FOOT_INDEX.value),
            "RIGHT_FOOT_INDEX": p(L.RIGHT_FOOT_INDEX.value),
        }
        if SHOW["connections"]:
            mp_drawing.draw_landmarks(frame, res_sol.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # ---------------- Angles ----------------
    angles = {
        "Left Shoulder": safe_angle(kp["LEFT_ELBOW"], kp["LEFT_SHOULDER"], kp["LEFT_HIP"]),
        "Right Shoulder": safe_angle(kp["RIGHT_ELBOW"], kp["RIGHT_SHOULDER"], kp["RIGHT_HIP"]),
        "Left Elbow": safe_angle(kp["LEFT_SHOULDER"], kp["LEFT_ELBOW"], kp["LEFT_WRIST"]),
        "Right Elbow": safe_angle(kp["RIGHT_SHOULDER"], kp["RIGHT_ELBOW"], kp["RIGHT_WRIST"]),
        "Left Hip": safe_angle(kp["LEFT_SHOULDER"], kp["LEFT_HIP"], kp["LEFT_KNEE"]),
        "Right Hip": safe_angle(kp["RIGHT_SHOULDER"], kp["RIGHT_HIP"], kp["RIGHT_KNEE"]),
        "Left Knee": safe_angle(kp["LEFT_HIP"], kp["LEFT_KNEE"], kp["LEFT_ANKLE"]),
        "Right Knee": safe_angle(kp["RIGHT_HIP"], kp["RIGHT_KNEE"], kp["RIGHT_ANKLE"]),
        "Left Ankle": safe_angle(kp["LEFT_KNEE"], kp["LEFT_ANKLE"], kp["LEFT_FOOT_INDEX"]),
        "Right Ankle": safe_angle(kp["RIGHT_KNEE"], kp["RIGHT_ANKLE"], kp["RIGHT_FOOT_INDEX"]),
    }

    # ---------------- COM / Polygon / Stability ----------------
    com = estimate_com(kp)
    poly = build_support_polygon(kp)
    stable, stability_score, inside = compute_stability(kp, com, poly, angles, h)

    # ---------------- Forces ----------------
    lk, rk = angles["Left Knee"], angles["Right Knee"]
    lb = 0.5 + ((180.0 - lk) / 180.0) * 0.35
    rb = 0.5 + ((180.0 - rk) / 180.0) * 0.35
    lg = 1.0 - min(max(kp["LEFT_ANKLE"][1] / float(h), 0.0), 1.0)
    rg = 1.0 - min(max(kp["RIGHT_ANKLE"][1] / float(h), 0.0), 1.0)
    lb += 0.15 * lg; rb += 0.15 * rg
    lb = max(lb, 0.05); rb = max(rb, 0.05)
    s = lb + rb
    ls, rs = lb/s, rb/s

    lfN = BODY_WEIGHT_N * ls
    rfN = BODY_WEIGHT_N * rs
    lfpx = int(np.clip(lfN * PX_PER_NEWTON, 12, 120))
    rfpx = int(np.clip(rfN * PX_PER_NEWTON, 12, 120))

    la = 0.35 + ((180.0 - angles["Left Elbow"]) / 180.0) * 0.25
    ra = 0.35 + ((180.0 - angles["Right Elbow"]) / 180.0) * 0.25
    lhfN = BODY_WEIGHT_N * 0.10 * la
    rhfN = BODY_WEIGHT_N * 0.10 * ra
    lhfpx = int(np.clip(lhfN * PX_PER_NEWTON, 8, 60))
    rhfpx = int(np.clip(rhfN * PX_PER_NEWTON, 8, 60))

    # ---------------- Draw ----------------
    if SHOW["landmarks"]:
        for k in ["NOSE","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
                  "LEFT_WRIST","RIGHT_WRIST","LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
                  "LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL","RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"]:
            cv2.circle(frame, tuple(kp[k]), 5, (0,255,0), -1)

    if SHOW["support_polygon"] and poly is not None and len(poly) >= 3:
        pc = (0,255,0) if stable else (0,0,255)
        cv2.polylines(frame, [poly.astype(np.int32)], True, pc, 2)
        fill = frame.copy()
        cv2.fillPoly(fill, [poly.astype(np.int32)], pc)
        frame = cv2.addWeighted(fill, 0.12, frame, 0.88, 0)

    if SHOW["com"]:
        cv2.circle(frame, tuple(com), 8, (255,0,255), -1)
        draw_label(frame, "COM", (int(com[0])+10, int(com[1])-10), (255,0,255), 0.6, 2)

    if SHOW["cog"]:
        cv2.line(frame, (int(com[0]), int(com[1])), (int(com[0]), h-1), (255,255,0), 2)
        cv2.circle(frame, (int(com[0]), h-1), 6, (255,255,0), -1)

    if SHOW["stability_state"]:
        lab = "STABLE" if stable else "UNSTABLE"
        col = (0,255,0) if stable else (0,0,255)
        draw_label(frame, f"{lab} {int(round(stability_score))}/100", (12, h-20), col, 0.8, 2)

    # Arcs + labels
    angle_specs = [
        ("Left Shoulder", kp["LEFT_ELBOW"], kp["LEFT_SHOULDER"], kp["LEFT_HIP"], (255,255,0), "LS"),
        ("Right Shoulder", kp["RIGHT_ELBOW"], kp["RIGHT_SHOULDER"], kp["RIGHT_HIP"], (255,255,0), "RS"),
        ("Left Elbow", kp["LEFT_SHOULDER"], kp["LEFT_ELBOW"], kp["LEFT_WRIST"], (0,255,255), "LE"),
        ("Right Elbow", kp["RIGHT_SHOULDER"], kp["RIGHT_ELBOW"], kp["RIGHT_WRIST"], (0,255,255), "RE"),
        ("Left Hip", kp["LEFT_SHOULDER"], kp["LEFT_HIP"], kp["LEFT_KNEE"], (255,0,255), "LH"),
        ("Right Hip", kp["RIGHT_SHOULDER"], kp["RIGHT_HIP"], kp["RIGHT_KNEE"], (255,0,255), "RH"),
        ("Left Knee", kp["LEFT_HIP"], kp["LEFT_KNEE"], kp["LEFT_ANKLE"], (0,200,255), "LK"),
        ("Right Knee", kp["RIGHT_HIP"], kp["RIGHT_KNEE"], kp["RIGHT_ANKLE"], (0,200,255), "RK"),
        ("Left Ankle", kp["LEFT_KNEE"], kp["LEFT_ANKLE"], kp["LEFT_FOOT_INDEX"], (200,255,0), "LA"),
        ("Right Ankle", kp["RIGHT_KNEE"], kp["RIGHT_ANKLE"], kp["RIGHT_FOOT_INDEX"], (200,255,0), "RA"),
    ]
    for name,a,b,c,color,short in angle_specs:
        if SHOW["angle_arcs"]:
            draw_arc(frame, a,b,c,color,26,2)
        if SHOW["joint_angles"] or SHOW["joint_labels"]:
            txt = []
            if SHOW["joint_labels"]: txt.append(short)
            if SHOW["joint_angles"]: txt.append(f"{int(round(angles[name]))}°")
            draw_label(frame, " ".join(txt), (int(b[0])+8, int(b[1])-8), (255,255,255), 0.5, 2)

    # Feet
    if SHOW["foot_vectors"]:
        lfs, rfs = kp["LEFT_FOOT_INDEX"], kp["RIGHT_FOOT_INDEX"]
        draw_arrow(frame, lfs, (lfs[0], lfs[1]-lfpx), (0,0,255), 3, 0.2)
        draw_arrow(frame, rfs, (rfs[0], rfs[1]-rfpx), (0,0,255), 3, 0.2)

    # Hands
    if SHOW["hand_vectors"] or SHOW["hand_xy_components"]:
        for side, fpx, fN in [("LEFT", lhfpx, lhfN), ("RIGHT", rhfpx, rhfN)]:
            elbow, wrist = kp[f"{side}_ELBOW"], kp[f"{side}_WRIST"]
            d = normalize(wrist - elbow)
            if np.linalg.norm(d) < 1e-6: d = np.array([1.0,0.0], dtype=np.float32)
            start = wrist; end = start + (d * fpx).astype(np.int32)
            if SHOW["hand_vectors"]:
                draw_arrow(frame, start, end, (255,0,0), 3, 0.2)
            if SHOW["hand_xy_components"]:
                draw_arrow(frame, start, start + np.array([fpx,0]), (255,180,0), 2, 0.25)
                draw_arrow(frame, start, start + np.array([0,-fpx]), (0,255,0), 2, 0.25)

    # Stats
    if SHOW["stats_box"]:
        bx, by = 10, 25
        cv2.rectangle(frame, (bx-5, by-20), (bx+270, by+270), (0,0,0), -1)
        cv2.rectangle(frame, (bx-5, by-20), (bx+270, by+270), (255,255,255), 1)
        if SHOW["title"]:
            draw_label(frame, "Biomechanics Overlay", (bx, by), (255,255,255), 0.6, 2)
            by += 28
        lines = [
            f"Left Knee:  {int(round(angles['Left Knee']))} deg",
            f"Right Knee: {int(round(angles['Right Knee']))} deg",
            f"L Foot F:   {int(round(lfN))} N",
            f"R Foot F:   {int(round(rfN))} N",
            f"Stability:  {int(round(stability_score))}/100",
            f"State:      {'STABLE' if stable else 'UNSTABLE'}"
        ]
        yy = by
        for line in lines:
            draw_label(frame, line, (bx, yy), (255,255,255), 0.5, 1)
            yy += 20

    # Timeline
    timeline.append({
        "frame": int(frame_idx),
        "time_sec": float(frame_idx / fps),
        "angles": {k: float(v) for k, v in angles.items()},
        "com": [int(com[0]), int(com[1])],
        "support_polygon": poly.astype(int).tolist() if poly is not None else None,
        "stable": bool(stable),
        "stability_score": float(stability_score),
        "support_inside_polygon": bool(inside),
        "left_foot_force_N": float(lfN),
        "right_foot_force_N": float(rfN),
        "left_hand_force_N": float(lhfN),
        "right_hand_force_N": float(rhfN),
    })

    writer.write(frame)
    frame_idx += 1

# ------------------------------
# 8) CLEANUP
# ------------------------------
cap.release()
writer.release()
pose_task.close()
pose_sol.close()

with open(TIMELINE_JSON, "w", encoding="utf-8") as f:
    json.dump(timeline, f, ensure_ascii=False, indent=2)

print("Saved:", OUTPUT_VIDEO)
print("Saved:", TIMELINE_JSON)
display(Video(OUTPUT_VIDEO, embed=True, width=420))
