# """
# Sentinel AI — app.py
# Deploy: Hugging Face Spaces (Streamlit)
# Run locally: streamlit run app.py
# """

# import streamlit as st
# import cv2
# import numpy as np
# import time
# import tempfile
# import os
# from datetime import datetime
# from ultralytics import YOLO
# import supervision as sv

# from analysis.behaviour import detect_behaviour, reset_history
# from analysis.suspicious import detect_suspicious
# from analysis.movement import detect_movement
# from analysis.pose_tasks import draw_skeleton

# # ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Sentinel AI",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🛡️",
# )

# # ── CSS ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');

# html, body, [class*="css"] {
#     font-family: 'Barlow Condensed', sans-serif;
#     background: #070b0f;
#     color: #c8d8e8;
# }
# section[data-testid="stSidebar"] {
#     background: #0b1018 !important;
#     border-right: 1px solid #1a2a3a;
# }
# div[data-testid="metric-container"] {
#     background: #0d1b2a;
#     border: 1px solid #1e3a52;
#     border-radius: 8px;
#     padding: 14px 18px;
# }
# div[data-testid="metric-container"] label {
#     color: #5a8fa8 !important;
#     font-family: 'Share Tech Mono', monospace !important;
#     font-size: .72rem !important;
#     letter-spacing: .12em;
#     text-transform: uppercase;
# }
# div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
#     font-family: 'Barlow Condensed', sans-serif !important;
#     font-size: 2.4rem !important;
#     font-weight: 800;
#     color: #00cfff !important;
#     line-height: 1;
# }
# .sec-hdr {
#     font-family: 'Share Tech Mono', monospace;
#     font-size: .75rem;
#     letter-spacing: .18em;
#     text-transform: uppercase;
#     color: #3a7a9c;
#     border-bottom: 1px solid #1a3a52;
#     padding-bottom: 4px;
#     margin-bottom: 10px;
#     margin-top: 4px;
# }
# .alert-badge {
#     display: block;
#     background: #1a0d0d;
#     border: 1px solid #8b0000;
#     border-left: 3px solid #ff2a2a;
#     border-radius: 4px;
#     padding: 6px 12px;
#     margin: 3px 0;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: .8rem;
#     color: #ff6b6b;
# }
# .action-badge {
#     display: block;
#     background: #0a1a10;
#     border: 1px solid #005522;
#     border-left: 3px solid #00ee66;
#     border-radius: 4px;
#     padding: 6px 12px;
#     margin: 3px 0;
#     font-family: 'Share Tech Mono', monospace;
#     font-size: .8rem;
#     color: #00ee88;
# }
# .nosignal {
#     height: 360px;
#     background: #080c10;
#     border: 1px solid #1a3a52;
#     border-radius: 6px;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     font-family: 'Share Tech Mono', monospace;
#     color: #2a5a7a;
#     font-size: 1rem;
#     letter-spacing: .2em;
# }
# .pill { display:inline-block; border-radius:20px; padding:3px 14px;
#         font-family:'Share Tech Mono',monospace; font-size:.75rem;
#         letter-spacing:.08em; margin-top:4px; }
# .pill-live  { background:#0a2a10; border:1px solid #00aa44; color:#00ee66; }
# .pill-video { background:#0a1a2a; border:1px solid #0066cc; color:#44aaff; }
# .pill-off   { background:#1a1a1a; border:1px solid #444;    color:#888;    }
# .stButton>button {
#     font-family: 'Share Tech Mono', monospace !important;
#     background: #0d1b2a !important;
#     border: 1px solid #1e4060 !important;
#     color: #00cfff !important;
#     border-radius: 4px !important;
#     width: 100%;
# }
# .stButton>button:hover {
#     background: #102030 !important;
#     border-color: #00cfff !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ── SESSION STATE ─────────────────────────────────────────────────────────────
# defaults = {
#     "running":       False,
#     "mode":          "none",
#     "people_hist":   [],
#     "time_hist":     [],
#     "alert_log":     [],
#     "actions":       [],
#     "people_count":  0,
#     "car_count":     0,
#     "bus_count":     0,
#     "alerts":        [],
#     "processed_frame": None,
#     "seen_ids":      set(),
#     "five_min_count": 0,
#     "start_time":    time.time(),
# }
# for k, v in defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ── LOAD MODEL (cached) ───────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8n.pt")

# @st.cache_resource
# def load_tracker():
#     return sv.ByteTrack()

# model          = load_model()
# byte_tracker   = load_tracker()
# box_annotator  = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # ── PROCESS FRAME ─────────────────────────────────────────────────────────────
# def process_frame(frame: np.ndarray):
#     frame = cv2.resize(frame, (640, 480))

#     results    = model(frame, verbose=False)[0]
#     detections = sv.Detections.from_ultralytics(results)
#     detections = byte_tracker.update_with_detections(detections)

#     # counts
#     people_count = 0
#     car_count    = 0
#     bus_count    = 0

#     if detections.class_id is not None:
#         for cid in detections.class_id:
#             if cid == 0:   people_count += 1
#             elif cid == 2: car_count    += 1
#             elif cid == 5: bus_count    += 1

#     # labels
#     labels = []
#     if detections.tracker_id is not None:
#         for tid in detections.tracker_id:
#             labels.append(f"ID {tid}")
#             if tid not in st.session_state.seen_ids:
#                 st.session_state.seen_ids.add(tid)
#                 st.session_state.five_min_count += 1

#     # skeleton + behaviour
#     actions = []
#     if detections.xyxy is not None and detections.class_id is not None:
#         for i, box in enumerate(detections.xyxy):
#             if detections.class_id[i] != 0 or i > 6:
#                 continue
#             x1, y1, x2, y2 = map(int, box)
#             pid = int(detections.tracker_id[i]) if detections.tracker_id is not None else i

#             # draw skeleton
#             frame = draw_skeleton(frame, (x1, y1, x2, y2))

#             # behaviour
#             behaviour = detect_behaviour(pid, (x1, y1, x2, y2))
#             actions.append({"id": pid, "action": behaviour})

#             # label on frame
#             color = (0, 255, 255) if behaviour == "Normal" else \
#                     (0, 165, 255) if behaviour == "Active" else \
#                     (0, 0, 255)
#             cv2.putText(frame, behaviour, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

#     # annotate boxes
#     frame = box_annotator.annotate(frame, detections)
#     frame = label_annotator.annotate(frame, detections, labels)

#     # alerts
#     alerts   = detect_suspicious(detections)
#     movement = detect_movement(detections)
#     all_alerts = alerts + movement
#     if people_count > 5:
#         all_alerts.append("⚠ Crowd Detected")

#     # people count overlay
#     cv2.rectangle(frame, (0, 0), (220, 36), (10, 20, 35), -1)
#     cv2.putText(frame, f"PEOPLE: {people_count}", (8, 26),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 207, 255), 2)

#     return frame, people_count, car_count, bus_count, all_alerts, actions


# # ══════════════════════════════════════════════════════════════════════════════
# #  SIDEBAR
# # ══════════════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("""
#     <div style='text-align:center;padding:12px 0 20px'>
#       <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.9rem;
#                   font-weight:800;color:#00cfff;letter-spacing:.08em;'>
#         SENTINEL AI
#       </div>
#       <div style='font-family:"Share Tech Mono",monospace;font-size:.62rem;
#                   color:#3a6a8a;letter-spacing:.22em;margin-top:2px;'>
#         SURVEILLANCE SYSTEM v3.0
#       </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # status pill
#     st.markdown("<div class='sec-hdr'>STATUS</div>", unsafe_allow_html=True)
#     if st.session_state.running:
#         pill_cls = "pill-live"  if st.session_state.mode == "camera" else "pill-video"
#         pill_txt = "LIVE CAMERA" if st.session_state.mode == "camera" else "VIDEO RUNNING"
#         st.markdown(f"<span class='pill {pill_cls}'>● {pill_txt}</span>", unsafe_allow_html=True)
#     else:
#         st.markdown("<span class='pill pill-off'>○ OFFLINE</span>", unsafe_allow_html=True)

#     st.divider()

#     # mode selection
#     st.markdown("<div class='sec-hdr'>SELECT MODE</div>", unsafe_allow_html=True)
#     mode_sel = st.radio("", ["📷 Live Camera", "🎬 Upload Video"],
#                         label_visibility="collapsed")

#     st.divider()

#     # controls
#     st.markdown("<div class='sec-hdr'>CONTROLS</div>", unsafe_allow_html=True)

#     if mode_sel == "📷 Live Camera":
#         if not st.session_state.running:
#             if st.button("▶  START CAMERA"):
#                 st.session_state.running = True
#                 st.session_state.mode    = "camera"
#                 reset_history()
#                 st.session_state.seen_ids.clear()
#                 st.session_state.five_min_count = 0
#                 st.rerun()
#         else:
#             if st.button("■  STOP"):
#                 st.session_state.running = False
#                 st.session_state.mode    = "none"
#                 st.rerun()

#     else:  # video upload
#         uploaded = st.file_uploader("Upload video",
#                                     type=["mp4", "avi", "mov", "mkv"],
#                                     label_visibility="collapsed")
#         if uploaded and not st.session_state.running:
#             if st.button("▶  PROCESS VIDEO"):
#                 st.session_state.running      = True
#                 st.session_state.mode         = "video"
#                 st.session_state.uploaded_file = uploaded
#                 reset_history()
#                 st.session_state.seen_ids.clear()
#                 st.session_state.five_min_count = 0
#                 st.rerun()
#         elif st.session_state.running and st.session_state.mode == "video":
#             if st.button("■  STOP"):
#                 st.session_state.running = False
#                 st.session_state.mode    = "none"
#                 st.rerun()

#     st.divider()

#     if st.button("🗑  Clear History"):
#         st.session_state.people_hist.clear()
#         st.session_state.time_hist.clear()
#         st.session_state.alert_log.clear()
#         st.rerun()

#     st.divider()
#     st.markdown(
#         "<div style='font-family:\"Share Tech Mono\",monospace;font-size:.6rem;"
#         "color:#1a3a5a;text-align:center;line-height:1.8;'>"
#         "Browser camera permission required<br>for live mode</div>",
#         unsafe_allow_html=True
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN CONTENT
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <div style='display:flex;align-items:baseline;gap:14px;margin-bottom:16px;'>
#   <span style='font-family:"Barlow Condensed",sans-serif;font-size:2rem;
#                font-weight:800;color:#fff;letter-spacing:.04em;'>
#     AI SURVEILLANCE DASHBOARD
#   </span>
# </div>
# """, unsafe_allow_html=True)

# # ── METRICS ───────────────────────────────────────────────────────────────────
# st.markdown("<div class='sec-hdr'>LIVE COUNTS</div>", unsafe_allow_html=True)
# m1, m2, m3, m4, m5 = st.columns(5)
# m1.metric("People Now",     st.session_state.people_count)
# m2.metric("Total Seen",     st.session_state.five_min_count)
# m3.metric("Cars",           st.session_state.car_count)
# m4.metric("Buses",          st.session_state.bus_count)
# m5.metric("Active Alerts",  len(st.session_state.alerts))

# st.markdown("<br>", unsafe_allow_html=True)

# # ── VIDEO FEED ────────────────────────────────────────────────────────────────
# st.markdown("<div class='sec-hdr'>FEED</div>", unsafe_allow_html=True)
# col_feed, col_right = st.columns([3, 2])

# with col_feed:
#     frame_placeholder = st.empty()

#     # ── LIVE CAMERA MODE ──────────────────────────────────────────────────────
#     if st.session_state.running and st.session_state.mode == "camera":
#         camera_frame = st.camera_input(
#             "📷 Camera Feed",
#             label_visibility="collapsed",
#             key="cam_input"
#         )

#         if camera_frame is not None:
#             # decode
#             file_bytes = np.asarray(bytearray(camera_frame.read()), dtype=np.uint8)
#             frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#             processed, pc, cc, bc, alerts, actions = process_frame(frame_bgr)

#             # update session
#             st.session_state.people_count = pc
#             st.session_state.car_count    = cc
#             st.session_state.bus_count    = bc
#             st.session_state.alerts       = alerts
#             st.session_state.actions      = actions

#             now = datetime.now().strftime("%H:%M:%S")
#             st.session_state.people_hist.append(pc)
#             st.session_state.time_hist.append(now)
#             if len(st.session_state.people_hist) > 100:
#                 st.session_state.people_hist.pop(0)
#                 st.session_state.time_hist.pop(0)
#             for a in alerts:
#                 st.session_state.alert_log.append({"time": now, "alert": a})
#             st.session_state.alert_log = st.session_state.alert_log[-60:]

#             # show processed frame (BGR → RGB)
#             frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb, use_container_width=True)
#         else:
#             frame_placeholder.markdown(
#                 "<div class='nosignal'>WAITING FOR CAMERA...</div>",
#                 unsafe_allow_html=True
#             )

#     # ── VIDEO UPLOAD MODE ─────────────────────────────────────────────────────
#     elif st.session_state.running and st.session_state.mode == "video":
#         uploaded_file = st.session_state.get("uploaded_file", None)
#         if uploaded_file is not None:
#             tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#             tfile.write(uploaded_file.read())
#             tfile.close()

#             cap = cv2.VideoCapture(tfile.name)
#             stframe = frame_placeholder.empty()
#             stop_flag = False

#             frame_count = 0
#             while cap.isOpened() and st.session_state.running:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 frame_count += 1
#                 if frame_count % 2 != 0:
#                     continue

#                 processed, pc, cc, bc, alerts, actions = process_frame(frame)

#                 st.session_state.people_count = pc
#                 st.session_state.car_count    = cc
#                 st.session_state.bus_count    = bc
#                 st.session_state.alerts       = alerts
#                 st.session_state.actions      = actions

#                 now = datetime.now().strftime("%H:%M:%S")
#                 st.session_state.people_hist.append(pc)
#                 st.session_state.time_hist.append(now)
#                 if len(st.session_state.people_hist) > 100:
#                     st.session_state.people_hist.pop(0)
#                     st.session_state.time_hist.pop(0)
#                 for a in alerts:
#                     st.session_state.alert_log.append({"time": now, "alert": a})
#                 st.session_state.alert_log = st.session_state.alert_log[-60:]

#                 frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#                 stframe.image(frame_rgb, use_container_width=True)

#             cap.release()
#             os.unlink(tfile.name)
#             st.session_state.running = False
#             st.rerun()
#         else:
#             frame_placeholder.markdown(
#                 "<div class='nosignal'>NO VIDEO FILE</div>",
#                 unsafe_allow_html=True
#             )

#     # ── OFFLINE ───────────────────────────────────────────────────────────────
#     else:
#         frame_placeholder.markdown(
#             "<div class='nosignal'>[ NO SIGNAL — START TRACKER ]</div>",
#             unsafe_allow_html=True
#         )

# # ── RIGHT PANEL ───────────────────────────────────────────────────────────────
# with col_right:
#     # Actions
#     st.markdown("<div class='sec-hdr'>SKELETON ACTIONS</div>", unsafe_allow_html=True)
#     actions = st.session_state.get("actions", [])
#     if actions:
#         ACTION_COLOR = {
#             "Normal":    ("#00ee88", "#0a1a10", "#00ee66"),
#             "Active":    ("#ffaa00", "#1a1000", "#ffaa00"),
#             "Fast Move": ("#ff4444", "#1a0d0d", "#ff2a2a"),
#         }
#         for act in actions:
#             pid    = act.get("id", "?")
#             action = act.get("action", "Normal")
#             fg, bg, border = ACTION_COLOR.get(action, ("#c8d8e8", "#0d1b2a", "#1e4060"))
#             st.markdown(f"""
#             <div style='background:{bg};border:1px solid {border}44;
#                         border-left:3px solid {border};border-radius:4px;
#                         padding:6px 12px;margin:3px 0;
#                         font-family:"Share Tech Mono",monospace;font-size:.82rem;color:{fg};'>
#                 🧍 <span style='color:#00cfff;font-weight:800;'>ID-{pid}</span>
#                 &nbsp;→&nbsp;{action.upper()}
#             </div>""", unsafe_allow_html=True)
#     else:
#         st.markdown(
#             "<div style='color:#2a5a3a;font-family:\"Share Tech Mono\",monospace;"
#             "font-size:.8rem;padding:4px 0;'>No actions detected</div>",
#             unsafe_allow_html=True
#         )

#     st.markdown("<br>", unsafe_allow_html=True)

#     # Alerts
#     st.markdown("<div class='sec-hdr'>ACTIVE ALERTS</div>", unsafe_allow_html=True)
#     alerts = st.session_state.get("alerts", [])
#     if alerts:
#         for a in alerts:
#             st.markdown(f"<div class='alert-badge'>⚠ {a}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(
#             "<div style='color:#2a6a3a;font-family:\"Share Tech Mono\",monospace;"
#             "font-size:.8rem;'>✓ No active alerts</div>",
#             unsafe_allow_html=True
#         )

# st.markdown("<br>", unsafe_allow_html=True)

# # ── CHART ─────────────────────────────────────────────────────────────────────
# if st.session_state.time_hist:
#     import plotly.graph_objects as go
#     st.markdown("<div class='sec-hdr'>ANALYTICS — PEOPLE COUNT</div>", unsafe_allow_html=True)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=st.session_state.time_hist,
#         y=st.session_state.people_hist,
#         mode="lines+markers",
#         line=dict(color="rgb(0,180,255)", width=2),
#         marker=dict(size=3),
#         fill="tozeroy",
#         fillcolor="rgba(0,180,255,0.08)",
#     ))
#     fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#         margin=dict(l=0, r=0, t=4, b=0), height=160, showlegend=False,
#         xaxis=dict(showgrid=False, color="#2a5a7a", tickfont_size=8, nticks=6),
#         yaxis=dict(showgrid=True, gridcolor="#0d1e2e", color="#2a5a7a",
#                    tickfont_size=8, rangemode="tozero"),
#     )
#     st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# # ── ALERT HISTORY ─────────────────────────────────────────────────────────────
# with st.expander(f"📜 Alert History ({len(st.session_state.alert_log)} entries)"):
#     if st.session_state.alert_log:
#         import pandas as pd
#         df = pd.DataFrame(st.session_state.alert_log[::-1])
#         st.dataframe(df, use_container_width=True, hide_index=True)
#     else:
#         st.info("Koi alert nahi abhi tak.")



# """
# Sentinel AI — app.py
# Deploy: Hugging Face Spaces (Streamlit)
# Run locally: streamlit run app.py
# """

# import streamlit as st
# import cv2
# import numpy as np
# import time
# import tempfile
# import os
# from datetime import datetime
# from ultralytics import YOLO
# import supervision as sv

# from analysis.behaviour import detect_behaviour, reset_history
# from analysis.suspicious import detect_suspicious
# from analysis.movement import detect_movement
# from analysis.pose_tasks import draw_skeleton

# # ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Sentinel AI",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🛡️",
# )

# # ── CSS ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');
# html, body, [class*="css"] {
#     font-family: 'Barlow Condensed', sans-serif;
#     background: #070b0f; color: #c8d8e8;
# }
# section[data-testid="stSidebar"] {
#     background: #0b1018 !important;
#     border-right: 1px solid #1a2a3a;
# }
# div[data-testid="metric-container"] {
#     background: #0d1b2a; border: 1px solid #1e3a52;
#     border-radius: 8px; padding: 14px 18px;
# }
# div[data-testid="metric-container"] label {
#     color: #5a8fa8 !important;
#     font-family: 'Share Tech Mono', monospace !important;
#     font-size: .72rem !important; letter-spacing: .12em; text-transform: uppercase;
# }
# div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
#     font-family: 'Barlow Condensed', sans-serif !important;
#     font-size: 2.4rem !important; font-weight: 800;
#     color: #00cfff !important; line-height: 1;
# }
# .sec-hdr {
#     font-family: 'Share Tech Mono', monospace; font-size: .75rem;
#     letter-spacing: .18em; text-transform: uppercase; color: #3a7a9c;
#     border-bottom: 1px solid #1a3a52; padding-bottom: 4px;
#     margin-bottom: 10px; margin-top: 4px;
# }
# .alert-badge {
#     display: block; background: #1a0d0d;
#     border: 1px solid #8b0000; border-left: 3px solid #ff2a2a;
#     border-radius: 4px; padding: 6px 12px; margin: 3px 0;
#     font-family: 'Share Tech Mono', monospace; font-size: .8rem; color: #ff6b6b;
# }
# .nosignal {
#     height: 360px; background: #080c10; border: 1px solid #1a3a52;
#     border-radius: 6px; display: flex; align-items: center;
#     justify-content: center; font-family: 'Share Tech Mono', monospace;
#     color: #2a5a7a; font-size: 1rem; letter-spacing: .2em;
# }
# .pill { display:inline-block; border-radius:20px; padding:3px 14px;
#         font-family:'Share Tech Mono',monospace; font-size:.75rem;
#         letter-spacing:.08em; margin-top:4px; }
# .pill-live  { background:#0a2a10; border:1px solid #00aa44; color:#00ee66; }
# .pill-video { background:#0a1a2a; border:1px solid #0066cc; color:#44aaff; }
# .pill-off   { background:#1a1a1a; border:1px solid #444;    color:#888;    }
# .stButton>button {
#     font-family: 'Share Tech Mono', monospace !important;
#     background: #0d1b2a !important; border: 1px solid #1e4060 !important;
#     color: #00cfff !important; border-radius: 4px !important; width: 100%;
# }
# .stButton>button:hover { background: #102030 !important; border-color: #00cfff !important; }
# </style>
# """, unsafe_allow_html=True)

# # ── SESSION STATE ─────────────────────────────────────────────────────────────
# defaults = {
#     "running": False, "mode": "none",
#     "people_hist": [], "time_hist": [], "alert_log": [],
#     "actions": [], "people_count": 0, "car_count": 0, "bus_count": 0,
#     "alerts": [], "seen_ids": set(), "five_min_count": 0,
#     "start_time": time.time(), "uploaded_file": None,
# }
# for k, v in defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ── LOAD MODEL ────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     return YOLO("yolov8n.pt")

# @st.cache_resource
# def load_tracker():
#     return sv.ByteTrack()

# model           = load_model()
# byte_tracker    = load_tracker()
# box_annotator   = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # ── PROCESS FRAME ─────────────────────────────────────────────────────────────
# def process_frame(frame: np.ndarray):
#     frame = cv2.resize(frame, (640, 480))

#     results    = model(frame, verbose=False)[0]
#     detections = sv.Detections.from_ultralytics(results)
#     detections = byte_tracker.update_with_detections(detections)

#     people_count = car_count = bus_count = 0
#     if detections.class_id is not None:
#         for cid in detections.class_id:
#             if cid == 0:   people_count += 1
#             elif cid == 2: car_count    += 1
#             elif cid == 5: bus_count    += 1

#     labels = []
#     if detections.tracker_id is not None:
#         for tid in detections.tracker_id:
#             labels.append(f"ID {tid}")
#             if tid not in st.session_state.seen_ids:
#                 st.session_state.seen_ids.add(tid)
#                 st.session_state.five_min_count += 1

#     actions = []
#     if detections.xyxy is not None and detections.class_id is not None:
#         for i, box in enumerate(detections.xyxy):
#             if detections.class_id[i] != 0 or i > 6:
#                 continue
#             x1, y1, x2, y2 = map(int, box)
#             pid = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
#             frame = draw_skeleton(frame, (x1, y1, x2, y2))
#             behaviour = detect_behaviour(pid, (x1, y1, x2, y2))
#             actions.append({"id": pid, "action": behaviour})
#             color = (0,255,255) if behaviour=="Normal" else (0,165,255) if behaviour=="Active" else (0,0,255)
#             cv2.putText(frame, behaviour, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

#     frame = box_annotator.annotate(frame, detections)
#     frame = label_annotator.annotate(frame, detections, labels)

#     alerts     = detect_suspicious(detections)
#     movement   = detect_movement(detections)
#     all_alerts = alerts + movement
#     if people_count > 5:
#         all_alerts.append("⚠ Crowd Detected")

#     cv2.rectangle(frame, (0,0), (220,36), (10,20,35), -1)
#     cv2.putText(frame, f"PEOPLE: {people_count}", (8,26),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,207,255), 2)

#     return frame, people_count, car_count, bus_count, all_alerts, actions


# def update_session(pc, cc, bc, alerts, actions):
#     st.session_state.people_count = pc
#     st.session_state.car_count    = cc
#     st.session_state.bus_count    = bc
#     st.session_state.alerts       = alerts
#     st.session_state.actions      = actions
#     now = datetime.now().strftime("%H:%M:%S")
#     st.session_state.people_hist.append(pc)
#     st.session_state.time_hist.append(now)
#     if len(st.session_state.people_hist) > 100:
#         st.session_state.people_hist.pop(0)
#         st.session_state.time_hist.pop(0)
#     for a in alerts:
#         st.session_state.alert_log.append({"time": now, "alert": a})
#     st.session_state.alert_log = st.session_state.alert_log[-60:]


# # ══════════════════════════════════════════════════════════════════════════════
# #  SIDEBAR
# # ══════════════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("""
#     <div style='text-align:center;padding:12px 0 20px'>
#       <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.9rem;
#                   font-weight:800;color:#00cfff;letter-spacing:.08em;'>SENTINEL AI</div>
#       <div style='font-family:"Share Tech Mono",monospace;font-size:.62rem;
#                   color:#3a6a8a;letter-spacing:.22em;margin-top:2px;'>
#                   SURVEILLANCE SYSTEM v3.0</div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("<div class='sec-hdr'>STATUS</div>", unsafe_allow_html=True)
#     if st.session_state.running:
#         pill_cls = "pill-live" if st.session_state.mode == "camera" else "pill-video"
#         pill_txt = "LIVE CAMERA" if st.session_state.mode == "camera" else "VIDEO RUNNING"
#         st.markdown(f"<span class='pill {pill_cls}'>● {pill_txt}</span>", unsafe_allow_html=True)
#     else:
#         st.markdown("<span class='pill pill-off'>○ OFFLINE</span>", unsafe_allow_html=True)

#     st.divider()

#     st.markdown("<div class='sec-hdr'>SELECT MODE</div>", unsafe_allow_html=True)
#     mode_sel = st.radio("", ["📷 Live Camera", "🎬 Upload Video"],
#                         label_visibility="collapsed")

#     st.divider()
#     st.markdown("<div class='sec-hdr'>CONTROLS</div>", unsafe_allow_html=True)

#     if mode_sel == "📷 Live Camera":
#         if not st.session_state.running:
#             if st.button("▶  START CAMERA"):
#                 st.session_state.running = True
#                 st.session_state.mode    = "camera"
#                 reset_history()
#                 st.session_state.seen_ids.clear()
#                 st.session_state.five_min_count = 0
#                 st.rerun()
#         else:
#             if st.button("■  STOP"):
#                 st.session_state.running = False
#                 st.session_state.mode    = "none"
#                 st.rerun()
#     else:
#         uploaded = st.file_uploader("Upload video",
#                                     type=["mp4","avi","mov","mkv"],
#                                     label_visibility="collapsed")
#         if uploaded:
#             st.session_state.uploaded_file = uploaded

#         if st.session_state.uploaded_file and not st.session_state.running:
#             if st.button("▶  PROCESS VIDEO"):
#                 st.session_state.running = True
#                 st.session_state.mode    = "video"
#                 reset_history()
#                 st.session_state.seen_ids.clear()
#                 st.session_state.five_min_count = 0
#                 st.rerun()
#         elif st.session_state.running and st.session_state.mode == "video":
#             if st.button("■  STOP"):
#                 st.session_state.running = False
#                 st.session_state.mode    = "none"
#                 st.rerun()

#     st.divider()
#     if st.button("🗑  Clear History"):
#         for k in ("people_hist","time_hist","alert_log"):
#             st.session_state[k].clear()
#         st.rerun()


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN LAYOUT — render karo pehle, data baad mein bharo
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <div style='margin-bottom:16px;'>
#   <span style='font-family:"Barlow Condensed",sans-serif;font-size:2rem;
#                font-weight:800;color:#fff;letter-spacing:.04em;'>
#     AI SURVEILLANCE DASHBOARD
#   </span>
# </div>
# """, unsafe_allow_html=True)

# # ── METRICS (placeholders) ────────────────────────────────────────────────────
# st.markdown("<div class='sec-hdr'>LIVE COUNTS</div>", unsafe_allow_html=True)
# m1, m2, m3, m4, m5 = st.columns(5)
# metric_people  = m1.empty()
# metric_total   = m2.empty()
# metric_cars    = m3.empty()
# metric_buses   = m4.empty()
# metric_alerts  = m5.empty()

# def render_metrics():
#     metric_people.metric("People Now",    st.session_state.people_count)
#     metric_total.metric("Total Seen",     st.session_state.five_min_count)
#     metric_cars.metric("Cars",            st.session_state.car_count)
#     metric_buses.metric("Buses",          st.session_state.bus_count)
#     metric_alerts.metric("Active Alerts", len(st.session_state.alerts))

# render_metrics()

# st.markdown("<br>", unsafe_allow_html=True)

# # ── FEED + RIGHT PANEL ────────────────────────────────────────────────────────
# st.markdown("<div class='sec-hdr'>FEED</div>", unsafe_allow_html=True)
# col_feed, col_right = st.columns([3, 2])

# with col_feed:
#     frame_placeholder = st.empty()

# with col_right:
#     actions_header = st.empty()
#     actions_body   = st.empty()
#     st.markdown("<br>", unsafe_allow_html=True)
#     alerts_header  = st.empty()
#     alerts_body    = st.empty()

# def render_right_panel():
#     actions = st.session_state.get("actions", [])
#     actions_header.markdown("<div class='sec-hdr'>SKELETON ACTIONS</div>", unsafe_allow_html=True)
#     ACTION_COLOR = {
#         "Normal":    ("#00ee88", "#0a1a10", "#00ee66"),
#         "Active":    ("#ffaa00", "#1a1000", "#ffaa00"),
#         "Fast Move": ("#ff4444", "#1a0d0d", "#ff2a2a"),
#     }
#     if actions:
#         html = ""
#         for act in actions:
#             pid    = act.get("id", "?")
#             action = act.get("action", "Normal")
#             fg, bg, border = ACTION_COLOR.get(action, ("#c8d8e8","#0d1b2a","#1e4060"))
#             html += f"""<div style='background:{bg};border:1px solid {border}44;
#                 border-left:3px solid {border};border-radius:4px;
#                 padding:6px 12px;margin:3px 0;
#                 font-family:"Share Tech Mono",monospace;font-size:.82rem;color:{fg};'>
#                 🧍 <span style='color:#00cfff;font-weight:800;'>ID-{pid}</span>
#                 &nbsp;→&nbsp;{action.upper()}</div>"""
#         actions_body.markdown(html, unsafe_allow_html=True)
#     else:
#         actions_body.markdown(
#             "<div style='color:#2a5a3a;font-family:\"Share Tech Mono\",monospace;"
#             "font-size:.8rem;padding:4px 0;'>No actions detected</div>",
#             unsafe_allow_html=True)

#     alerts_header.markdown("<div class='sec-hdr'>ACTIVE ALERTS</div>", unsafe_allow_html=True)
#     alerts = st.session_state.get("alerts", [])
#     if alerts:
#         html = "".join([f"<div class='alert-badge'>⚠ {a}</div>" for a in alerts])
#         alerts_body.markdown(html, unsafe_allow_html=True)
#     else:
#         alerts_body.markdown(
#             "<div style='color:#2a6a3a;font-family:\"Share Tech Mono\",monospace;"
#             "font-size:.8rem;'>✓ No active alerts</div>",
#             unsafe_allow_html=True)

# render_right_panel()

# # ── CHART PLACEHOLDER ─────────────────────────────────────────────────────────
# st.markdown("<br>", unsafe_allow_html=True)
# chart_placeholder = st.empty()

# # ── ALERT HISTORY ─────────────────────────────────────────────────────────────
# history_expander = st.expander(f"📜 Alert History ({len(st.session_state.alert_log)} entries)")
# with history_expander:
#     history_placeholder = st.empty()
#     if st.session_state.alert_log:
#         import pandas as pd
#         df = pd.DataFrame(st.session_state.alert_log[::-1])
#         history_placeholder.dataframe(df, use_container_width=True, hide_index=True)
#     else:
#         history_placeholder.info("Koi alert nahi abhi tak.")


# def render_chart():
#     if st.session_state.time_hist:
#         import plotly.graph_objects as go
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=st.session_state.time_hist, y=st.session_state.people_hist,
#             mode="lines+markers",
#             line=dict(color="rgb(0,180,255)", width=2),
#             marker=dict(size=3), fill="tozeroy",
#             fillcolor="rgba(0,180,255,0.08)",
#         ))
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#             margin=dict(l=0,r=0,t=4,b=0), height=160, showlegend=False,
#             xaxis=dict(showgrid=False, color="#2a5a7a", tickfont_size=8, nticks=6),
#             yaxis=dict(showgrid=True, gridcolor="#0d1e2e", color="#2a5a7a",
#                        tickfont_size=8, rangemode="tozero"),
#         )
#         chart_placeholder.plotly_chart(fig, use_container_width=True,
#                                        config={"displayModeBar": False})


# # ══════════════════════════════════════════════════════════════════════════════
# #  PROCESSING LOOPS
# # ══════════════════════════════════════════════════════════════════════════════

# # ── LIVE CAMERA ───────────────────────────────────────────────────────────────
# if st.session_state.running and st.session_state.mode == "camera":
#     # st.camera_input captures one frame per interaction
#     # We use a loop with rerun for continuous processing
#     camera_frame = frame_placeholder.camera_input(
#         "Camera", label_visibility="collapsed", key="cam_input"
#     )

#     if camera_frame is not None:
#         file_bytes = np.asarray(bytearray(camera_frame.read()), dtype=np.uint8)
#         frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#         processed, pc, cc, bc, alerts, actions = process_frame(frame_bgr)
#         update_session(pc, cc, bc, alerts, actions)

#         frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame_rgb, use_container_width=True)

#         render_metrics()
#         render_right_panel()
#         render_chart()

#         # auto rerun for next frame
#         time.sleep(0.05)
#         st.rerun()
#     else:
#         frame_placeholder.markdown(
#             "<div class='nosignal'>📷 ALLOW CAMERA PERMISSION → CLICK BELOW</div>",
#             unsafe_allow_html=True)

# # ── VIDEO UPLOAD ──────────────────────────────────────────────────────────────
# elif st.session_state.running and st.session_state.mode == "video":
#     uploaded_file = st.session_state.get("uploaded_file", None)

#     if uploaded_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_file.read())
#         tfile.close()

#         cap = cv2.VideoCapture(tfile.name)
#         frame_count = 0

#         while cap.isOpened() and st.session_state.running:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % 2 != 0:
#                 continue

#             processed, pc, cc, bc, alerts, actions = process_frame(frame)
#             update_session(pc, cc, bc, alerts, actions)

#             frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb, use_container_width=True)

#             render_metrics()
#             render_right_panel()
#             render_chart()

#         cap.release()
#         try:
#             os.unlink(tfile.name)
#         except:
#             pass

#         st.session_state.running = False
#         st.rerun()
#     else:
#         frame_placeholder.markdown(
#             "<div class='nosignal'>NO VIDEO FILE — Upload karo sidebar mein</div>",
#             unsafe_allow_html=True)

# # ── OFFLINE ───────────────────────────────────────────────────────────────────
# else:
#     frame_placeholder.markdown(
#         "<div class='nosignal'>[ NO SIGNAL — START TRACKER ]</div>",
#         unsafe_allow_html=True)

"""
Sentinel AI — app.py
Deploy: Hugging Face Spaces (Streamlit)
Run locally: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

from analysis.behaviour import detect_behaviour, reset_history
from analysis.suspicious import detect_suspicious
from analysis.movement import detect_movement
from analysis.pose_tasks import draw_skeleton

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif;
    background: #070b0f; color: #c8d8e8;
}
section[data-testid="stSidebar"] {
    background: #0b1018 !important;
    border-right: 1px solid #1a2a3a;
}
div[data-testid="metric-container"] {
    background: #0d1b2a; border: 1px solid #1e3a52;
    border-radius: 8px; padding: 14px 18px;
}
div[data-testid="metric-container"] label {
    color: #5a8fa8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .72rem !important; letter-spacing: .12em; text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2.4rem !important; font-weight: 800;
    color: #00cfff !important; line-height: 1;
}
.sec-hdr {
    font-family: 'Share Tech Mono', monospace; font-size: .75rem;
    letter-spacing: .18em; text-transform: uppercase; color: #3a7a9c;
    border-bottom: 1px solid #1a3a52; padding-bottom: 4px;
    margin-bottom: 10px; margin-top: 4px;
}
.alert-badge {
    display: block; background: #1a0d0d;
    border: 1px solid #8b0000; border-left: 3px solid #ff2a2a;
    border-radius: 4px; padding: 6px 12px; margin: 3px 0;
    font-family: 'Share Tech Mono', monospace; font-size: .8rem; color: #ff6b6b;
}
.nosignal {
    height: 360px; background: #080c10; border: 1px solid #1a3a52;
    border-radius: 6px; display: flex; align-items: center;
    justify-content: center; font-family: 'Share Tech Mono', monospace;
    color: #2a5a7a; font-size: 1rem; letter-spacing: .2em;
}
.pill { display:inline-block; border-radius:20px; padding:3px 14px;
        font-family:'Share Tech Mono',monospace; font-size:.75rem;
        letter-spacing:.08em; margin-top:4px; }
.pill-live  { background:#0a2a10; border:1px solid #00aa44; color:#00ee66; }
.pill-video { background:#0a1a2a; border:1px solid #0066cc; color:#44aaff; }
.pill-off   { background:#1a1a1a; border:1px solid #444;    color:#888;    }
.stButton>button {
    font-family: 'Share Tech Mono', monospace !important;
    background: #0d1b2a !important; border: 1px solid #1e4060 !important;
    color: #00cfff !important; border-radius: 4px !important; width: 100%;
}
.stButton>button:hover { background: #102030 !important; border-color: #00cfff !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "running": False, "mode": "none",
    "people_hist": [], "time_hist": [], "alert_log": [],
    "actions": [], "people_count": 0, "car_count": 0, "bus_count": 0,
    "alerts": [], "seen_ids": set(), "five_min_count": 0,
    "start_time": time.time(), "uploaded_file": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_tracker():
    return sv.ByteTrack()

model           = load_model()
byte_tracker    = load_tracker()
box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ── PROCESS FRAME ─────────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray):
    frame = cv2.resize(frame, (640, 480))

    results    = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    people_count = car_count = bus_count = 0
    if detections.class_id is not None:
        for cid in detections.class_id:
            if cid == 0:   people_count += 1
            elif cid == 2: car_count    += 1
            elif cid == 5: bus_count    += 1

    labels = []
    if detections.tracker_id is not None:
        for tid in detections.tracker_id:
            labels.append(f"ID {tid}")
            if tid not in st.session_state.seen_ids:
                st.session_state.seen_ids.add(tid)
                st.session_state.five_min_count += 1

    actions = []
    if detections.xyxy is not None and detections.class_id is not None:
        for i, box in enumerate(detections.xyxy):
            if detections.class_id[i] != 0 or i > 6:
                continue
            x1, y1, x2, y2 = map(int, box)
            pid = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
            frame = draw_skeleton(frame, (x1, y1, x2, y2))
            behaviour = detect_behaviour(pid, (x1, y1, x2, y2))
            actions.append({"id": pid, "action": behaviour})
            color = (0,255,255) if behaviour=="Normal" else (0,165,255) if behaviour=="Active" else (0,0,255)
            cv2.putText(frame, behaviour, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)

    alerts     = detect_suspicious(detections)
    movement   = detect_movement(detections)
    all_alerts = alerts + movement
    if people_count > 5:
        all_alerts.append("⚠ Crowd Detected")

    cv2.rectangle(frame, (0,0), (220,36), (10,20,35), -1)
    cv2.putText(frame, f"PEOPLE: {people_count}", (8,26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,207,255), 2)

    return frame, people_count, car_count, bus_count, all_alerts, actions


def update_session(pc, cc, bc, alerts, actions):
    st.session_state.people_count = pc
    st.session_state.car_count    = cc
    st.session_state.bus_count    = bc
    st.session_state.alerts       = alerts
    st.session_state.actions      = actions
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.people_hist.append(pc)
    st.session_state.time_hist.append(now)
    if len(st.session_state.people_hist) > 100:
        st.session_state.people_hist.pop(0)
        st.session_state.time_hist.pop(0)
    for a in alerts:
        st.session_state.alert_log.append({"time": now, "alert": a})
    st.session_state.alert_log = st.session_state.alert_log[-60:]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 20px'>
      <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.9rem;
                  font-weight:800;color:#00cfff;letter-spacing:.08em;'>SENTINEL AI</div>
      <div style='font-family:"Share Tech Mono",monospace;font-size:.62rem;
                  color:#3a6a8a;letter-spacing:.22em;margin-top:2px;'>
                  SURVEILLANCE SYSTEM v3.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>STATUS</div>", unsafe_allow_html=True)
    if st.session_state.running:
        pill_cls = "pill-live" if st.session_state.mode == "camera" else "pill-video"
        pill_txt = "LIVE CAMERA" if st.session_state.mode == "camera" else "VIDEO RUNNING"
        st.markdown(f"<span class='pill {pill_cls}'>● {pill_txt}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pill pill-off'>○ OFFLINE</span>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='sec-hdr'>SELECT MODE</div>", unsafe_allow_html=True)
    mode_sel = st.radio("", ["📷 Live Camera", "🎬 Upload Video"],
                        label_visibility="collapsed")

    st.divider()
    st.markdown("<div class='sec-hdr'>CONTROLS</div>", unsafe_allow_html=True)

    if mode_sel == "📷 Live Camera":
        if not st.session_state.running:
            if st.button("▶  START CAMERA"):
                st.session_state.running = True
                st.session_state.mode    = "camera"
                reset_history()
                st.session_state.seen_ids.clear()
                st.session_state.five_min_count = 0
                st.rerun()
        else:
            if st.button("■  STOP"):
                st.session_state.running = False
                st.session_state.mode    = "none"
                st.rerun()
    else:
        uploaded = st.file_uploader("Upload video",
                                    type=["mp4","avi","mov","mkv"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.uploaded_file = uploaded

        if st.session_state.running and st.session_state.mode == "video":
            if st.button("■  STOP"):
                st.session_state.running = False
                st.session_state.mode    = "none"
                st.session_state.uploaded_file = None
                st.rerun()
        elif uploaded is not None:
            if st.button("▶  PROCESS VIDEO"):
                st.session_state.running = True
                st.session_state.mode    = "video"
                st.session_state.uploaded_file = uploaded
                reset_history()
                st.session_state.seen_ids.clear()
                st.session_state.five_min_count = 0
                st.rerun()

    st.divider()
    if st.button("🗑  Clear History"):
        for k in ("people_hist","time_hist","alert_log"):
            st.session_state[k].clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT — render karo pehle, data baad mein bharo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:16px;'>
  <span style='font-family:"Barlow Condensed",sans-serif;font-size:2rem;
               font-weight:800;color:#fff;letter-spacing:.04em;'>
    AI SURVEILLANCE DASHBOARD
  </span>
</div>
""", unsafe_allow_html=True)

# ── METRICS (placeholders) ────────────────────────────────────────────────────
st.markdown("<div class='sec-hdr'>LIVE COUNTS</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
metric_people  = m1.empty()
metric_total   = m2.empty()
metric_cars    = m3.empty()
metric_buses   = m4.empty()
metric_alerts  = m5.empty()

def render_metrics():
    metric_people.metric("People Now",    st.session_state.people_count)
    metric_total.metric("Total Seen",     st.session_state.five_min_count)
    metric_cars.metric("Cars",            st.session_state.car_count)
    metric_buses.metric("Buses",          st.session_state.bus_count)
    metric_alerts.metric("Active Alerts", len(st.session_state.alerts))

render_metrics()

st.markdown("<br>", unsafe_allow_html=True)

# ── FEED + RIGHT PANEL ────────────────────────────────────────────────────────
st.markdown("<div class='sec-hdr'>FEED</div>", unsafe_allow_html=True)
col_feed, col_right = st.columns([3, 2])

with col_feed:
    frame_placeholder = st.empty()

with col_right:
    actions_header = st.empty()
    actions_body   = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    alerts_header  = st.empty()
    alerts_body    = st.empty()

def render_right_panel():
    actions = st.session_state.get("actions", [])
    actions_header.markdown("<div class='sec-hdr'>SKELETON ACTIONS</div>", unsafe_allow_html=True)
    ACTION_COLOR = {
        "Normal":    ("#00ee88", "#0a1a10", "#00ee66"),
        "Active":    ("#ffaa00", "#1a1000", "#ffaa00"),
        "Fast Move": ("#ff4444", "#1a0d0d", "#ff2a2a"),
    }
    if actions:
        html = ""
        for act in actions:
            pid    = act.get("id", "?")
            action = act.get("action", "Normal")
            fg, bg, border = ACTION_COLOR.get(action, ("#c8d8e8","#0d1b2a","#1e4060"))
            html += f"""<div style='background:{bg};border:1px solid {border}44;
                border-left:3px solid {border};border-radius:4px;
                padding:6px 12px;margin:3px 0;
                font-family:"Share Tech Mono",monospace;font-size:.82rem;color:{fg};'>
                🧍 <span style='color:#00cfff;font-weight:800;'>ID-{pid}</span>
                &nbsp;→&nbsp;{action.upper()}</div>"""
        actions_body.markdown(html, unsafe_allow_html=True)
    else:
        actions_body.markdown(
            "<div style='color:#2a5a3a;font-family:\"Share Tech Mono\",monospace;"
            "font-size:.8rem;padding:4px 0;'>No actions detected</div>",
            unsafe_allow_html=True)

    alerts_header.markdown("<div class='sec-hdr'>ACTIVE ALERTS</div>", unsafe_allow_html=True)
    alerts = st.session_state.get("alerts", [])
    if alerts:
        html = "".join([f"<div class='alert-badge'>⚠ {a}</div>" for a in alerts])
        alerts_body.markdown(html, unsafe_allow_html=True)
    else:
        alerts_body.markdown(
            "<div style='color:#2a6a3a;font-family:\"Share Tech Mono\",monospace;"
            "font-size:.8rem;'>✓ No active alerts</div>",
            unsafe_allow_html=True)

render_right_panel()

# ── CHART PLACEHOLDER ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
chart_placeholder = st.empty()

# ── ALERT HISTORY ─────────────────────────────────────────────────────────────
history_expander = st.expander(f"📜 Alert History ({len(st.session_state.alert_log)} entries)")
with history_expander:
    history_placeholder = st.empty()
    if st.session_state.alert_log:
        import pandas as pd
        df = pd.DataFrame(st.session_state.alert_log[::-1])
        history_placeholder.dataframe(df, use_container_width=True, hide_index=True)
    else:
        history_placeholder.info("Koi alert nahi abhi tak.")


def render_chart():
    if st.session_state.time_hist:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.time_hist, y=st.session_state.people_hist,
            mode="lines+markers",
            line=dict(color="rgb(0,180,255)", width=2),
            marker=dict(size=3), fill="tozeroy",
            fillcolor="rgba(0,180,255,0.08)",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=4,b=0), height=160, showlegend=False,
            xaxis=dict(showgrid=False, color="#2a5a7a", tickfont_size=8, nticks=6),
            yaxis=dict(showgrid=True, gridcolor="#0d1e2e", color="#2a5a7a",
                       tickfont_size=8, rangemode="tozero"),
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True,
                                       config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSING LOOPS
# ══════════════════════════════════════════════════════════════════════════════

# ── LIVE CAMERA ───────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.mode == "camera":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        frame_placeholder.markdown(
            "<div class='nosignal'>❌ CAMERA NOT FOUND — Check connection</div>",
            unsafe_allow_html=True)
    else:
        frame_count = 0
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            processed, pc, cc, bc, alerts, actions = process_frame(frame)
            update_session(pc, cc, bc, alerts, actions)

            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

            render_metrics()
            render_right_panel()
            render_chart()

        cap.release()
        st.session_state.running = False
        st.rerun()

# ── VIDEO UPLOAD ──────────────────────────────────────────────────────────────
elif st.session_state.running and st.session_state.mode == "video":
    uploaded_file = st.session_state.get("uploaded_file", None)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0

        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            processed, pc, cc, bc, alerts, actions = process_frame(frame)
            update_session(pc, cc, bc, alerts, actions)

            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

            render_metrics()
            render_right_panel()
            render_chart()

        cap.release()
        try:
            os.unlink(tfile.name)
        except:
            pass

        st.session_state.running = False
        st.rerun()
    else:
        frame_placeholder.markdown(
            "<div class='nosignal'>NO VIDEO FILE — Upload karo sidebar mein</div>",
            unsafe_allow_html=True)

# ── OFFLINE ───────────────────────────────────────────────────────────────────
else:
    frame_placeholder.markdown(
        "<div class='nosignal'>[ NO SIGNAL — START TRACKER ]</div>",
        unsafe_allow_html=True)