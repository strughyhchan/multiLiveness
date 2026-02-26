import math
import os
import random
import sys
import time
import threading

import customtkinter as ctk
import cv2
import numpy as np

# Prevent matplotlib cache permission warnings triggered by mediapipe import.
_app_dir = os.path.dirname(os.path.abspath(__file__))
_mpl_cache_dir = os.path.join(_app_dir, ".mplconfig")
os.makedirs(_mpl_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache_dir)

import mediapipe as mp
from deepface import DeepFace
from PIL import Image


class AntiSpoofingApp(ctk.CTk):
    """Face anti-spoofing GUI using 3 Core Defense Engines."""

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Ultimate Face Anti-Spoofing Test Environment")
        self.geometry("960x720")
        self.minsize(800, 600)

        self.cap = None
        self.camera_index = None
        self.camera_backend = None
        self.latest_frame_bgr = None
        self.video_loop_running = False

        self.recording_active = False
        self.challenge_end_time = 0.0
        self.current_challenge = None
        
        # Data collection for defenses
        self._reset_data()
        
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=30, qualityLevel=0.3, minDistance=7, blockSize=7)

        self.face_mesh = None
        self.face_init_error = None
        self._init_face_mesh()

        # DeepFace: registered user image path (mock database)
        self.registered_face_path = os.path.join(_app_dir, "user_db", "IMG_3868.jpg")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._try_initialize_camera()

    def _reset_data(self):
        """Prepare fresh data arrays for the next 3-second cycle."""
        self.collected_data = {
            "lip_dists": [],
            "jaw_dists": [],
            "nose_positions": [],
            "forehead_positions": [],
            "bg_motion_vectors": [],
            "nose_motion_vectors": [],
            "forehead_motion_vectors": [],
            "mission_metrics": []
        }
        self.old_gray = None
        self.p0 = None
        # Debug visualization state
        self.debug_overlay = None
        self.bg_trail_mask = None
        self.nose_trail = []
        self.current_nose_speed = 0.0
        self.current_bg_speed = 0.0
        # Smart frame capture for DeepFace
        self.frame_start = None   # Captured at T=0
        self.frame_action = None  # Captured at T=1.5s
        self.challenge_midpoint = 0.0

    def _build_ui(self):
        """Create main video area, test button, and status/result labels."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self,
            text="Starting camera...",
            width=900,
            height=540,
            corner_radius=12,
            anchor="center",
        )
        self.video_label.grid(row=0, column=0, padx=24, pady=(24, 14), sticky="nsew")

        self.start_button = ctk.CTkButton(
            self,
            text="Start Ultimate Test",
            command=self.start_liveness_test,
            height=42,
            state="disabled",
        )
        self.start_button.grid(row=1, column=0, padx=24, pady=(0, 10))

        self.status_label = ctk.CTkLabel(self, text="Status: Initializing...")
        self.status_label.grid(row=2, column=0, padx=24, pady=(0, 8))

        self.result_label = ctk.CTkLabel(self, text="Result: -")
        self.result_label.grid(row=3, column=0, padx=24, pady=(0, 20))

    def _init_face_mesh(self):
        """Initialize MediaPipe Face Mesh."""
        if not hasattr(mp, "solutions"):
            self.face_mesh = None
            self.face_init_error = "Face Mesh API unavailable."
            return

        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            self.face_mesh = None
            self.face_init_error = str(exc)

    def _try_initialize_camera(self):
        """Retry camera initialization to handle permission timing."""
        if self.cap is None or not self.cap.isOpened():
            self.cap, self.camera_index, self.camera_backend = self._init_camera()

        if self.cap is not None and self.cap.isOpened():
            if self.face_mesh is None:
                self.status_label.configure(text=f"Status: Setup error. {self.face_init_error}")
                self.start_button.configure(state="disabled")
            else:
                self.status_label.configure(
                    text=f"Status: Webcam ready (index {self.camera_index}, {self.camera_backend})."
                )
                self.start_button.configure(state="normal")
            if not self.video_loop_running:
                self.video_loop_running = True
                self.update_video_frame()
            return

        self.status_label.configure(
            text="Status: Webcam not available yet. Allow camera permission, then wait for auto-retry..."
        )
        self.after(1200, self._try_initialize_camera)

    def _init_camera(self):
        """Try available camera indices with platform-appropriate backend."""
        candidate_indices = range(5)
        backend_candidates = [("Default", None)]

        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backend_candidates.insert(0, ("AVFoundation", cv2.CAP_AVFOUNDATION))

        for backend_name, backend_code in backend_candidates:
            for idx in candidate_indices:
                if backend_code is None:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend_code)

                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue

                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, idx, backend_name

                cap.release()

        return None, None, None

    def update_video_frame(self):
        """Read webcam frame, convert BGR to RGB, and refresh the preview.
        When recording is active, overlay debug trails and velocity HUD."""
        if self.cap is None or not self.cap.isOpened():
            return

        ok, frame_bgr = self.cap.read()
        if ok:
            self.latest_frame_bgr = frame_bgr
            display_frame = frame_bgr.copy()

            # Overlay debug visuals during active recording
            if self.recording_active and self.debug_overlay is not None:
                # Blend the persistent trail mask onto the display frame
                if self.bg_trail_mask is not None:
                    trail_gray = cv2.cvtColor(self.bg_trail_mask, cv2.COLOR_BGR2GRAY)
                    _, trail_bin = cv2.threshold(trail_gray, 1, 255, cv2.THRESH_BINARY)
                    trail_region = cv2.bitwise_and(self.bg_trail_mask, self.bg_trail_mask, mask=trail_bin)
                    inv_mask = cv2.bitwise_not(trail_bin)
                    bg = cv2.bitwise_and(display_frame, display_frame, mask=inv_mask)
                    display_frame = cv2.add(bg, trail_region)

                # Draw current BG tracking points (green circles)
                if self.p0 is not None:
                    for pt in self.p0.reshape(-1, 2):
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

                # Draw nose trail (red polyline)
                if len(self.nose_trail) >= 2:
                    pts = np.array(self.nose_trail, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(display_frame, [pts], False, (0, 0, 255), 2)
                    cv2.circle(display_frame, self.nose_trail[-1], 5, (0, 0, 255), -1)

                # Draw forehead dot (blue) for display detection debug
                if len(self.collected_data["forehead_positions"]) > 0:
                    fh = self.collected_data["forehead_positions"][-1]
                    cv2.circle(display_frame, fh, 6, (255, 0, 0), -1)
                    cv2.putText(display_frame, "FH", (fh[0]+8, fh[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # Velocity HUD (top-left corner)
                cv2.rectangle(display_frame, (10, 10), (320, 70), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Nose Speed: {self.current_nose_speed:.1f} px/f",
                            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"BG Speed:   {self.current_bg_speed:.1f} px/f",
                            (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            target_w = max(self.video_label.winfo_width(), 1)
            target_h = max(self.video_label.winfo_height(), 1)

            tk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(target_w, target_h),
            )
            self.video_label.configure(text="", image=tk_image)
            self.video_label.image = tk_image

        self.after(15, self.update_video_frame)

    def start_liveness_test(self):
        """Start a 3-second data collection loop for the 3 Defense Engines."""
        if self.recording_active:
            return

        if self.face_mesh is None:
            self.status_label.configure(text=f"Status: Face Mesh init failed.")
            self.result_label.configure(text="Result: SETUP ERROR (face mesh unavailable)")
            return

        if self.latest_frame_bgr is None:
            self.status_label.configure(text="Status: No frame available yet.")
            return

        challenges = ["OPEN MOUTH", "SMILE"]
        self.current_challenge = random.choice(challenges)
        
        self.recording_active = True
        self.challenge_end_time = time.monotonic() + 3.0
        self.challenge_midpoint = time.monotonic() + 1.5
        self._reset_data()

        # Capture frame at T=0 for DeepFace consistency check (will crop later inside loop)
        self.frame_start_crop = None

        self.start_button.configure(state="disabled")
        self.status_label.configure(text=f"MISSION: {self.current_challenge}! (3 seconds...)")
        self.result_label.configure(text="Result: Collecting data...")
        self.test_finished = False

        self._collection_loop()

    def _landmark_to_pixel(self, landmark, width, height):
        """Convert normalized landmark to clamped pixel coordinates."""
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return x, y

    def _collection_loop(self):
        """Execute each frame during the 3 second challenge to gather data."""
        if not self.recording_active:
            return

        frame = self.latest_frame_bgr
        if frame is not None:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_result = self.face_mesh.process(rgb)

            if mesh_result.multi_face_landmarks:
                landmarks = mesh_result.multi_face_landmarks[0].landmark
                
                # Core Points
                nose = self._landmark_to_pixel(landmarks[1], w, h)
                upper_lip = self._landmark_to_pixel(landmarks[13], w, h)
                lower_lip = self._landmark_to_pixel(landmarks[14], w, h)
                chin = self._landmark_to_pixel(landmarks[152], w, h)
                
                # Save points for Muscle Correlation and Jitter Analysis
                lip_dist = math.hypot(upper_lip[0] - lower_lip[0], upper_lip[1] - lower_lip[1])
                jaw_dist = math.hypot(nose[0] - chin[0], nose[1] - chin[1])
                
                self.collected_data["lip_dists"].append(lip_dist)
                self.collected_data["jaw_dists"].append(jaw_dist)
                self.collected_data["nose_positions"].append(nose)
                
                # Forehead (landmark 10) - stable point for display detection
                forehead = self._landmark_to_pixel(landmarks[10], w, h)
                self.collected_data["forehead_positions"].append(forehead)
                
                # Active Liveness Mission Metrics
                if self.current_challenge == "SMILE":
                    left_lip = self._landmark_to_pixel(landmarks[61], w, h)
                    right_lip = self._landmark_to_pixel(landmarks[291], w, h)
                    self.collected_data["mission_metrics"].append(math.hypot(left_lip[0] - right_lip[0], left_lip[1] - right_lip[1]))
                elif self.current_challenge == "OPEN MOUTH":
                    self.collected_data["mission_metrics"].append(lip_dist)

                # Track nose position for debug red trail
                self.nose_trail.append(nose)

                # Standard frame capture
                if self.frame_start is None:
                    self.frame_start = frame.copy()

                # Optical Flow for Background tracking (Video Replay Defense)
                if self.old_gray is None:
                    # Recompute for background mask just in case
                    all_pts = np.array([self._landmark_to_pixel(lm, w, h) for lm in landmarks])
                    fx, fy, fw, fh = cv2.boundingRect(all_pts)
                    
                    # Inner box = face bounding box (exclude face itself)
                    inner_x1 = max(0, fx)
                    inner_y1 = max(0, fy)
                    inner_x2 = min(w, fx + fw)
                    inner_y2 = min(h, fy + fh)
                    
                    # Outer box = face bbox expanded by 50% in all directions
                    expand_w = int(fw * 0.5)
                    expand_h = int(fh * 0.5)
                    outer_x1 = max(0, fx - expand_w)
                    outer_y1 = max(0, fy - expand_h)
                    outer_x2 = min(w, fx + fw + expand_w)
                    outer_y2 = min(h, fy + fh + expand_h)
                    
                    # Create donut mask: white in outer box, black in inner box
                    mask = np.zeros_like(gray)
                    cv2.rectangle(mask, (outer_x1, outer_y1), (outer_x2, outer_y2), 255, -1)
                    cv2.rectangle(mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 0, -1)
                    
                    self.p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                    self.old_gray = gray
                    # Initialize the debug trail mask (same size as frame, black)
                    self.bg_trail_mask = np.zeros_like(frame)
                    self.debug_overlay = frame.copy()
                elif self.p0 is not None and len(self.p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **self.lk_params)
                    
                    if p1 is not None and st is not None:
                        st = st.flatten()
                        good_new = p1[st == 1].reshape(-1, 2)
                        good_old = self.p0[st == 1].reshape(-1, 2)
                        
                        if len(good_new) > 0:
                            # Use median to avoid outliers from moving hair/hands in background
                            bg_motion = np.median(good_new - good_old, axis=0)  # shape (2,)
                            self.collected_data["bg_motion_vectors"].append(bg_motion)
                            self.current_bg_speed = float(np.linalg.norm(bg_motion))

                            # Draw green trails on persistent mask
                            for new_pt, old_pt in zip(good_new, good_old):
                                a = (int(new_pt[0]), int(new_pt[1]))
                                b = (int(old_pt[0]), int(old_pt[1]))
                                self.bg_trail_mask = cv2.line(self.bg_trail_mask, a, b, (0, 255, 0), 1)
                        else:
                            self.collected_data["bg_motion_vectors"].append(np.array([0.0, 0.0]))
                            self.current_bg_speed = 0.0
                            
                        self.p0 = good_new.reshape(-1, 1, 2)
                    else:
                        self.collected_data["bg_motion_vectors"].append(np.array([0.0, 0.0]))
                        self.current_bg_speed = 0.0
                        
                    self.old_gray = gray
                    self.debug_overlay = frame.copy()
                    
                    # Add corresponding nose motion vector and update speed HUD
                    if len(self.collected_data["nose_positions"]) >= 2:
                        prev_nose = np.array(self.collected_data["nose_positions"][-2])
                        curr_nose = np.array(self.collected_data["nose_positions"][-1])
                        nose_vec = curr_nose - prev_nose
                        self.collected_data["nose_motion_vectors"].append(nose_vec)
                        self.current_nose_speed = float(np.linalg.norm(nose_vec))
                    else:
                        self.collected_data["nose_motion_vectors"].append(np.array([0.0, 0.0]))
                        self.current_nose_speed = 0.0
                    
                    # Add forehead motion vector (stable during expressions)
                    if len(self.collected_data["forehead_positions"]) >= 2:
                        prev_fh = np.array(self.collected_data["forehead_positions"][-2])
                        curr_fh = np.array(self.collected_data["forehead_positions"][-1])
                        self.collected_data["forehead_motion_vectors"].append(curr_fh - prev_fh)
                    else:
                        self.collected_data["forehead_motion_vectors"].append(np.array([0.0, 0.0]))

                    # Speed HUD logic ends here

        # Capture frame at midpoint (T=1.5s) for DeepFace consistency check
        if (self.frame_action is None
                and frame is not None
                and time.monotonic() >= self.challenge_midpoint):
            self.frame_action = frame.copy()

        if time.monotonic() < self.challenge_end_time:
            self.after(50, self._collection_loop)
        else:
            self._evaluate_liveness()

    def _evaluate_liveness(self):
        """Evaluate Liveness Rules after 3 seconds of data collection."""
        result = {"gate1_passed": False, "gate2_passed": False, "gate3_passed": False, "error": None}
        data = self.collected_data

        if len(data["nose_positions"]) < 10:
            self._finalize_liveness_test("Result: SPOOF DETECTED (Insufficient Data)")
            return

        self.status_label.configure(text="Status: Checking Liveness...")

        # -----------------------------------------------------------------
        # Defense 1: Muscle Correlation (Defeats Paper Masks)
        # -----------------------------------------------------------------
        lip_dists = np.array(data["lip_dists"])
        jaw_dists = np.array(data["jaw_dists"])
        lip_range = np.max(lip_dists) - np.min(lip_dists)
        jaw_variance = np.var(jaw_dists)
        
        if lip_range > 15 and jaw_variance < 1.0:
            self._finalize_liveness_test("Result: SPOOF DETECTED (Paper Mask)")
            return

        # -----------------------------------------------------------------
        # Defense 2: Spatiotemporal Jitter (Defeats Deepfakes)
        # -----------------------------------------------------------------
        nose_positions = np.array(data["nose_positions"])
        if len(nose_positions) >= 3:
            velocities = np.linalg.norm(np.diff(nose_positions, axis=0), axis=1)
            accelerations = np.abs(np.diff(velocities))
            glitch_count = np.sum(accelerations > 20.0)
            if glitch_count > 2:
                self._finalize_liveness_test("Result: SPOOF DETECTED (Deepfake Jitter)")
                return

        # -----------------------------------------------------------------
        # Defense 3: Background-Foreground Locking (Defeats Video Replays)
        # Use FOREHEAD motion (stable during expressions) instead of nose tip
        # -----------------------------------------------------------------
        bg_motions = data["bg_motion_vectors"]
        face_motions = data["forehead_motion_vectors"]  # Forehead = stable reference
        min_len = min(len(bg_motions), len(face_motions))
        if min_len > 3:
            spoof_frames = 0
            evaluated_frames = 0
            for i in range(min_len):
                face_vec = np.array(face_motions[i])
                bg_vec = np.array(bg_motions[i])
                face_mag = float(np.linalg.norm(face_vec))
                bg_mag = float(np.linalg.norm(bg_vec))
                if face_mag > 1.0:
                    evaluated_frames += 1
                    
                    dot_product = np.dot(face_vec, bg_vec)
                    cos_sim = dot_product / (face_mag * bg_mag) if (face_mag * bg_mag) != 0 else 0
                    mag_ratio = min(face_mag, bg_mag) / max(face_mag, bg_mag) if max(face_mag, bg_mag) > 0 else 0
                    
                    if cos_sim > 0.80 and mag_ratio > 0.7:
                        spoof_frames += 1
                        
            if evaluated_frames > 3 and (spoof_frames / evaluated_frames) > 0.35:
                self._finalize_liveness_test(
                    f"Result: SPOOF DETECTED (Background Locked - {spoof_frames}/{evaluated_frames} frames)"
                )
                return

        # -----------------------------------------------------------------
        # Evaluate Random Mission
        # -----------------------------------------------------------------
        mission_passed = False
        metrics = data["mission_metrics"]
        if len(metrics) > 0:
            initial = metrics[0]
            if self.current_challenge == "SMILE":
                if np.max(metrics) > initial * 1.2:
                    mission_passed = True
            elif self.current_challenge == "TURN HEAD RIGHT":
                if np.max(metrics) > initial + 0.15: 
                    mission_passed = True
            elif self.current_challenge == "OPEN MOUTH":
                if np.max(metrics) > initial * 1.5:
                    mission_passed = True

        if mission_passed:
            # Gate 1 PASSED -> Proceed to Gate 2 & 3 (DeepFace CNN) in a thread.
            self.status_label.configure(text="Status: Processing CNN... (Identity Check)")
            self.result_label.configure(text="Result: Liveness OK. Verifying identity...")
            self._verification_result = None
            thread = threading.Thread(target=self._run_deepface_verification, daemon=True)
            thread.start()
            self._poll_verification_result()
        else:
            self._finalize_liveness_test("Result: SPOOF DETECTED (Mission Failed)")

    def _run_deepface_verification(self):
        """Gate 2 & 3: Run DeepFace verification in a background thread.
        Gate 2: Consistency check (start frame vs action frame).
        Gate 3: Identity check (start frame vs registered user)."""
        result = {"gate2_passed": False, "gate3_passed": False, "error": None}

        try:
            t_total = time.monotonic()

            # --- Gate 2: Consistency (Was the face swapped mid-test?) ---
            if self.frame_start is not None and self.frame_action is not None:
                # Convert BGR -> RGB for DeepFace
                start_rgb = cv2.cvtColor(self.frame_start, cv2.COLOR_BGR2RGB)
                action_rgb = cv2.cvtColor(self.frame_action, cv2.COLOR_BGR2RGB)
                
                t0 = time.monotonic()
                consistency = DeepFace.verify(
                    start_rgb, action_rgb,
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=False
                )
                print(f"[TIMING] Gate 2 (consistency): {time.monotonic() - t0:.2f}s")
                result["gate2_passed"] = consistency.get("verified", False)
            else:
                # If midpoint frame wasn't captured, skip consistency (pass)
                result["gate2_passed"] = True

            if not result["gate2_passed"]:
                self.after(0, lambda: self._on_verification_complete(result))
                return

            # --- Gate 3: Identity (Is this the registered user?) ---
            db_path = os.path.join(_app_dir, "user_db")
            if os.path.isdir(db_path) and len(os.listdir(db_path)) > 0 and self.frame_start is not None:
                start_rgb = cv2.cvtColor(self.frame_start, cv2.COLOR_BGR2RGB)
                
                is_match = False
                for filename in os.listdir(db_path):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    img_path = os.path.join(db_path, filename)
                    try:
                        # Load image manually to prevent DeepFace internal file path parsing bugs
                        db_img = cv2.imread(img_path)
                        if db_img is None:
                            continue
                        db_img_rgb = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
                        
                        t1 = time.monotonic()
                        # Crucial Fix:
                        # The DB image may be a selfie where the OpenCV detector completely fails to find a bounding box.
                        # If we use enforce_detection=True (or default), it throws "Exception while processing img2_path".
                        # So we MUST set enforce_detection=False here to tell DeepFace: "Just use whatever pixels I gave you".
                        identity = DeepFace.verify(
                            start_rgb, db_img_rgb,
                            model_name="ArcFace",
                            detector_backend="opencv",
                            distance_metric="cosine",
                            enforce_detection=False
                        )
                        print(f"[TIMING] Gate 3 verify '{filename}': {time.monotonic() - t1:.2f}s, distance: {identity.get('distance')}")
                        
                        # ArcFace default cosine threshold is around 0.68.
                        distance = identity.get("distance", 1.0)
                        if identity.get("verified", False) or distance < 0.70:
                            is_match = True
                            break # Found a match, no need to check other photos
                    except Exception as e:
                        print(f"[DEBUG] Gate 3 Error on {filename}: {e}")
                        pass # Ignore errors for individual files
                        
                result["gate3_passed"] = is_match
            else:
                # No registered face DB -> skip identity check (pass)
                result["gate3_passed"] = True
                result["error"] = "no_db"

            print(f"[TIMING] Total verification: {time.monotonic() - t_total:.2f}s")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            result["error"] = str(exc)
            # If DeepFace fails (no face detected, model error, etc.)
            # default to failing safely
            result["gate2_passed"] = result.get("gate2_passed", False)
            result["gate3_passed"] = False

        print(f"[DEBUG] Thread done. Result: {result}")
        self._verification_result = result

    def _poll_verification_result(self):
        """Poll for verification result from background thread."""
        if self._verification_result is not None:
            print(f"[DEBUG] Poll picked up result: {self._verification_result}")
            self._on_verification_complete(self._verification_result)
        else:
            self.after(100, self._poll_verification_result)

    def _on_verification_complete(self, result):
        """Handle DeepFace verification result on the main thread."""
        print(f"[DEBUG] _on_verification_complete CALLED with: {result}")
        if not result["gate2_passed"]:
            self._finalize_liveness_test(
                "Result: SPOOF DETECTED (Face Swapped during test!)"
            )
            return

        if not result["gate3_passed"]:
            if result.get("error") and result["error"] != "no_db":
                self._finalize_liveness_test(
                    f"Result: VERIFICATION ERROR ({result['error'][:80]})"
                )
            else:
                self._finalize_liveness_test(
                    "Result: ACCESS DENIED (Unknown Person)"
                )
            return

        if result.get("error") == "no_db":
            self._finalize_liveness_test(
                "Result: REAL FACE (Verified) — No user DB, identity check skipped"
            )
        else:
            self._finalize_liveness_test(
                "Result: ACCESS GRANTED! (Real Face & Verified User)"
            )

    def _finalize_liveness_test(self, result_text):
        """End the challenge phase and display result."""
        if getattr(self, "test_finished", False):
            print(f"[DEBUG] _finalize_liveness_test BLOCKED (Test already finished): {result_text}")
            return
            
        print(f"[DEBUG] _finalize_liveness_test: {result_text}")
        self.recording_active = False
        self.test_finished = True
        self.start_button.configure(state="normal")
        self.status_label.configure(text="Status: Test complete.")
        self.result_label.configure(text=result_text)

    def on_close(self):
        """Release camera and MediaPipe resources on app exit."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.face_mesh is not None:
            self.face_mesh.close()
        self.destroy()


if __name__ == "__main__":
    app = AntiSpoofingApp()
    app.mainloop()
