import os
import re
import tempfile
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionXI Football Analyzer",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .hero {
            padding: 28px;
            border-radius: 18px;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #0f172a, #1e293b);
        }

        .hero h1 {
            color: white;
            margin: 0;
        }

        .hero p {
            color: #cbd5e1;
            margin-top: 8px;
        }

        .card {
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>⚽ VisionXI Football Analyzer</h1>
        <p>
            AI-powered football video analysis with player tracking,
            ball detection and possession estimation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


model = load_model()


# ============================================================
# PLAYER DATABASE
# ============================================================

ROSTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "suriname_player_numbers_2026.csv"
)

PLAYER_IMAGE_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Players"
)


@st.cache_data
def load_roster():

    if not os.path.exists(ROSTER_PATH):
        return pd.DataFrame()

    df = pd.read_csv(
        ROSTER_PATH,
        dtype=str
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    for column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


roster = load_roster()


# ============================================================
# PLAYER IDENTIFICATION HELPERS
# ============================================================

def normalize_player_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_player_record(roster_df, player_name):
    if roster_df.empty or not player_name:
        return None

    name_column = "player_name" if "player_name" in roster_df.columns else "name"
    matches = roster_df[roster_df[name_column].str.strip() == str(player_name).strip()]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_roster_field(record, candidate_columns):
    """
    Return the first non-empty value found in `record` for any of the
    given candidate column names. Roster CSVs vary in how they name
    things (e.g. 'shirt_number' vs 'number', 'position' vs 'pos'), so
    this checks a few common spellings instead of assuming one.
    """
    if not record:
        return None

    for column in candidate_columns:
        if column in record:
            value = str(record[column]).strip()
            if value:
                return value

    return None


def find_player_image(player_name):
    if not player_name or not os.path.isdir(PLAYER_IMAGE_DIR):
        return None

    target = normalize_player_name(player_name)
    for filename in os.listdir(PLAYER_IMAGE_DIR):
        stem, _ = os.path.splitext(filename)
        if normalize_player_name(stem) == target:
            return os.path.join(PLAYER_IMAGE_DIR, filename)
    return None


@st.cache_resource
def load_reference_features(image_path):
    """Prepare SIFT features from the selected player's database photo."""
    if not image_path or not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)

    sift = cv2.SIFT_create(nfeatures=500)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 8:
        return None

    return descriptors


def player_visual_match(frame, box, reference_descriptors, min_good_matches=8):
    """Compare a YOLO person crop with the selected player's reference photo."""
    if reference_descriptors is None:
        return False, 0

    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    if x2 <= x1 or y2 <= y1:
        return False, 0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0

    # The upper part contains the face/shirt and is more useful than the pitch.
    crop = crop[:max(1, int(crop.shape[0] * 0.80)), :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)

    sift = cv2.SIFT_create(nfeatures=500)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 4:
        return False, 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    try:
        pairs = matcher.knnMatch(reference_descriptors, descriptors, k=2)
    except cv2.error:
        return False, 0

    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.72 * n.distance:
                good.append(m)

    return len(good) >= min_good_matches, len(good)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")

confidence = st.sidebar.slider(
    "Detection confidence",
    0.10,
    0.90,
    0.30,
    0.05
)

frame_skip = st.sidebar.slider(
    "Process every Nth frame",
    1,
    5,
    2
)

possession_distance = st.sidebar.slider(
    "Possession distance",
    50,
    300,
    160,
    10
)

pixels_per_meter = st.sidebar.slider(
    "Pixels per meter (calibration)",
    5,
    100,
    20,
    1,
    help=(
        "Used to convert on-screen movement into real-world distance. "
        "Measure a known real-world length in your footage (e.g. the "
        "penalty box width, ~40m, or the pitch width, ~68m) in pixels "
        "and divide by its real length in meters to set this accurately. "
        "Distance/speed metrics are only as good as this calibration."
    )
)

st.sidebar.markdown("---")

st.sidebar.header("🇸🇷 Suriname Player Database")

use_roster = st.sidebar.checkbox(
    "Use player database",
    value=True
)

selected_player = None

if use_roster and not roster.empty:

    # Create player selection list
    player_options = ["All Players"]

    if "name" in roster.columns:
        player_options += roster["name"].tolist()
    elif "player_name" in roster.columns:
        player_options += roster["player_name"].tolist()

    selected_player = st.sidebar.selectbox(
        "🎯 Select Player to Search",
        player_options,
        index=0
    )


#test
if use_roster:

    if roster.empty:

        st.sidebar.error(
            "Player database was not found."
        )

    else:

        st.sidebar.success(
            f"{len(roster)} players loaded"
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_distance(point1, point2):
    return np.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


def calculate_player_metrics(
    player_positions,
    player_touches,
    fps,
    frame_skip,
    pixels_per_meter
):
    """
    Turn raw per-frame foot positions and touch counts into performance
    metrics: distance covered, touches, time visible and average speed.

    Distance is approximate: it assumes a flat, consistent scale across
    the frame (no perspective correction) and that gaps between detections
    of the same track are spaced `frame_skip / fps` seconds apart.
    """

    metrics = {}

    seconds_per_sample = (
        frame_skip / fps
        if fps
        else 0
    )

    for player_id, points in player_positions.items():

        pixel_distance = 0.0

        for previous_point, next_point in zip(points, points[1:]):

            pixel_distance += calculate_distance(
                previous_point,
                next_point
            )

        meters = (
            pixel_distance / pixels_per_meter
            if pixels_per_meter
            else 0.0
        )

        time_visible = seconds_per_sample * max(len(points) - 1, 0)

        avg_speed_kmh = (
            (meters / time_visible) * 3.6
            if time_visible > 0
            else 0.0
        )

        metrics[player_id] = {
            "distance_m": meters,
            "touches": player_touches.get(player_id, 0),
            "time_visible_s": time_visible,
            "avg_speed_kmh": avg_speed_kmh
        }

    return metrics


def generate_player_heatmap(
    positions,
    background_frame,
    bins_x=32,
    bins_y=20,
    min_points=4
):
    """
    Build a position heatmap for one player, overlaid on a real frame from
    the analyzed video so the density map has genuine spatial context.

    Returns an RGB numpy image ready for st.image, or None if there isn't
    enough tracked data / no background frame to draw on.
    """

    if (
        background_frame is None
        or positions is None
        or len(positions) < min_points
    ):
        return None

    height, width = background_frame.shape[:2]

    xs = np.array(
        [point[0] for point in positions],
        dtype=float
    )

    ys = np.array(
        [point[1] for point in positions],
        dtype=float
    )

    heatmap, _, _ = np.histogram2d(
        xs,
        ys,
        bins=[bins_x, bins_y],
        range=[[0, width], [0, height]]
    )

    # np.histogram2d indexes as [x_bin, y_bin]; transpose to [row=y, col=x]
    # so it lines up with image coordinates before resizing.
    heatmap = heatmap.T

    heatmap = cv2.GaussianBlur(
        heatmap.astype(np.float32),
        (0, 0),
        sigmaX=1.4,
        sigmaY=1.4
    )

    if heatmap.max() > 0:
        heatmap_norm = (
            heatmap / heatmap.max() * 255
        ).astype(np.uint8)
    else:
        heatmap_norm = heatmap.astype(np.uint8)

    heatmap_resized = cv2.resize(
        heatmap_norm,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_resized,
        cv2.COLORMAP_JET
    )

    # Only tint areas with meaningful density so the background video
    # frame stays clean and readable everywhere the player wasn't.
    mask = (heatmap_resized > 12).astype(np.uint8)
    mask_3ch = cv2.merge([mask, mask, mask])

    blended = cv2.addWeighted(
        background_frame,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    result = np.where(
        mask_3ch == 1,
        blended,
        background_frame
    )

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def find_player_by_shirt_number(
    shirt_number,
    roster_df
):

    if roster_df.empty:
        return pd.DataFrame()

    if "shirt_number" not in roster_df.columns:
        return pd.DataFrame()

    matches = roster_df[
        roster_df["shirt_number"].astype(str)
        == str(shirt_number)
    ]

    return matches


def classify_team_from_color(
    frame,
    box,
    team_a_rgb,
    team_b_rgb
):

    x1, y1, x2, y2 = map(
        int,
        box
    )

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return "Unknown"

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return "Unknown"

    crop_height, crop_width = crop.shape[:2]

    torso = crop[
        int(crop_height * 0.15):
        int(crop_height * 0.60),

        int(crop_width * 0.20):
        int(crop_width * 0.80)
    ]

    if torso.size == 0:
        return "Unknown"

    torso_rgb = cv2.cvtColor(
        torso,
        cv2.COLOR_BGR2RGB
    )

    average_color = np.mean(
        torso_rgb.reshape(-1, 3),
        axis=0
    )

    distance_a = np.linalg.norm(
        average_color - team_a_rgb
    )

    distance_b = np.linalg.norm(
        average_color - team_b_rgb
    )

    if min(
        distance_a,
        distance_b
    ) > 180:

        return "Unknown"

    if distance_a < distance_b:
        return "Team A"

    return "Team B"


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(
    input_path,
    output_path,
    confidence,
    frame_skip,
    possession_distance,
    team_a_rgb,
    team_b_rgb,
    selected_player_name=None,
    reference_descriptors=None
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    codec = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        codec,
        fps,
        (width, height)
    )

    frame_number = 0

    current_possession = None

    candidate_player = None

    candidate_count = 0

    possession_seconds = defaultdict(float)

    player_touches = defaultdict(int)

    # For each player, the video timestamp (in seconds) of each estimated
    # touch - lets us show the selected player a timeline of when they
    # touched the ball, not just a raw count.
    player_touch_timestamps = defaultdict(list)

    previous_possession = None

    player_team = {}

    player_positions = defaultdict(list)

    ball_history = []

    # Selected-player search state.
    # These variables must be initialized before the video loop because
    # they are updated inside the YOLO detection loop and returned later.
    target_found = False
    target_match_frames = 0
    target_best_matches = 0
    target_track_id = None

    # A clean, undrawn-on snapshot of the video, captured once, used later
    # as the background for per-player position heatmaps.
    background_frame = None

    progress = st.progress(0)

    status = st.empty()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        if background_frame is None:
            background_frame = frame.copy()

        if (
            frame_number % frame_skip
            != 0
        ):

            writer.write(frame)

            continue

        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            classes=[0, 32],
            verbose=False
        )

        result = results[0]

        players = []

        ball_center = None

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = [
                    None
                ] * len(boxes)

            class_ids = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            coordinates = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            for box, track_id, class_id in zip(
                coordinates,
                track_ids,
                class_ids
            ):

                # COCO class 32 = sports ball
                if class_id == 32:

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    ball_center = (
                        int((x1 + x2) / 2),
                        int((y1 + y2) / 2)
                    )

                    continue

                # COCO class 0 = person
                if class_id != 0:
                    continue

                if track_id is None:
                    continue

                player_id = int(
                    track_id
                )

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                foot_point = (
                    int((x1 + x2) / 2),
                    y2
                )

                team = classify_team_from_color(
                    frame,
                    box,
                    team_a_rgb,
                    team_b_rgb
                )

                player_team[player_id] = team

                player_positions[
                    player_id
                ].append(
                    foot_point
                )

                players.append(
                    {
                        "id": player_id,
                        "box": (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        "foot": foot_point,
                        "team": team
                    }
                )

                # ------------------------------------------------
                # SELECTED PLAYER SEARCH
                # ------------------------------------------------
                if selected_player_name and reference_descriptors is not None:
                    is_match, good_matches = player_visual_match(
                        frame,
                        (x1, y1, x2, y2),
                        reference_descriptors
                    )

                    if good_matches > target_best_matches:
                        target_best_matches = good_matches

                    if is_match:
                        target_match_frames += 1
                        target_track_id = player_id

                        # Require several matching frames to reduce false positives.
                        if target_match_frames >= 3:
                            target_found = True

        # ----------------------------------------------------
        # SMOOTH BALL LOCATION
        # ----------------------------------------------------

        if ball_center is not None:

            ball_history.append(
                ball_center
            )

            if len(ball_history) > 5:

                ball_history.pop(0)

            average_ball = np.mean(
                np.array(
                    ball_history
                ),
                axis=0
            )

            ball_center = (
                int(average_ball[0]),
                int(average_ball[1])
            )

        # ----------------------------------------------------
        # FIND PLAYER CLOSEST TO BALL
        # ----------------------------------------------------

        nearest_player = None

        nearest_distance = float(
            "inf"
        )

        if ball_center is not None:

            for player in players:

                distance = (
                    calculate_distance(
                        ball_center,
                        player["foot"]
                    )
                )

                if (
                    distance
                    < nearest_distance
                ):

                    nearest_distance = distance

                    nearest_player = player

        # ----------------------------------------------------
        # POSSESSION
        # ----------------------------------------------------

        if (
            nearest_player is not None
            and nearest_distance
            <= possession_distance
        ):

            candidate = (
                nearest_player["id"]
            )

            if candidate == candidate_player:

                candidate_count += 1

            else:

                candidate_player = candidate

                candidate_count = 1

            if candidate_count >= 3:

                current_possession = (
                    candidate_player
                )

        else:

            candidate_player = None

            candidate_count = 0

        # ----------------------------------------------------
        # POSSESSION TIME
        # ----------------------------------------------------

        if current_possession is not None:

            possession_seconds[
                current_possession
            ] += (
                frame_skip / fps
            )

        # ----------------------------------------------------
        # TOUCH COUNTING
        # ----------------------------------------------------
        # A "touch" is counted each time confirmed possession switches
        # to a player who didn't have it the moment before (i.e. the
        # start of a new possession spell for that player). This is an
        # approximation - it won't catch every single tap of the ball,
        # but it tracks meaningful touches without needing frame-perfect
        # ball-contact detection.

        if (
            current_possession is not None
            and current_possession != previous_possession
        ):

            player_touches[
                current_possession
            ] += 1

            player_touch_timestamps[
                current_possession
            ].append(
                frame_number / fps
            )

        previous_possession = current_possession

        # ----------------------------------------------------
        # DRAW BALL
        # ----------------------------------------------------

        if ball_center is not None:

            cv2.circle(
                frame,
                ball_center,
                10,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                "BALL",
                (
                    ball_center[0] + 12,
                    ball_center[1]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # ----------------------------------------------------
        # DRAW PLAYERS
        # ----------------------------------------------------

        for player in players:

            x1, y1, x2, y2 = (
                player["box"]
            )

            player_id = player["id"]

            team = player["team"]

            if team == "Team A":

                box_color = (
                    0,
                    0,
                    255
                )

            elif team == "Team B":

                box_color = (
                    255,
                    0,
                    0
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )

            thickness = 2

            if (
                player_id
                == current_possession
            ):

                box_color = (
                    0,
                    255,
                    255
                )

                thickness = 4

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                thickness
            )

            if (
                selected_player_name
                and target_found
                and player_id == target_track_id
            ):
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    5
                )

                cv2.putText(
                    frame,
                    f"TARGET: {selected_player_name}",
                    (x1, min(height - 10, y2 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            label = (
                f"ID {player_id} | "
                f"{team}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            if (
                player_id
                == current_possession
            ):

                cv2.putText(
                    frame,
                    "POSSESSION",
                    (
                        x1,
                        y2 + 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

        # ----------------------------------------------------
        # TOP INFORMATION
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 55),
            (15, 23, 42),
            -1
        )

        if current_possession is None:

            possession_text = (
                "No possession"
            )

        else:

            team = player_team.get(
                current_possession,
                "Unknown"
            )

            possession_text = (
                f"{team} | "
                f"Player ID "
                f"{current_possession}"
            )

        cv2.putText(
            frame,
            f"Possession: {possession_text}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # WRITE FRAME
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if total_frames > 0:

            progress.progress(
                min(
                    frame_number
                    / total_frames,
                    1.0
                )
            )

            status.text(
                f"Processing frame "
                f"{frame_number:,} / "
                f"{total_frames:,}"
            )

    cap.release()
    writer.release()

    progress.empty()
    status.empty()

    return (
        possession_seconds,
        player_team,
        player_positions,
        player_touches,
        player_touch_timestamps,
        fps,
        target_found,
        target_match_frames,
        target_best_matches,
        target_track_id,
        background_frame
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📹 Upload a football video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)


# ============================================================
# SELECTED PLAYER PREVIEW
# ============================================================

selected_player_image = None
reference_descriptors = None

if selected_player and selected_player != "All Players":

    # --------------------------------------------------------
    # SHIRT NUMBER & POSITION (from the roster database)
    # --------------------------------------------------------

    player_record = get_player_record(roster, selected_player)

    if player_record:

        shirt_number = get_roster_field(
            player_record,
            ["shirt_number", "number", "jersey_number", "squad_number"]
        )

        position = get_roster_field(
            player_record,
            ["position", "pos"]
        )

        detail_col1, detail_col2 = st.sidebar.columns(2)

        detail_col1.metric(
            "Shirt Number",
            shirt_number if shirt_number else "—"
        )

        detail_col2.metric(
            "Position",
            position if position else "—"
        )

    else:

        st.sidebar.info(
            "No roster record found for this player's number/position."
        )

    selected_player_image = find_player_image(selected_player)
    if selected_player_image:
        reference_descriptors = load_reference_features(selected_player_image)

        st.sidebar.image(
            selected_player_image,
            caption=f"Search target: {selected_player}",
            use_container_width=True
        )

        if reference_descriptors is None:
            st.sidebar.warning(
                "Could not extract visual features from this player's image."
            )
    else:
        st.sidebar.warning(
            "No matching player image was found in data/Players."
        )


# ============================================================
# HOME PAGE
# ============================================================

if uploaded_file is None:

    st.subheader(
        "Football Video Analysis"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Player Detection",
            "YOLO"
        )

    with col2:

        st.metric(
            "Player Tracking",
            "ByteTrack"
        )

    with col3:

        st.metric(
            "Ball Possession",
            "AI"
        )

    st.markdown("---")

    if not roster.empty:

        st.subheader(
            "🇸🇷 Suriname Player Database"
        )

        st.write(
            "The application currently has "
            f"{len(roster)} players available "
            "for roster cross-checking."
        )

else:

    st.subheader(
        "🎥 Uploaded Video"
    )

    st.video(
        uploaded_file
    )

    st.markdown("---")

    if st.button(
        "🚀 Start AI Analysis",
        type="primary",
        use_container_width=True
    ):

        team_a_rgb = np.array(
            [220, 50, 50],
            dtype=float
        )

        team_b_rgb = np.array(
            [50, 80, 220],
            dtype=float
        )

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            input_path = temp_file.name

        output_path = os.path.join(
            tempfile.gettempdir(),
            "visionxi_result.mp4"
        )

        try:

            with st.spinner(
                "Analyzing football video..."
            ):

                (
                    possession_seconds,
                    player_team,
                    player_positions,
                    player_touches,
                    player_touch_timestamps,
                    video_fps,
                    target_found,
                    target_match_frames,
                    target_best_matches,
                    target_track_id,
                    background_frame
                ) = analyze_video(
                    input_path,
                    output_path,
                    confidence,
                    frame_skip,
                    possession_distance,
                    team_a_rgb,
                    team_b_rgb,
                    selected_player_name=(
                        selected_player
                        if selected_player != "All Players"
                        else None
                    ),
                    reference_descriptors=reference_descriptors
                )

            st.success(
                "✅ Analysis completed!"
            )

            # ------------------------------------------------
            # PERFORMANCE METRICS (distance, touches, speed)
            # ------------------------------------------------

            player_metrics = calculate_player_metrics(
                player_positions,
                player_touches,
                video_fps,
                frame_skip,
                pixels_per_meter
            )

            # ------------------------------------------------
            # SELECTED PLAYER RESULT
            # ------------------------------------------------
            if selected_player and selected_player != "All Players":
                if target_found:
                    st.success(
                        f"🟢 Player found: {selected_player}"
                    )
                    st.info(
                        f"The selected player was matched across "
                        f"{target_match_frames} processed frames. "
                        f"Best visual match: {target_best_matches} feature matches."
                    )

                    # --------------------------------------------
                    # SELECTED PLAYER PERFORMANCE CARD
                    # --------------------------------------------

                    selected_metrics = player_metrics.get(
                        target_track_id,
                        {}
                    )

                    st.subheader(
                        f"📈 {selected_player} — Performance Metrics"
                    )

                    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                    perf_col1.metric(
                        "Distance Covered",
                        f"{selected_metrics.get('distance_m', 0.0):.1f} m"
                    )

                    perf_col2.metric(
                        "Touches",
                        selected_metrics.get("touches", 0)
                    )

                    perf_col3.metric(
                        "Possession Time",
                        f"{possession_seconds.get(target_track_id, 0.0):.1f} s"
                    )

                    perf_col4.metric(
                        "Avg Speed",
                        f"{selected_metrics.get('avg_speed_kmh', 0.0):.1f} km/h"
                    )

                    st.caption(
                        "Distance and speed are estimates based on the "
                        "'Pixels per meter' calibration slider in the sidebar. "
                        "Calibrate it against a known real-world length in your "
                        "footage for more accurate results."
                    )

                    # --------------------------------------------
                    # SELECTED PLAYER BALL TOUCHES
                    # --------------------------------------------

                    st.subheader(
                        f"⚽ {selected_player} — Ball Touches"
                    )

                    touch_timestamps = player_touch_timestamps.get(
                        target_track_id,
                        []
                    )

                    touch_count = len(touch_timestamps)

                    if touch_count > 0:

                        st.success(
                            f"🟢 {selected_player} touched the ball an "
                            f"estimated {touch_count} time"
                            f"{'s' if touch_count != 1 else ''} in this video."
                        )

                        timeline_rows = [
                            {
                                "Touch #": index + 1,
                                "Video Time": (
                                    f"{int(timestamp // 60):02d}:"
                                    f"{int(timestamp % 60):02d}"
                                )
                            }
                            for index, timestamp in enumerate(touch_timestamps)
                        ]

                        with st.expander(
                            f"View touch timeline ({touch_count} touches)"
                        ):
                            st.dataframe(
                                pd.DataFrame(timeline_rows),
                                use_container_width=True,
                                hide_index=True
                            )

                    else:

                        st.warning(
                            f"🟡 No confirmed ball touches were detected for "
                            f"{selected_player} in this video."
                        )

                    st.caption(
                        "Touches are estimated from ball-proximity possession: "
                        "each time the tracked ball comes within the sidebar's "
                        "'Possession distance' of this player and stays there, "
                        "it's counted as one touch. Fast, off-camera, or heavily "
                        "occluded touches may be missed or undercounted."
                    )

                    # --------------------------------------------
                    # SELECTED PLAYER HEATMAP
                    # --------------------------------------------

                    st.subheader(
                        f"🔥 {selected_player} — Position Heatmap"
                    )

                    heatmap_image = generate_player_heatmap(
                        player_positions.get(target_track_id, []),
                        background_frame
                    )

                    if heatmap_image is not None:

                        st.image(
                            heatmap_image,
                            use_container_width=True,
                            caption=(
                                "Warmer colors = more time spent in that "
                                "area of the frame"
                            )
                        )

                    else:

                        st.info(
                            "Not enough tracked positions to build a "
                            "heatmap for this player yet."
                        )
                else:
                    st.error(
                        f"🔴 Player not found: {selected_player}"
                    )
                    st.info(
                        "Try a clearer player image, a lower detection confidence, "
                        "or a video where the player's face/body is more visible."
                    )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.subheader(
                "📊 Match Overview"
            )

            total_possession = sum(
                possession_seconds.values()
            )

            team_a_seconds = 0

            team_b_seconds = 0

            for player_id, seconds in (
                possession_seconds.items()
            ):

                team = player_team.get(
                    player_id,
                    "Unknown"
                )

                if team == "Team A":

                    team_a_seconds += seconds

                elif team == "Team B":

                    team_b_seconds += seconds

            if total_possession > 0:

                team_a_percentage = (
                    team_a_seconds
                    / total_possession
                    * 100
                )

                team_b_percentage = (
                    team_b_seconds
                    / total_possession
                    * 100
                )

            else:

                team_a_percentage = 0

                team_b_percentage = 0

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Tracked Players",
                len(player_team)
            )

            col2.metric(
                "Team A Possession",
                f"{team_a_percentage:.1f}%"
            )

            col3.metric(
                "Team B Possession",
                f"{team_b_percentage:.1f}%"
            )

            # ------------------------------------------------
            # POSSESSION CHART
            # ------------------------------------------------

            st.subheader(
                "⚽ Ball Possession"
            )

            possession_df = pd.DataFrame(
                {
                    "Team": [
                        "Team A",
                        "Team B"
                    ],
                    "Possession (%)": [
                        team_a_percentage,
                        team_b_percentage
                    ]
                }
            )

            st.bar_chart(
                possession_df.set_index(
                    "Team"
                )
            )

            # ------------------------------------------------
            # PLAYER TABLE
            # ------------------------------------------------

            st.subheader(
                "👤 Player Analysis"
            )

            rows = []

            for player_id, team in (
                player_team.items()
            ):

                seconds = possession_seconds.get(
                    player_id,
                    0.0
                )

                percentage = 0

                if total_possession > 0:

                    percentage = (
                        seconds
                        / total_possession
                        * 100
                    )

                player_metric = player_metrics.get(
                    player_id,
                    {}
                )

                rows.append(
                    {
                        "Tracking ID":
                            player_id,

                        "Team":
                            team,

                        "Possession (sec)":
                            round(
                                seconds,
                                2
                            ),

                        "Possession (%)":
                            round(
                                percentage,
                                2
                            ),

                        "Distance (m)":
                            round(
                                player_metric.get(
                                    "distance_m",
                                    0.0
                                ),
                                1
                            ),

                        "Touches":
                            player_metric.get(
                                "touches",
                                0
                            ),

                        "Avg Speed (km/h)":
                            round(
                                player_metric.get(
                                    "avg_speed_kmh",
                                    0.0
                                ),
                                1
                            )
                    }
                )

            if rows:

                player_df = pd.DataFrame(
                    rows
                )

                player_df = player_df.sort_values(
                    "Possession (%)",
                    ascending=False
                )

                st.dataframe(
                    player_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No players were tracked in this video."
                )

            # ------------------------------------------------
            # ROSTER
            # ------------------------------------------------

            if use_roster and not roster.empty:

                st.subheader(
                    "🇸🇷 Suriname Player Database"
                )

                st.dataframe(
                    roster,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # PROCESSED VIDEO
            # ------------------------------------------------

            st.subheader(
                "🎬 AI Processed Video"
            )

            if os.path.exists(
                output_path
            ):

                st.video(
                    output_path
                )

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    st.download_button(
                        "⬇️ Download analyzed video",
                        video_file,
                        file_name=(
                            "visionxi_analysis.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True
                    )

        except Exception as error:

            st.error(
                "The video analysis failed."
            )

            st.exception(
                error
            )

        finally:

            if os.path.exists(
                input_path
            ):

                os.remove(
                    input_path
                )