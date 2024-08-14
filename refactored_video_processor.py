import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Optional, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, input_path: str, output_folder: str, file_type: str, batch_size: int = 5):
        """
        Initialize the VideoProcessor class with input and output paths.

        Args:
            input_path (str): Path to the input video or folder.
            output_folder (str): Path to the output folder for results.
            batch_size (int): Number of frames to process in a batch.
            file_type (str): Video file type as string.
        """
        self.input_path = input_path
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.file_type = file_type

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

    def detect_pose_landmarks_batch(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Detect pose landmarks in a batch of frames using MediaPipe Pose.

        Args:
            frames (List[np.ndarray]): List of input video frames.

        Returns:
            List[Optional[np.ndarray]]: List of arrays of landmark coordinates or None if no landmarks are detected.
        """
        results = [self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
        return [
            np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]) if res.pose_landmarks else None
            for res in results
        ]

    def process_video(self, video_path: str) -> List[np.ndarray]:
        """
        Process the video to extract pose landmarks and save them to an Excel file.

        Args:
            video_path (str): Path to the input video.

        Returns:
            List[np.ndarray]: List of landmark arrays for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        landmarks_list = []

        with ThreadPoolExecutor() as executor:
            while cap.isOpened():
                frames = []
                for _ in range(self.batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)

                if not frames:
                    break

                # Detect pose landmarks in batches
                batch_results = executor.submit(self.detect_pose_landmarks_batch, frames)
                landmarks_list.extend(batch_results.result())

        cap.release()

        # Save landmarks to Excel
        self.save_landmarks_to_excel(landmarks_list, video_path)

        return landmarks_list

    def save_landmarks_to_excel(self, landmarks_list: List[np.ndarray], video_path: str) -> None:
        """
        Save the detected landmarks to an Excel file.

        Args:
            landmarks_list (List[np.ndarray]): List of landmark arrays for each frame.
            video_path (str): Path to the input video.
        """
        data_frames = [
            pd.DataFrame(ldmk, columns=['X', 'Y', 'Z']).assign(Frame=idx)
            for idx, ldmk in enumerate(landmarks_list) if ldmk is not None
        ]

        if data_frames:  # Ensure there's data to concatenate
            overall_coordinates_df = pd.concat(data_frames, ignore_index=True)
            overall_coordinates_df = overall_coordinates_df[['Frame', 'X', 'Y', 'Z']]

            # Create output Excel path
            excel_filename = os.path.splitext(os.path.basename(video_path))[0] + ".xlsx"
            output_excel_path = os.path.join(self.output_folder, excel_filename)

            overall_coordinates_df.to_excel(output_excel_path, index=False)

    def overlay_skeleton_on_video(self, video_path: str, landmarks_array: List[np.ndarray]) -> None:
        """
        Overlay the skeleton on the video and save it.

        Args:
            video_path (str): Path to the input video.
            landmarks_array (List[np.ndarray]): List of landmark arrays for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video path
        video_filename = os.path.splitext(os.path.basename(video_path))[0] + "_overlay.mp4"
        output_video_path = os.path.join(self.output_folder, video_filename)

        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count < len(landmarks_array):
                landmarks = landmarks_array[frame_count]
                if landmarks is not None:
                    frame = self.draw_skeleton_on_frame(frame, landmarks)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

    def draw_skeleton_on_frame(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw the skeleton on a given video frame.

        Args:
            frame (np.ndarray): The input video frame.
            landmarks (np.ndarray): Landmark coordinates for the frame.

        Returns:
            np.ndarray: The frame with skeleton overlay.
        """
        height, width, _ = frame.shape
        landmarks[:, :2] *= [width, height]  # Scale to frame dimensions

        # Define connections for left, right limbs, and torso
        left_connections = [(11, 13), (13, 15), (15, 19), (15, 21), (23, 25), (25, 27), (27, 29), (29, 31)]
        right_connections = [(12, 14), (14, 16), (16, 20), (16, 22), (24, 26), (26, 28), (28, 30), (30, 32)]
        torso_connections = [(11, 12), (12, 24), (24, 23), (23, 11), (11, 23), (12, 24), (0, 1), (1, 2), (2, 3), (3, 7),
                             (0, 4), (4, 5), (5, 6), (6, 8)]

        def is_valid_point(point: Tuple[float, float]) -> bool:
            """
            Check if a point is valid and within frame boundaries.

            Args:
                point (Tuple[float, float]): The point coordinates.

            Returns:
                bool: True if the point is valid, False otherwise.
            """
            return all(0 <= v < s for v, s in zip(point, (width, height))) and not np.isnan(point).any()

        def draw_limb(limb: List[Tuple[int, int]], limb_color: Tuple[int, int, int]) -> None:
            """
            Draw a limb on the frame.

            Args:
                limb (List[Tuple[int, int]]): List of connections for the limb.
                limb_color (Tuple[int, int, int]): Color of the limb.
            """
            for connection in limb:
                pt1 = tuple(map(int, landmarks[connection[0], :2]))
                pt2 = tuple(map(int, landmarks[connection[1], :2]))
                if is_valid_point(pt1) and is_valid_point(pt2):
                    cv2.line(frame, pt1, pt2, limb_color, 2)

        draw_limb(torso_connections, (255, 255, 255))
        draw_limb(left_connections, (0, 0, 255))
        draw_limb(right_connections, (255, 0, 0))

        # Draw landmarks as circles
        for i, lm in enumerate(landmarks):
            if is_valid_point(lm[:2]):
                color = (255, 255, 255)
                if i in [11, 13, 15, 19, 21, 23, 25, 27, 29, 31]:
                    color = (0, 0, 255)  # Red for left-side landmarks
                elif i in [12, 14, 16, 20, 22, 24, 26, 28, 30, 32]:
                    color = (255, 0, 0)  # Blue for right-side landmarks
                cv2.circle(frame, tuple(map(int, lm[:2])), 3, color, -1)

        # Add legend
        self.add_legend(frame)

        return frame

    def add_legend(self, frame: np.ndarray) -> None:
        """
        Add a legend to the video frame.

        Args:
            frame (np.ndarray): The input video frame.
        """
        legend_x, legend_y = 20, 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Draw legend background
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + 180, legend_y + 70), (255, 255, 255), -1)

        # Add legend text
        cv2.putText(frame, 'Left Limbs (Red)', (legend_x + 10, legend_y + 20), font, font_scale, (0, 0, 255), thickness)
        cv2.putText(frame, 'Right Limbs (Blue)', (legend_x + 10, legend_y + 40), font, font_scale, (255, 0, 0),
                    thickness)
        cv2.putText(frame, 'Torso/Head (White)', (legend_x + 10, legend_y + 60), font, font_scale, (0, 0, 0), thickness)

    def process_videos(self) -> None:
        """
        Process videos in the specified folder, overlay skeletons, and save results.
        """
        video_paths = glob(os.path.join(self.input_path, '**', f'*{self.file_type}'), recursive=True)

        for video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
            os.makedirs(self.output_folder, exist_ok=True)

            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps if fps > 0 else 0  # Calculate duration in seconds
            cap.release()

            start_time = time.time()  # Start time before processing the video

            # Process the video and overlay skeleton
            landmarks_array = self.process_video(video_path)
            if landmarks_array:
                self.overlay_skeleton_on_video(video_path, landmarks_array)

            end_time = time.time()  # End time after processing the video

            # Calculate processing time
            processing_time = end_time - start_time

            # Print video length and processing time
            print(f"Video: {video_path}")
            print(f"Length: {duration:.2f} seconds")
            print(f"Processing Time: {processing_time:.2f} seconds\n")


def main():
    parser = argparse.ArgumentParser(description="Process and overlay skeletons on videos.")
    parser.add_argument("--input_path", type=str,
                        default=".",
                        help="Path to the input video or folder.")
    parser.add_argument("--output_folder", type=str,
                        default=".",
                        help="Path to the output folder.")
    parser.add_argument("--file_type", type=str,
                        default=".mov",
                        help="Path to the output folder.")

    args = parser.parse_args()

    processor = VideoProcessor(args.input_path, args.output_folder, args.file_type)
    processor.process_videos()


if __name__ == "__main__":
    main()
