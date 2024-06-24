from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import pandas as pd
import os
import sys
import numpy as np
sys.path.append('../')
from utils import get_center_of_box, get_width_of_box

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i+ batch_size], conf = 0.1)
            detections.extend(detections_batch)

        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k,v, in cls_names.items()}

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # tracks
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox" : bbox}
            
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox" : bbox}

            # to loop without the tracks
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox" : bbox}

            # print(detection_with_tracks)

        if stub_path != None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_box(bbox)
        width = get_width_of_box(bbox)

        x_center = int(x_center)
        width = int(width)
        axes = (width, int(0.35 * width))
        center = (x_center, y2)

        # print(f"ellipse for player_id {player_id} at {center} with axes {axes}")

        cv2.ellipse(frame, center = center, axes = axes, angle = 0, startAngle = 235, 
                    endAngle = -45, color = color, thickness = 2, lineType = cv2.LINE_AA)
        
        rectange_width = 40
        rectange_height = 20
        x1_rect = x_center - rectange_width//2
        x2_rect = x_center + rectange_width//2
        y1_rect = (y2 - rectange_height//2) + 15
        y2_rect = (y2 + rectange_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

            x1_text = x1_rect + 20
            if x1_text > 99:
                x1_text = x1_text - 10

            if track_id < 10:
                x1_text = x1_text + 5
            elif track_id > 100:
                x1_text = x1_text - 10    

            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame
    
    def draw_ball_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_box(bbox)

        triangle_points = np.array([
            [x, y],
            [x -10, y -10],
            [x +10, y -10]
        ])

        cv2.fillPoly(frame, [triangle_points], color)
        # cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def team_ball_control(self, frame, frame_num, team_ball_control):
        overlay =  frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num +1]

        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Possesion: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Possesion: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotaions(self, frame, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(frame):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # draw players
            for player_id, player in player_dict.items():
                # print(f"annotations for frame {frame_num}, player_id {player_id}")
                color = player.get("team_color", (0, 255, 0))
                frame = self.draw_ellipse(frame, player["bbox"], color, player_id)

                if player.get('ball_possession', False):
                    frame = self.draw_ball_triangle(frame, player["bbox"], color)

            # draw referees
            for _, referee in referee_dict.items():
                # print(f" annotations for frame {frame_num}, referee_id {referee_id}")
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # draw ball 
            for _, ball in ball_dict.items():
                # print(f" annotations for frame {frame_num}, ball_id {ball_id}")
                frame = self.draw_ball_triangle(frame, ball["bbox"], (0, 0, 255))

            # ball possession stats
            frame = self.team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate(method = 'linear')
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
