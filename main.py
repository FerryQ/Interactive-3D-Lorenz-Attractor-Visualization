import cv2
import mediapipe as mp
import numpy as np

# Configuration
HEIGHT = 1200
WIDTH = 1100
NUM_STEPS = 4000 
DT = 0.01
STEP = 0
# Colors for the attractor (White, Blue, Red, Green)
HUE = [(255,255,255),(255,0,0),(0,0,255),(0,255,0)]



def lorenz(xyz, s, r, b):
    """Calculates the derivatives for the Lorenz system."""
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

def run_simulation(s, r, b):
    """Generates the 3D points for the Lorenz Attractor."""
    xyzs = np.empty((NUM_STEPS + 1, 3))
    xyzs[0] = (0., 1., 1.05)  # Initial values 

    for i in range(NUM_STEPS):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i], s, r, b) * DT
    return xyzs

def draw_attractor(xyzs,img):
     for i in range(1,len(xyzs)):
        point_1 = (xyzs[i-1])
        point_2 = (xyzs[i])

        cv2.line(img,point_1,point_2,HUE[(i + STEP)%len(HUE)],1,lineType=cv2.LINE_AA)

def euler_to_rvec(pitch, yaw, roll):
    """Converts Euler angles to a Rotation Vector (Rodrigues)."""
    # Rotation matrix around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    # Rotation matrix around Y axis
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    # Rotation matrix around Z axis
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def projection(z,points,angle_z,angle_y,angle_x):
    """Projects 3D points onto the 2D screen plane."""
    # Camera intrinsics
    fx = WIDTH * 0.8
    fy = WIDTH * 0.8
    cx = WIDTH//4
    cy = HEIGHT//4
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], np.float32)
    dist_coeffs = np.zeros((5, 1), np.float32)

    # Rotation and Translation
    rvec = euler_to_rvec(angle_x, angle_y, angle_z)
    tvec = np.array([0, 0, z], dtype=np.float32)
    
    # Project 3D -> 2D
    points_2d, _ = cv2.projectPoints(points,
                                    rvec, tvec,
                                    camera_matrix,
                                    dist_coeffs)
    return points_2d.squeeze().astype(np.int32)

# --- Initialization ---
s_init = 10.0
r_init = 28.0
b_init = 2.667

# Generate initial shape
xyzs = run_simulation(s_init, r_init, b_init)
xyzs = xyzs - np.mean(xyzs, axis=0)

# Smoothing variables
smooth_x = 0
smooth_y = 0
smooth_z = 0
smooth_d = 0
alpha = 0.6

# MediaPipe setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Hands setup
connections = {(4,8)}
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    RGB_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)

    # Default interactive values for this frame
    dist_w = 100
    dist_xy = 0
    angle_z = 0.0
    angle_y = 0.0
    angle_x = 0.0
    param_b = 2.667


    if result.multi_hand_landmarks:
        # Roll
        for hand_landmarks,hand_info in zip(result.multi_hand_landmarks,result.multi_handedness):
            hand_label = hand_info.classification[0].label
            if hand_label == "Left":               
                hand_tip = hand_landmarks.landmark[0]
                index_hand = hand_landmarks.landmark[5]

                dx = hand_tip.x-index_hand.x
                dy = hand_tip.y-index_hand.y

                angle_z = -np.arctan2(dx,dy) + (np.pi / 2)

            # Drawing line between the finger tips
            mp_drawing.draw_landmarks(frame,hand_landmarks,connections)
    

        # Using "world" coordinates because I need the the z component
        for hand_landmarks,hand_info in zip(result.multi_hand_world_landmarks,result.multi_handedness):
            hand_label = hand_info.classification[0].label
            if hand_label == "Left":
                # Zoom effect with fingers on left hand
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                thumb_tip_cords = np.array((thumb_tip.x, thumb_tip.y, thumb_tip.z))
                index_tip_cords = np.array((index_tip.x, index_tip.y, index_tip.z))

                raw_dist = np.linalg.norm(thumb_tip_cords-index_tip_cords)
                dist_w = np.interp(raw_dist, [0.02, 0.15], [300, -50])
                
                # Pitch using world coordinates (needed the z axis)
                hand_inside = hand_landmarks.landmark[5]
                pinky = hand_landmarks.landmark[17]

                dz_2 = hand_inside.z - pinky.z
                dx = hand_inside.x - pinky.x

                angle_x = -np.arctan2(dz_2,dx)

            # Changing pramater "b" with your right hand fingers
            if hand_label == "Right":
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                thumb_tip_cords = np.array((thumb_tip.x, thumb_tip.y, thumb_tip.z))
                index_tip_cords = np.array((index_tip.x, index_tip.y, index_tip.z))

                raw_dist = np.linalg.norm(thumb_tip_cords-index_tip_cords)
                param_b = np.interp(raw_dist, [0.01, 0.15], [0.1, 20])

    # smoothing effect when zooming or rotation
    smooth_x = alpha * smooth_x + (1 - alpha) * angle_x
    smooth_y = alpha * smooth_y + (1 - alpha) * angle_y
    smooth_z = alpha * smooth_z + (1 - alpha) * angle_z
    smooth_d = alpha * smooth_d + (1 - alpha) * dist_w

    # Project points to 2D
    pixel_points = projection(smooth_d,xyzs,smooth_z,angle_y,smooth_x)
    draw_attractor(pixel_points,frame)
    STEP += 1 #increasing step for animation purposes
    text_info = f"b_param:{param_b}"
    text_exit = f"EXIT BY PRESSING \"Q\""


    (text_width, text_height), baseline = cv2.getTextSize(
        text=text_info, 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1, 
        thickness=2
    )
    (text_width2, text_height2), baseline = cv2.getTextSize(
        text=text_exit, 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1, 
        thickness=2
    )
    frame = cv2.putText(frame,text_info,(WIDTH-text_width//2,text_height),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    frame = cv2.putText(frame,text_exit,(WIDTH-text_width2//2,3*text_height),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)


    cv2.imshow("Interactive Lorenz AttractorZMRD",frame)
    xyzs = run_simulation(s_init, r_init, param_b)
    xyzs = xyzs - np.mean(xyzs, axis=0)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
