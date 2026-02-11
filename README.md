# 🦋 Interactive 3D Lorenz Attractor Visualization


## 1. Introduction and Overview
This project merges mathematical visualization with gesture-based human-computer interaction. The application renders a **Lorenz Attractor**—a system of differential equations known for its "butterfly" shape and chaotic behavior—in real-time.

Instead of static 3D plots, this system utilizes **MediaPipe Hand Tracking** to allow physical manipulation of the model. Users can rotate the attractor, zoom, and alter the chaos equations' parameters using natural hand gestures.

## 2. Features
The application supports **dual-handed interaction** with specific roles assigned to each hand:

* **🔄 3D Object Rotation (Left Hand):** Tracks the orientation (pitch/roll) of the left hand. Tilting your palm rotates the 3D attractor, mimicking holding a physical object.
* **🔍 Depth Control / Zoom (Left Hand Pinch):** * **Pinch Close:** Zoom Out.
    * **Spread Open:** Zoom In.
* **🌀 Chaos Parameter Manipulation (Right Hand):** Modifies the **$\beta$ (beta)** parameter of the Lorenz equation. Observe how the shape evolves from stable loops to chaotic patterns in real-time.
* **🌈 Dynamic Visualization:** Uses a gradient color loop to distinguish time steps and real-time projection for 3D-to-2D mapping.

---

## 3. How to Use

1.  **Launch:** Run the Python script with your webcam connected.
2.  **Left Hand (Navigation):** * **Rotate:** Tilt palm up/down or left/right.
    * **Zoom:** Pinch thumb and index finger.
3.  **Right Hand (Simulation):** * **Modify Shape:** Pinch to reduce $\beta$ (collapse wings), spread to increase $\beta$ (expand wings).
4.  **Exit:** Press the **'Q'** key.

---

## 4. Implementation Details

### 4.1. Computer Vision Pipeline
Built with **MediaPipe Hands** to detect 21 3D landmarks per hand.

* **Rotation:** Calculated via `numpy.arctan2` using vectors between the wrist, index, and pinky landmarks.
* **Pinching:** Uses Euclidean distance between Landmark 4 (Thumb Tip) and Landmark 8 (Index Tip), mapped to simulation values via `numpy.interp`.



### 4.2. Mathematical Model (The Lorenz System)
The system solves these differential equations:

$$\frac{dx}{dt} = \sigma(y - x)$$
$$\frac{dy}{dt} = x(\rho - z) - y$$
$$\frac{dz}{dt} = xy - \beta z$$

Numerical integration is performed using **Euler's method** over 4,000 steps ($dt=0.01$).



### 4.3. 3D Projection and Camera Geometry
Custom pipeline implemented using `cv2.projectPoints`:
1.  **Euler to Rodrigues:** Converts Euler angles to a Rotation Matrix, then to a Rodrigues Vector.
2.  **Transformation:** Projects 3D points ($X,Y,Z$) onto the 2D plane using a simulated camera matrix.

---

## 5. Video Demonstration

[![Watch the video](https://img.youtube.com/vi/pbHNGbqNk7g/0.jpg)](https://youtu.be/pbHNGbqNk7g)

