// Updated script.js with improved gaze scoring for eye movement
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver } = vision;

// DOM Elements
const demosSection = document.getElementById("demos");
const eyeContactScoreElement = document.getElementById("eye-contact-score");
const headPostureScoreElement = document.getElementById("head-posture-score");
const overallScoreElement = document.getElementById("overall-score");
const enableWebcamButton = document.getElementById("webcamButton");
const videoContainer = document.getElementById("videoContainer");

// Global Variables
let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;

// Scoring System
let totalEyeScore = 0;
let totalHeadScore = 0;
let totalFrames = 0;
let scoringStarted = false;
let baselineEAR = 0;
let frameCount = 0;
let eyeContactWeight = 0.6;
let headPostureWeight = 0.4;
const ADAPTIVE_FRAMES = 30;
const MAX_HEAD_ANGLE = 25;
const FACE_BOUNDS_RATIO = { min: 0.7, max: 1.3 };

// Enhanced deviation penalties for gaze
const deviationPenalty = {
  horizontal: {
    left: 0.1,   // Stricter penalty for left gaze
    center: 1.0,
    right: 0.1   // Stricter penalty for right gaze
  },
  vertical: {
    up: 0.1,     // Stricter penalty for upward gaze
    center: 1.0,
    down: 0.1    // Stricter penalty for downward gaze
  }
};

// Face Mesh Constants
const LEFT_EYE_INDICES = [33, 133, 159, 145];
const RIGHT_EYE_INDICES = [362, 263, 386, 374];
const LEFT_IRIS_INDICES = [468, 469, 470, 471];
const RIGHT_IRIS_INDICES = [473, 474, 475, 476];

// Video Elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// Initialize FaceLandmarker
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU",
      modelOptions: {
        refineLandmarks: true,
        contourThreshold: 0.5
      }
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO",
    numFaces: 1
  });
}
createFaceLandmarker();

// Webcam Functions
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam() {
  if (!faceLandmarker) return;

  if (webcamRunning) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE WEBCAM";
    videoContainer.style.display = "none";
    video.srcObject.getTracks().forEach(track => track.stop());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE WEBCAM";
    videoContainer.style.display = "block";
    
    // Reset counters
    totalEyeScore = 0;
    totalHeadScore = 0;
    totalFrames = 0;
    scoringStarted = false;
    baselineEAR = 0;
    frameCount = 0;

    const constraints = { 
      video: { 
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      } 
    };
    
    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
      });
  }
}

if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Video Processing
function adjustVideoSize() {
  const container = document.querySelector(".video-container");
  const videoAspectRatio = video.videoWidth / video.videoHeight;
  container.style.paddingTop = `${(1 / videoAspectRatio) * 100}%`;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  video.style.width = '100%';
  video.style.height = '100%';
}

// Enhanced Scoring Functions
function calculateHeadScore(rotation) {
  const steepness = 0.25;
  const anglePenalty = 0.15;
  
  const pitch = Math.abs(rotation.pitch);
  const yaw = Math.abs(rotation.yaw);
  const roll = Math.abs(rotation.roll);

  const baseScore = (
    1 / (1 + Math.exp(steepness * (pitch - MAX_HEAD_ANGLE))) +
    1 / (1 + Math.exp(steepness * (yaw - MAX_HEAD_ANGLE))) +
    1 / (1 + Math.exp(steepness * (roll - MAX_HEAD_ANGLE)))
  ) / 3;

  const excessPenalty = 
    Math.max(0, pitch - MAX_HEAD_ANGLE) * anglePenalty +
    Math.max(0, yaw - MAX_HEAD_ANGLE) * anglePenalty +
    Math.max(0, roll - MAX_HEAD_ANGLE) * anglePenalty;

  return Math.max(0, baseScore - excessPenalty);
}

function calculateGazeScore(gaze) {
  const leftHPenalty = deviationPenalty.horizontal[gaze.left.horizontal];
  const leftVPenalty = deviationPenalty.vertical[gaze.left.vertical];
  const rightHPenalty = deviationPenalty.horizontal[gaze.right.horizontal];
  const rightVPenalty = deviationPenalty.vertical[gaze.right.vertical];

  // Average penalties for both eyes
  const hPenalty = (leftHPenalty + rightHPenalty) / 2;
  const vPenalty = (leftVPenalty + rightVPenalty) / 2;

  // Stricter scoring: penalize more when eyes deviate from center
  let baseScore = hPenalty * vPenalty;
  if (gaze.left.horizontal !== 'center' || gaze.left.vertical !== 'center' ||
      gaze.right.horizontal !== 'center' || gaze.right.vertical !== 'center') {
    baseScore *= 0.7; // Additional penalty for any off-center gaze
  }

  return baseScore;
}

function isFaceForward(landmarks) {
  const faceBounds = getFaceBounds(landmarks);
  const ratio = (faceBounds.maxX - faceBounds.minX) / (faceBounds.maxY - faceBounds.minY);
  return ratio > FACE_BOUNDS_RATIO.min && ratio < FACE_BOUNDS_RATIO.max;
}

function getFaceBounds(landmarks) {
  const xs = landmarks.map(p => p.x);
  const ys = landmarks.map(p => p.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys)
  };
}

function calculateEyeClosureScore(leftEAR, rightEAR) {
  const eyeClosedThreshold = baselineEAR * 0.3;
  const avgEAR = (leftEAR + rightEAR) / 2;
  
  if (avgEAR < eyeClosedThreshold) return 0;
  if (avgEAR < baselineEAR * 0.5) return 0.3;
  return 1;
}

function calculateEAR(landmarks, eyeIndices) {
  const [p1, p2, p3, p4] = eyeIndices.map(i => landmarks[i]);
  const horizontal = Math.hypot(p1.x - p2.x, p1.y - p2.y);
  const vertical = Math.hypot(p3.x - p4.x, p3.y - p4.y);
  return vertical / (2 * horizontal + 1e-6);
}

// Detection Functions
function detectGazeDirection(landmarks) {
  const leftEyeBounds = getEyeBounds(landmarks, LEFT_EYE_INDICES);
  const rightEyeBounds = getEyeBounds(landmarks, RIGHT_EYE_INDICES);
  const leftIris = landmarks[LEFT_IRIS_INDICES[0]];
  const rightIris = landmarks[RIGHT_IRIS_INDICES[0]];

  return {
    left: calculateEyeZone(leftIris, leftEyeBounds),
    right: calculateEyeZone(rightIris, rightEyeBounds)
  };
}

function getEyeBounds(landmarks, indices) {
  const xs = indices.map(i => landmarks[i].x);
  const ys = indices.map(i => landmarks[i].y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys)
  };
}

function calculateEyeZone(iris, bounds) {
  const xQuarter = (bounds.maxX - bounds.minX) / 4;
  const yQuarter = (bounds.maxY - bounds.minY) / 4;
  const horizontalPos = iris.x - bounds.minX;
  const verticalPos = iris.y - bounds.minY;

  let hDir = 'center';
  let vDir = 'center';

  if (horizontalPos < xQuarter) hDir = 'left';
  else if (horizontalPos > 3 * xQuarter) hDir = 'right';
  if (verticalPos < yQuarter) vDir = 'up';
  else if (verticalPos > 3 * yQuarter) vDir = 'down';

  return { horizontal: hDir, vertical: vDir };
}

function matrixToEulerAngles(matrix) {
  const pitch = Math.atan2(matrix[9], matrix[10]) * (180 / Math.PI);
  const yaw = Math.atan2(-matrix[8], Math.sqrt(matrix[9] ** 2 + matrix[10] ** 2)) * (180 / Math.PI);
  const roll = Math.atan2(matrix[4], matrix[0]) * (180 / Math.PI);
  return { pitch, yaw, roll };
}

// Score Updates
function updateScores(eyeScore, headScore, gaze) {
  if (scoringStarted) {
    // Apply micro-movement penalty
    const movementPenalty = detectMicroMovements(gaze);
    eyeScore = Math.max(0, eyeScore - movementPenalty);

    totalEyeScore += eyeScore;
    totalHeadScore += headScore;

    const eyeAverage = totalEyeScore / totalFrames;
    const headAverage = totalHeadScore / totalFrames;
    const overall = (eyeAverage * eyeContactWeight) + (headAverage * headPostureWeight);

    eyeContactScoreElement.textContent = `${(eyeAverage * 100).toFixed(1)}%`;
    headPostureScoreElement.textContent = `${(headAverage * 100).toFixed(1)}%`;
    overallScoreElement.textContent = `${(overall * 100).toFixed(1)}%`;
  }
}

function provideRealTimeFeedback(eyeScore, headScore) {
  const messages = [];
  if (eyeScore < 0.5) messages.push("Look directly at the camera");
  if (headScore < 0.6) messages.push("Face the camera directly");
  if (messages.length > 0) showUserTips(messages.join(" • "));
}

function showUserTips(message) {
  const existingTips = document.querySelector('.glasses-warning');
  if (existingTips) existingTips.remove();

  const tipElement = document.createElement('div');
  tipElement.className = 'glasses-warning';
  tipElement.textContent = message;
  videoContainer.appendChild(tipElement);
  
  setTimeout(() => tipElement.remove(), 5000);
}

// Calibration Functions
function calibrateForGlasses(landmarks) {
  if (frameCount < ADAPTIVE_FRAMES && landmarks) {
    const leftEAR = calculateEAR(landmarks, LEFT_EYE_INDICES);
    const rightEAR = calculateEAR(landmarks, RIGHT_EYE_INDICES);
    baselineEAR += (leftEAR + rightEAR) / 2;
    frameCount++;
    if (frameCount === ADAPTIVE_FRAMES) {
      baselineEAR = baselineEAR / ADAPTIVE_FRAMES;
      scoringStarted = true; // Start scoring after calibration
    }
  }
}

// Detect Glasses Function
function detectGlasses(blendshapes) {
  if (!Array.isArray(blendshapes)) return false;
  const glassesCategories = ['eyeGlasses', 'darkGlasses'];
  for (let category of glassesCategories) {
    const score = blendshapes.find(b => b.categoryName === category)?.score || 0;
    if (score > 0.5) return true;
  }
  return false;
}

// Main Prediction Loop
async function predictWebcam() {
  if (!webcamRunning) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    return;
  }

  totalFrames++; // Always increment frame count

  adjustVideoSize();
  const startTimeMs = Date.now();
  const results = faceLandmarker.detectForVideo(video, startTimeMs);

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.faceLandmarks?.length > 0 && results.facialTransformationMatrixes?.length > 0) {
    const landmarks = results.faceLandmarks[0];
    const blendshapes = Array.isArray(results.faceBlendshapes) && results.faceBlendshapes.length > 0 ? results.faceBlendshapes[0] : [];

    // Detect glasses and adjust weights
    if (blendshapes && detectGlasses(blendshapes)) {
      eyeContactWeight = 0.4;
      headPostureWeight = 0.6;
    } else {
      eyeContactWeight = 0.6;
      headPostureWeight = 0.4;
    }

    if (frameCount >= ADAPTIVE_FRAMES) {
      const rotation = matrixToEulerAngles(results.facialTransformationMatrixes[0].data);
      const headScore = calculateHeadScore(rotation);
      const gaze = detectGazeDirection(landmarks);
      const eyeScore = calculateGazeScore(gaze) * calculateEyeClosureScore(
        calculateEAR(landmarks, LEFT_EYE_INDICES),
        calculateEAR(landmarks, RIGHT_EYE_INDICES)
      );

      updateScores(eyeScore, headScore, gaze);
      provideRealTimeFeedback(eyeScore, headScore);
    } else {
      calibrateForGlasses(landmarks);
    }
  } else {
    canvasCtx.restore();
    canvasCtx.fillStyle = 'red';
    canvasCtx.font = '60px Arial';
    canvasCtx.textAlign = 'left';
    canvasCtx.fillText('No Face Detected', 10, 60);
    
    if (scoringStarted) {
      updateScores(0, 0, null);
    }
  }
  
  canvasCtx.restore();
  window.requestAnimationFrame(predictWebcam);
}

// Micro-Movement Detection
let lastGazeDirection = { left: 'center', right: 'center' };
const MOVEMENT_PENALTY = 0.2;
let movementHistory = [];

function detectMicroMovements(currentGaze) {
  if (!currentGaze) return 0;
  let penalty = 0;
  
  movementHistory.push(currentGaze);
  if (movementHistory.length > 5) movementHistory.shift();

  if (currentGaze.left.horizontal !== lastGazeDirection.left.horizontal ||
      currentGaze.right.horizontal !== lastGazeDirection.right.horizontal) {
    penalty += MOVEMENT_PENALTY;
  }

  if (currentGaze.left.vertical !== lastGazeDirection.left.vertical ||
      currentGaze.right.vertical !== lastGazeDirection.right.vertical) {
    penalty += MOVEMENT_PENALTY;
  }

  lastGazeDirection = currentGaze;
  return Math.min(penalty, 0.4); // Max penalty of 40%
}