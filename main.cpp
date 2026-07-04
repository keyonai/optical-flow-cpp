// ─────────────────────────────────────────────────────────────
// Optical Flow Visualization  (C++ / LibTorch + OpenCV)
// ─────────────────────────────────────────────────────────────
// This program:
//   1. Captures frames from a webcam
//   2. Runs the RAFT-small deep optical flow model via LibTorch
//      every few frames (flow doesn't need per-frame updates)
//   3. Visualizes motion as:
//        - Hot-pink arrows showing direction + speed of movement
//        - HSV color map (hue = direction, brightness = speed)
//   4. Displays the annotated webcam feed and color map side by side

#include <opencv2/opencv.hpp>   // webcam, drawing, colorspace conversions
#include <torch/script.h>       // LibTorch TorchScript model loading
#include <iostream>
#include <string>
#include <cmath>                // std::sqrt, std::atan2
#include <algorithm>            // std::clamp

// ── Constants ──────────────────────────────────────────────
// Model input resolution (must be divisible by 8 for RAFT's feature pyramid)
const int FLOW_W = 480;
const int FLOW_H = 360;

// Run RAFT every N frames — it's slow on CPU, caching between runs keeps
// the display smooth while still showing meaningful motion
const int FLOW_INTERVAL = 4;


// ── Preprocessing ───────────────────────────────────────────
// Convert an OpenCV BGR frame to a LibTorch float tensor [1, 3, H, W]
// with pixel values normalized to [0, 1].
// RAFT applies its own internal normalization (mean 0.5, std 0.5).
torch::Tensor frameToTensor(const cv::Mat& frame) {

    // Step 1: BGR → RGB (RAFT expects RGB)
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    // Step 2: uint8 [0,255] → float32 [0,1]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Step 3: OpenCV HWC layout → LibTorch [1, C, H, W]
    // from_blob wraps existing memory — must .clone() so the tensor owns its data
    torch::Tensor t = torch::from_blob(
        float_img.data,
        {1, float_img.rows, float_img.cols, 3},
        torch::kFloat32
    ).clone();

    return t.permute({0, 3, 1, 2}).contiguous();   // [1, H, W, 3] → [1, 3, H, W]
}


// ── Post-processing ─────────────────────────────────────────
// Convert RAFT output tensor [1, 2, H, W] to an OpenCV float Mat [H, W, 2].
// The two channels are (horizontal displacement, vertical displacement) in pixels.
cv::Mat tensorToFlowMat(const torch::Tensor& flow_tensor) {
    torch::Tensor flow = flow_tensor
        .squeeze(0)              // [2, H, W]
        .permute({1, 2, 0})      // [H, W, 2]
        .contiguous();

    int H = (int)flow.size(0);
    int W = (int)flow.size(1);

    cv::Mat mat(H, W, CV_32FC2);
    std::memcpy(mat.data, flow.data_ptr<float>(), H * W * 2 * sizeof(float));
    return mat;
}


// ── Flow → Color visualization ──────────────────────────────
// Encodes the flow field as an HSV image:
//   Hue        = direction of motion (angle of the 2D vector)
//   Value      = speed (magnitude of the 2D vector)
//   Saturation = always 255 (vivid colors even for slow motion)
// Resizes the result to `outSize` to match the display frame.
cv::Mat flowToColorViz(const cv::Mat& flow, const cv::Size& outSize) {

    // Split into u (x-direction) and v (y-direction) channels
    std::vector<cv::Mat> channels(2);
    cv::split(flow, channels);
    cv::Mat& u = channels[0];
    cv::Mat& v = channels[1];

    // Compute angle [0°, 360°] and magnitude from (u, v)
    cv::Mat magnitude, angle;
    cv::cartToPolar(u, v, magnitude, angle, true);   // true = degrees

    // Normalize magnitude to [0, 255] for the Value channel
    cv::Mat mag_norm;
    cv::normalize(magnitude, mag_norm, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);

    // Build HSV image (OpenCV HSV: H ∈ [0,180], S ∈ [0,255], V ∈ [0,255])
    cv::Mat h_ch = angle / 2.0f;              // [0°,360°] → [0,180] for OpenCV
    cv::Mat s_ch = cv::Mat::ones(flow.size(), CV_32F) * 255.0f;
    cv::Mat v_ch = mag_norm;

    cv::Mat hsv_chs[] = {h_ch, s_ch, v_ch};
    cv::Mat hsv;
    cv::merge(hsv_chs, 3, hsv);

    // Convert to uint8 BGR for display
    cv::Mat hsv8, bgr, output;
    hsv.convertTo(hsv8, CV_8UC3);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    cv::resize(bgr, output, outSize);
    return output;
}


// ── Arrow overlay ────────────────────────────────────────────
// Draws sparse motion arrows on `frame` sampled every `step` pixels.
// Only draws arrows where motion exceeds the noise threshold.
// Arrows are hot pink — direction shows where pixels moved, length shows speed.
void drawFlowArrows(cv::Mat& frame, const cv::Mat& flow, int step = 24) {
    // Scale from flow resolution to display resolution
    float scaleX = (float)frame.cols / flow.cols;
    float scaleY = (float)frame.rows / flow.rows;

    for (int fy = 0; fy < flow.rows; fy += step) {
        for (int fx = 0; fx < flow.cols; fx += step) {

            cv::Vec2f f = flow.at<cv::Vec2f>(fy, fx);
            float dx  = f[0];
            float dy  = f[1];
            float mag = std::sqrt(dx * dx + dy * dy);

            // Skip near-zero motion (camera noise / still pixels)
            if (mag < 0.5f) continue;

            // Map flow grid position → display pixel position
            int px = (int)(fx * scaleX);
            int py = (int)(fy * scaleY);

            // Scale arrow length so small motions are still visible
            float arrowScale = 4.0f;
            int ex = std::clamp(px + (int)(dx * arrowScale), 0, frame.cols - 1);
            int ey = std::clamp(py + (int)(dy * arrowScale), 0, frame.rows - 1);

            // Color arrow to match the HSV flow map: hue = direction angle
            float angleDeg = std::atan2(dy, dx) * (180.0f / (float)M_PI);
            if (angleDeg < 0) angleDeg += 360.0f;
            uint8_t hue = (uint8_t)(angleDeg / 2.0f);   // OpenCV HSV hue: 0-180
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);

            cv::arrowedLine(frame,
                cv::Point(px, py),
                cv::Point(ex, ey),
                cv::Scalar(c[0], c[1], c[2]),
                1, cv::LINE_AA, 0, 0.35);
        }
    }
}


// ── Main ─────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // ── Load TorchScript model ──
    // Generated by export_raft.py — must be in the same folder as the exe
    std::cout << "Loading RAFT model..." << std::endl;
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("raft_small.pt");
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::cerr << "→ Run export_raft.py first to generate raft_small.pt" << std::endl;
        return -1;
    }
    std::cout << "Model loaded. Running on CPU." << std::endl;

    // ── Open webcam ──
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open webcam." << std::endl;
        return -1;
    }
    std::cout << "Running live — press 'q' or close window to quit." << std::endl;

    cv::Mat prevSmall, currSmall;   // frames resized to FLOW_W x FLOW_H for model input
    cv::Mat flowMat;                // cached flow field [FLOW_H, FLOW_W, CV_32FC2]
    cv::Mat flowViz;                // cached HSV color visualization
    int frameCount = 0;

    // Grab and resize the first frame to initialize prevSmall
    cv::Mat firstFrame;
    cap >> firstFrame;
    cv::resize(firstFrame, prevSmall, cv::Size(FLOW_W, FLOW_H));

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Keep a full-res copy for display; resize a copy for the model
        cv::Mat displayFrame = frame.clone();
        cv::resize(frame, currSmall, cv::Size(FLOW_W, FLOW_H));

        // ── Run RAFT every FLOW_INTERVAL frames ──
        if (frameCount % FLOW_INTERVAL == 0 || flowMat.empty()) {

            // Build input tensors from the two consecutive frames
            torch::Tensor t1 = frameToTensor(prevSmall);
            torch::Tensor t2 = frameToTensor(currSmall);

            // Run inference (no gradient needed)
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs = {t1, t2};
            torch::Tensor output = model.forward(inputs).toTensor();  // [1, 2, H, W]

            // Convert tensor → OpenCV Mat, then build color visualization
            flowMat = tensorToFlowMat(output);
            flowViz = flowToColorViz(flowMat, displayFrame.size());
        }

        // ── Draw motion arrows on display frame ──
        if (!flowMat.empty()) {
            drawFlowArrows(displayFrame, flowMat);
        }

        // ── Labels ──
        cv::putText(displayFrame, "Optical Flow (RAFT-small)",
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65,
            cv::Scalar(180, 105, 255), 2);
        cv::putText(displayFrame, "Arrows: direction + speed of motion",
            cv::Point(10, 58), cv::FONT_HERSHEY_SIMPLEX, 0.42,
            cv::Scalar(210, 210, 210), 1);

        if (!flowViz.empty()) {
            cv::putText(flowViz, "Flow Color Map",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                cv::Scalar(255, 255, 255), 2);
            cv::putText(flowViz, "Hue = direction  |  Brightness = speed",
                cv::Point(10, 58), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(210, 210, 210), 1);
        }

        // ── Display side by side ──
        if (!flowViz.empty()) {
            cv::Mat combined;
            cv::hconcat(displayFrame, flowViz, combined);
            cv::imshow("Optical Flow  |  q=quit", combined);
        } else {
            cv::imshow("Optical Flow  |  q=quit", displayFrame);
        }

        // Roll frames: current becomes previous
        prevSmall = currSmall.clone();
        frameCount++;

        // Quit on 'q', Esc, or window close
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (cv::getWindowProperty("Optical Flow  |  q=quit",
                                   cv::WND_PROP_VISIBLE) < 1) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
