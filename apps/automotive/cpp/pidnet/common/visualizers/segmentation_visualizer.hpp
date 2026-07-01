/**
 * @file segmentation_visualizer.hpp
 * @brief Semantic and instance segmentation result visualizers
 */

#ifndef SEGMENTATION_VISUALIZER_HPP
#define SEGMENTATION_VISUALIZER_HPP

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "common/base/i_visualizer.hpp"

namespace dxapp {

inline constexpr const char* kDefaultSegmentationPalette = "aurora";

inline cv::Vec3b rgb(int r, int g, int b) {
    return cv::Vec3b(static_cast<unsigned char>(b),
                     static_cast<unsigned char>(g),
                     static_cast<unsigned char>(r));
}

inline std::string normalizeSegmentationPaletteName(std::string name) {
    for (char& ch : name) {
        if (ch == '_') {
            ch = '-';
        } else {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return name;
}

inline const std::vector<std::string>& getSegmentationPaletteNames() {
    static const std::vector<std::string> names = {
        "aurora", "neon", "pastel", "sunset", "graphite"
    };
    return names;
}

inline std::string getSegmentationPaletteOptionsText() {
    std::string options;
    for (const auto& name : getSegmentationPaletteNames()) {
        if (!options.empty()) options += ", ";
        options += name;
    }
    return options;
}

inline bool isValidSegmentationPalette(const std::string& palette_name) {
    const std::string normalized = normalizeSegmentationPaletteName(palette_name);
    const auto& names = getSegmentationPaletteNames();
    return std::find(names.begin(), names.end(), normalized) != names.end();
}

inline const std::vector<cv::Vec3b>& getSegmentationPalette(const std::string& palette_name) {
    // Color order follows Cityscapes classes:
    // road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
    // vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle, void.
    static const std::vector<cv::Vec3b> aurora = {
        rgb(78, 73, 138), rgb(96, 183, 199), rgb(56, 63, 78), rgb(126, 100, 172),
        rgb(166, 128, 151), rgb(185, 188, 195), rgb(255, 204, 77), rgb(255, 149, 89),
        rgb(45, 170, 116), rgb(124, 212, 154), rgb(96, 197, 230), rgb(255, 94, 122),
        rgb(255, 112, 67), rgb(62, 130, 255), rgb(39, 83, 184), rgb(0, 166, 217),
        rgb(34, 107, 148), rgb(178, 97, 255), rgb(54, 221, 187), rgb(0, 0, 0)
    };
    static const std::vector<cv::Vec3b> neon = {
        rgb(20, 20, 38), rgb(0, 240, 255), rgb(55, 60, 85), rgb(190, 80, 255),
        rgb(255, 0, 170), rgb(170, 190, 220), rgb(255, 242, 0), rgb(255, 122, 0),
        rgb(0, 255, 140), rgb(98, 255, 86), rgb(0, 168, 255), rgb(255, 35, 80),
        rgb(255, 76, 210), rgb(50, 120, 255), rgb(0, 67, 220), rgb(0, 218, 255),
        rgb(91, 255, 255), rgb(185, 50, 255), rgb(40, 255, 210), rgb(0, 0, 0)
    };
    static const std::vector<cv::Vec3b> pastel = {
        rgb(132, 121, 169), rgb(238, 173, 218), rgb(112, 116, 126), rgb(164, 151, 196),
        rgb(213, 171, 180), rgb(190, 194, 199), rgb(249, 206, 116), rgb(245, 180, 111),
        rgb(126, 194, 151), rgb(170, 221, 170), rgb(146, 199, 224), rgb(238, 122, 139),
        rgb(238, 141, 118), rgb(116, 154, 218), rgb(99, 119, 175), rgb(103, 176, 207),
        rgb(103, 147, 179), rgb(174, 134, 219), rgb(111, 203, 190), rgb(0, 0, 0)
    };
    static const std::vector<cv::Vec3b> sunset = {
        rgb(88, 63, 99), rgb(236, 102, 176), rgb(61, 68, 82), rgb(139, 91, 158),
        rgb(190, 118, 136), rgb(195, 185, 177), rgb(255, 213, 79), rgb(255, 154, 68),
        rgb(77, 163, 111), rgb(185, 194, 116), rgb(72, 169, 219), rgb(255, 82, 98),
        rgb(255, 111, 61), rgb(58, 134, 255), rgb(43, 88, 179), rgb(0, 159, 211),
        rgb(27, 118, 153), rgb(197, 79, 255), rgb(42, 212, 188), rgb(0, 0, 0)
    };
    static const std::vector<cv::Vec3b> graphite = {
        rgb(66, 72, 86), rgb(95, 132, 147), rgb(52, 56, 65), rgb(100, 90, 125),
        rgb(128, 105, 117), rgb(160, 166, 173), rgb(245, 197, 66), rgb(232, 142, 76),
        rgb(68, 145, 101), rgb(132, 174, 119), rgb(88, 153, 187), rgb(231, 76, 96),
        rgb(222, 92, 82), rgb(80, 122, 196), rgb(62, 88, 153), rgb(57, 147, 177),
        rgb(72, 117, 145), rgb(158, 106, 205), rgb(68, 189, 170), rgb(0, 0, 0)
    };

    const std::string normalized = normalizeSegmentationPaletteName(palette_name);
    if (normalized == "neon") return neon;
    if (normalized == "pastel") return pastel;
    if (normalized == "sunset") return sunset;
    if (normalized == "graphite") return graphite;
    return aurora;
}

/**
 * @brief Visualizer for semantic segmentation results
 */
class SemanticSegmentationVisualizer : public IVisualizer<SegmentationResult> {
public:
    SemanticSegmentationVisualizer() = default;

    /**
     * @brief Construct with a named color palette.
     * @param palette_name One of getSegmentationPaletteNames().
     * @param skip_bg When true, black/void pixels are not drawn.
     */
    explicit SemanticSegmentationVisualizer(std::string palette_name, bool skip_bg = false)
        : palette_name_(normalizeSegmentationPaletteName(std::move(palette_name))),
          skip_background_(skip_bg) {}

    /**
     * @brief Construct with a custom color palette.
     * @param palette  Custom color palette (index = class id).
     * @param skip_bg  When true, class 0 is treated as background and not drawn.
     */
    SemanticSegmentationVisualizer(std::vector<cv::Vec3b> palette, bool skip_bg = true)
        : custom_palette_(std::move(palette)), skip_background_(skip_bg) {}

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<SegmentationResult>& results,
                 const PreprocessContext& ctx) override {
        if (results.empty()) return frame.clone();
        
        const auto& seg = results[0];  // Typically one result for semantic seg
        const auto& palette = custom_palette_.empty()
            ? getSegmentationPalette(palette_name_)
            : custom_palette_;
        
        // Create colored mask
        cv::Mat colored_mask(seg.height, seg.width, CV_8UC3);
        auto* dst = colored_mask.ptr<cv::Vec3b>();
        int total = seg.height * seg.width;
        int psize = static_cast<int>(palette.size());
        for (int i = 0; i < total; ++i) {
            int class_id = seg.mask[i];
            int palette_idx = class_id >= 0 ? class_id % psize : psize - 1;
            const auto& c = palette[palette_idx];
            if (skip_background_ && c == cv::Vec3b(0, 0, 0))
                dst[i] = cv::Vec3b(0, 0, 0);
            else
                dst[i] = c;
        }
        
        // Remove letterbox padding before resize (matching original)
        cv::Mat unpadded_mask;
        if (ctx.pad_x > 0 || ctx.pad_y > 0) {
            int unpad_w = seg.width - 2 * ctx.pad_x;
            int unpad_h = seg.height - 2 * ctx.pad_y;
            // Safety check to prevent ROI assertion errors
            if (unpad_w > 0 && unpad_h > 0 &&
                ctx.pad_x + unpad_w <= colored_mask.cols &&
                ctx.pad_y + unpad_h <= colored_mask.rows) {
                cv::Rect crop_region(ctx.pad_x, ctx.pad_y, unpad_w, unpad_h);
                unpadded_mask = colored_mask(crop_region).clone();
            } else {
                unpadded_mask = colored_mask;
            }
        } else {
            unpadded_mask = colored_mask;
        }
        
        // Resize mask to match frame size. Use linear interpolation to reduce
        // blocky appearance when upscaling low-res segmentation outputs.
        cv::Mat resized_mask;
        cv::resize(unpadded_mask, resized_mask, frame.size(), 0, 0, cv::INTER_LINEAR);
        
        // Blend with original frame
        cv::Mat output;
        cv::addWeighted(frame, 1.0 - alpha_, resized_mask, alpha_, 0, output);
        
        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        (void)line_thickness;
        (void)font_scale;
        alpha_ = alpha;
    }

private:
    float alpha_{0.6f};
    std::string palette_name_{kDefaultSegmentationPalette};
    std::vector<cv::Vec3b> custom_palette_;
    bool skip_background_{false};
};

/**
 * @brief Visualizer for instance segmentation results
 */
class InstanceSegmentationVisualizer : public IVisualizer<InstanceSegmentationResult> {
public:
    InstanceSegmentationVisualizer(bool show_boxes = true) : show_boxes_(show_boxes) {}

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<InstanceSegmentationResult>& results,
                 const PreprocessContext& ctx) override {
        cv::Mat output = frame.clone();

        // Scale factor from original image space to display frame space.
        // ctx.original_width/height reflect the source image dimensions that the
        // postprocessor used when mapping boxes back from model space.  When
        // displayResize() has down-scaled the frame (e.g. 1920x1080 → 960x540)
        // the box coordinates must be scaled accordingly before drawing.
        float disp_scale = 1.0f;
        if (ctx.original_width > 0 && ctx.original_height > 0 &&
            (ctx.original_width > output.cols || ctx.original_height > output.rows)) {
            disp_scale = std::min(static_cast<float>(output.cols) / ctx.original_width,
                                  static_cast<float>(output.rows) / ctx.original_height);
        }
        const float x_off = (ctx.original_width > 0)
            ? (output.cols - ctx.original_width * disp_scale) / 2.0f : 0.0f;
        const float y_off = (ctx.original_height > 0)
            ? (output.rows - ctx.original_height * disp_scale) / 2.0f : 0.0f;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& inst = results[i];
            
            // Get color for this instance (use instance index for variety)
            const auto& colors = getSegmentationPalette(kDefaultSegmentationPalette);
            cv::Vec3b color = colors[i % colors.size()];
            cv::Scalar box_color(color[0], color[1], color[2]);
            
            // Draw mask overlay first (so boxes appear on top)
            if (!inst.mask.empty()) {
                cv::Mat binary_mask = convertToBinaryMask(inst.mask);

                if (binary_mask.size() != frame.size()) {
                    cv::resize(binary_mask, binary_mask, frame.size());
                }

                blendMaskRegion(binary_mask, color, alpha_, output);
            }

            // Draw bounding box and label
            if (show_boxes_ && inst.box.size() >= 4) {
                cv::Point pt1(static_cast<int>(inst.box[0] * disp_scale + x_off), static_cast<int>(inst.box[1] * disp_scale + y_off));
                cv::Point pt2(static_cast<int>(inst.box[2] * disp_scale + x_off), static_cast<int>(inst.box[3] * disp_scale + y_off));
                cv::rectangle(output, pt1, pt2, box_color, line_thickness_);
                
                std::string label = inst.class_name + ": " + 
                    std::to_string(static_cast<int>(inst.confidence * 100)) + "%";
                int baseline;
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale_, 1, &baseline);
                cv::putText(output, label, cv::Point(pt1.x, std::max(pt1.y - 5, 15)),
                           cv::FONT_HERSHEY_SIMPLEX, font_scale_, box_color, 1);
            }
        }

        return output;
    }

    void setParameters(int line_thickness = 2,
                       double font_scale = 0.5,
                       float alpha = 0.6f) override {
        line_thickness_ = line_thickness;
        font_scale_ = font_scale;
        alpha_ = alpha;
    }

private:
    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.4f};
    bool show_boxes_{true};

    /** Convert mask to binary uint8 format. */
    static cv::Mat convertToBinaryMask(const cv::Mat& mask) {
        cv::Mat binary_mask;
        if (mask.type() == CV_32FC1 || mask.type() == CV_64FC1) {
            mask.convertTo(binary_mask, CV_8UC1, 255.0);
            cv::threshold(binary_mask, binary_mask, 127, 255, cv::THRESH_BINARY);
        } else {
            mask.convertTo(binary_mask, CV_8UC1);
        }
        return binary_mask;
    }

    /** Blend a color into target where mask > 0. */
    static void blendMaskRegion(const cv::Mat& mask, const cv::Vec3b& color,
                                float alpha, cv::Mat& target) {
        cv::Mat color_mat(target.size(), CV_8UC3, cv::Scalar(color[0], color[1], color[2]));
        cv::Mat blended;
        cv::addWeighted(target, 1.0 - alpha, color_mat, alpha, 0, blended);
        blended.copyTo(target, mask);
    }
};

}  // namespace dxapp

#endif  // SEGMENTATION_VISUALIZER_HPP
