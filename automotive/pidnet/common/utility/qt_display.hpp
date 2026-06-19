/**
 * @file qt_display.hpp
 * @brief Lightweight Qt5 display backend for pidnet (replaces OpenCV HighGUI).
 */

#ifndef DXAPP_QT_DISPLAY_HPP
#define DXAPP_QT_DISPLAY_HPP

#include <opencv2/core.hpp>

#include <string>

namespace dxapp {

void configureDisplay(bool full_screen, bool show_exit_button, int argc = 0, char** argv = nullptr);
void resetDisplayState();
void shutdownDisplay();
bool consumeExitButtonClick();
bool windowShouldClose(const std::string& winname = "Output");
void setDisplayFps(double fps);
void showOutput(const cv::Mat& frame);

}  // namespace dxapp

#endif  // DXAPP_QT_DISPLAY_HPP
